"""Which accounts can this customer be offered today? Code decides the set; a model only ranks inside it.

Many of the account tasks are "which account should I open?", and the right answer turns on things a
model gets wrong when it reads 700 documents: whether the customer is eligible, which promotions are
still running on the date, which savings account pairs with the checking account they already hold.
This module settles those in code:

  1. every class of the product is tried against the same open-account rules the validator enforces
     (caps, balances, ages, 14 days, a name the customer already has), so a class that would be refused
     is never offered;
  2. a promotion counts only between its dates, and where two windows overlap the one that started later
     is the current one (documents 013 and 014, and 015 and 016, overlap from 11/01 to 11/12);
  3. each candidate carries the documents it rests on, so an answer can cite them.

The model is handed the candidates and the reasons the others are out. A class that is not a candidate
cannot become a plan (``choose``). Choosing between candidates, given what the customer wants, is the
model's job; the tier-1 evidence replays the reference answer and the live probe measures the ranking.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from agent import bank_accounts as accounts
from agent.bank_accounts import Product
from agent.specialist import ValidationContext
from agent.specialists import accounts_rules as rules
from agent.specialists.credit_rules import day
from agent.workflows import account_products as products

_TOOL_TYPE = {
    Product.CHECKING: "checking",
    Product.SAVINGS: "savings",
    Product.BUSINESS_CHECKING: "business_checking",
    Product.BUSINESS_SAVINGS: "business_savings",
}


@dataclass(frozen=True)
class Candidate:
    product: Product
    name: str  # the exact class string
    page: str
    documents: tuple[str, ...]  # real document ids of the product page, empty if the index was not given
    promotion: Optional[str]  # which running promotion names it, and where it comes in the order
    promotion_rank: Optional[int]  # 1 is first
    boosts_with: tuple[str, ...]  # the customer's own accounts this one pairs with for a higher savings APY


@dataclass(frozen=True)
class Candidates:
    offered: tuple[Candidate, ...]
    not_offered: tuple[tuple[str, str], ...]  # (class, why), in the bank's order


def running_promotion(product: Product, today: date) -> Optional[products.Promotion]:
    """The promotion in force for a product on a date; of two overlapping ones, the one that started later."""
    live = [
        p
        for p in products.PROMOTIONS
        if p.product is product
        and (start := day(p.start)) is not None
        and (end := day(p.end)) is not None
        and start <= today <= end
    ]
    return max(live, key=lambda p: day(p.start) or date.min) if live else None


def _documents(page: str, doc_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(i for i in doc_ids if i == page or i.startswith(page + "_")))


def _boosts(product: Product, name: str, held: list[accounts.BankAccount]) -> tuple[str, ...]:
    """Of the accounts the customer holds, those that pair with this class for an APY boost (document 012)."""
    if product is Product.SAVINGS:
        mine = {a.name for a in held if a.product is Product.CHECKING and a.is_open}
        return tuple(sorted(c for c, s in products.APY_PAIRINGS if s == name and c in mine))
    if product is Product.CHECKING:
        mine = {a.name for a in held if a.product is Product.SAVINGS and a.is_open}
        return tuple(sorted(s for c, s in products.APY_PAIRINGS if c == name and s in mine))
    return ()


def candidates(
    db: Any, user_id: str, today: date, product: Product, *, doc_ids: Sequence[str] = ()
) -> Candidates:
    """Every class of the product, split into those the customer can be offered and those they cannot, with why."""
    validation = ValidationContext(db, user_id, today)
    held = accounts.accounts_of(db, user_id)
    promo = running_promotion(product, today)
    offered: list[Candidate] = []
    out: list[tuple[str, str]] = []
    for name in products.CLASSES[product]:
        why = rules.open_account(
            validation, {"user_id": user_id, "account_type": _TOOL_TYPE[product], "account_class": name}
        )
        if why is not None:
            out.append((name, why))
            continue
        rank = promo.priority.index(name) + 1 if promo is not None and name in promo.priority else None
        page = products.PAGE[(product, name)]
        offered.append(
            Candidate(
                product=product,
                name=name,
                page=page,
                documents=_documents(page, doc_ids),
                promotion=f"{promo.label} puts it number {rank} ({promo.document})" if promo is not None and rank else None,
                promotion_rank=rank,
                boosts_with=_boosts(product, name, held),
            )
        )
    return Candidates(tuple(offered), tuple(out))


def choose(
    db: Any, user_id: str, today: date, product: Product, name: str, *, doc_ids: Sequence[str] = ()
) -> Candidate | str:
    """The candidate with this exact name, or the reason it cannot be opened. A name that is not a class of
    the product at all reads like one the bank does not offer."""
    found = candidates(db, user_id, today, product, doc_ids=doc_ids)
    for candidate in found.offered:
        if candidate.name == name:
            return candidate
    for other, why in found.not_offered:
        if other == name:
            return why
    return f"{name!r} is not the exact name of a {product.value} account the bank offers"


def render(found: Candidates, today: date) -> str:
    """The set as text for a model: each candidate with its promotion and pairings, then what is out and why."""
    lines = [f"Accounts this customer can open today ({today:%m/%d/%Y}):"]
    for c in found.offered:
        extras = [x for x in (c.promotion, f"raises the savings rate with your {', '.join(c.boosts_with)}" if c.boosts_with else None) if x]
        lines.append(f"- {c.name}" + (f" ({'; '.join(extras)})" if extras else ""))
    if not found.offered:
        lines.append("- none")
    if found.not_offered:
        lines.append("Not available to this customer:")
        lines.extend(f"- {n}: {why}" for n, why in found.not_offered)
    return "\n".join(lines)
