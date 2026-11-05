"""Which of the customer's bank accounts does a request mean? Settled in code, never by a model.

Same rules as ``agent/cards.py``: a customer with one account needs no question; with several they
get a list written here (name, type and opening date, never an id) and may answer with the
account's name ("Bronze savings"), its number in the list, or an id. Another customer's account
never appears in a list and never resolves, and reads exactly like one that does not exist.

The bank stores accounts in two layouts. The ones it starts with have ``class`` ("checking",
"saving", "business_checking") and ``level`` ("Blue Account"). An account opened by
``open_bank_account_4821`` has ``account_type`` ("savings") and ``account_class`` instead, so
every tool that reads ``class`` cannot see it (docs/accounts-policy-decisions.md, section 2).
``BankAccount`` is one reading of both.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any, Optional

from agent.specialists.credit_rules import day, money


class Product(StrEnum):
    CHECKING = "checking"
    SAVINGS = "savings"
    BUSINESS_CHECKING = "business checking"
    BUSINESS_SAVINGS = "business savings"


_PRODUCT_OF: Mapping[str, Product] = {
    "checking": Product.CHECKING,
    "personal_checking": Product.CHECKING,
    "savings": Product.SAVINGS,
    "saving": Product.SAVINGS,  # the bank's own data writes it this way
    "personal_savings": Product.SAVINGS,
    "business_checking": Product.BUSINESS_CHECKING,
    "business_savings": Product.BUSINESS_SAVINGS,
    "business_saving": Product.BUSINESS_SAVINGS,
}
OPEN_STATUSES = frozenset({"OPEN", "ACTIVE"})


@dataclass(frozen=True)
class BankAccount:
    account_id: str
    user_id: str
    product: Product
    name: str  # the exact class string, e.g. "Blue Account" or "Navy Blue"
    status: str
    opened: Optional[date]
    balance: Optional[float]  # None when the record has no readable balance

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES

    @property
    def is_checking(self) -> bool:
        return self.product in (Product.CHECKING, Product.BUSINESS_CHECKING)

    def age_days(self, today: date) -> Optional[int]:
        return (today - self.opened).days if self.opened is not None else None


def read(row: Mapping[str, Any]) -> Optional[BankAccount]:
    """One account row as a BankAccount, in either layout; None if it is not a bank account we understand."""
    kind = str(row.get("account_type") or row.get("class") or "").strip().lower()
    product = _PRODUCT_OF.get(kind)
    if product is None or not row.get("account_id") or not row.get("user_id"):
        return None
    name = str(row.get("account_class") or row.get("level") or "").strip()
    holdings = row.get("current_holdings", row.get("balance"))
    return BankAccount(
        account_id=str(row["account_id"]),
        user_id=str(row["user_id"]),
        product=product,
        name=name,
        status=str(row.get("status") or "").upper(),
        opened=day(row.get("date_opened")),
        balance=money(holdings) if holdings is not None else None,
    )


def accounts_of(db: Any, user_id: str, *, include_closed: bool = False) -> list[BankAccount]:
    """The customer's bank accounts, oldest first (then by id), so a numbered list is stable."""
    found = []
    for row in db.accounts.data.values():
        if not isinstance(row, Mapping) or row.get("user_id") != user_id:
            continue
        account = read(row)
        if account is not None and (include_closed or account.status != "CLOSED"):
            found.append(account)
    return sorted(found, key=lambda a: (a.opened or date.min, a.account_id))


def get(db: Any, user_id: str, account_id: Any) -> Optional[BankAccount]:
    """The customer's own account with this id (closed ones included); None for anyone else's or an unknown id."""
    row = db.accounts.data.get(str(account_id))
    if not isinstance(row, Mapping) or row.get("user_id") != user_id:
        return None
    return read(row)


def describe(account: BankAccount) -> str:
    when = f", opened {account.opened:%m/%d/%Y}" if account.opened else ""
    return f"your {account.name} ({account.product.value}){when}"


def label(account: BankAccount) -> str:
    """How a plan names an account to the customer: its name and kind, never its id."""
    return f"your {account.name} ({account.product.value})"


_FILLER = {"account", "accounts", "my", "the", "your", "a", "an"}


def _words(text: str) -> set[str]:
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return {"savings" if w == "saving" else w for w in cleaned.split() if w not in _FILLER}


def resolve(named: str, accounts: Sequence[BankAccount]) -> Optional[BankAccount]:
    """The one account a name means: its id, its number in the list, or its name and type
    ("Bronze savings", "the Blue Account") when that picks out exactly one. None when it matches
    none or several: the caller then asks."""
    named = str(named).strip()
    if not named:
        return None
    by_id = [a for a in accounts if a.account_id == named]
    if by_id:
        return by_id[0]
    # A number in the list we showed. Only a plain one: "01" is an account id (often another customer's),
    # and must never be read as "the first one".
    if named.isdigit() and named == str(int(named)) and len(named) <= 2 and 1 <= int(named) <= len(accounts):
        return accounts[int(named) - 1]
    wanted = _words(named)
    if not wanted:
        return None
    found = [a for a in accounts if wanted <= _words(a.name) | _words(a.product.value)]
    return found[0] if len(found) == 1 else None
