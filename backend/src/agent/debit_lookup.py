"""Search tools for a debit specialist: a card's transactions, found without asking the customer for ids.

The bank stores a debit transaction against a checking account, not a card, and a card belongs to one account. So
"this card's transactions" means the transactions of the account the card is on. These tools take a card the way a
customer names it (its id, its last four digits, or the account it is on: "my Evergreen card") and say which account
and card the rows came from. They read the verified customer's own rows only.
"""

import json
import re
from collections.abc import Mapping
from datetime import date
from typing import Any, Optional

from agent.cards import owned_cards, resolve
from agent.credit_lookup import MAX_ROWS, _money, _plain, _schema, merchant_fits, parse_when
from agent.toolbox import ExtraTool

FIND = "find_debit_transactions"
CARDS = "list_debit_cards"
NAMES = frozenset({FIND, CARDS})
_TEXT = {"type": "string"}


def _accounts_for(db: Any, user_id: str, card: str) -> tuple[Optional[list[str]], str]:
    """(account ids, how they were chosen). None means the card named matched nothing the customer has."""
    mine = {a["account_id"]: a for a in db.accounts.data.values() if a.get("user_id") == user_id}
    if not card.strip():
        return list(mine), "every account"
    named = card.strip()
    if named in mine:
        return [named], "that account"
    cards = owned_cards(db, user_id, statuses=("ACTIVE", "FROZEN", "PENDING", "CLOSED"))
    found = resolve(named, cards)
    if found is not None:
        return [str(found["account_id"])], "that card's account"
    wanted = "".join(w for w in re.findall(r"[a-z0-9]+", named.lower()) if w not in ("my", "the", "a", "card", "cards", "account"))

    def level(a: Mapping[str, Any]) -> str:
        return _plain(a.get("level")).replace("account", "")

    # "Green" is the Green Account, not also the Light Green one; a part of a name is used only when nothing is exact.
    exact = [i for i, a in mine.items() if wanted and level(a) == wanted]
    partial = [i for i, a in mine.items() if wanted and wanted in level(a)]
    return (exact or partial or None), "accounts whose level matches"


def _rows(db: Any, user_id: str, accounts: list[str]) -> list[dict[str, Any]]:
    cards = {c["account_id"]: c for c in db.debit_cards.data.values() if c.get("user_id") == user_id and c.get("status") != "CLOSED"}
    levels = {a["account_id"]: a.get("level") for a in db.accounts.data.values() if a.get("user_id") == user_id}
    out = []
    for t in db.bank_account_transaction_history.data.values():
        if t.get("account_id") not in accounts:
            continue
        card = cards.get(t["account_id"], {})
        out.append(
            {
                "transaction_id": t.get("transaction_id"),
                "date": t.get("date"),
                "description": t.get("description"),
                "amount": t.get("amount"),
                "type": t.get("type"),
                "status": t.get("status"),
                "account": levels.get(t["account_id"]),
                "card_last_4": card.get("last_4_digits"),
            }
        )
    return out


def find_transactions(
    db: Any, user_id: str, today: date, card: str, merchant: str, amount: str, on_or_after: str, on_or_before: str,
    transaction_id: str,
) -> dict[str, Any]:
    accounts, how = _accounts_for(db, user_id, card)
    if accounts is None:
        return {"matches": [], "tip": "that card is not one of this customer's; use list_debit_cards to see theirs"}
    rows = _rows(db, user_id, accounts)
    if transaction_id.strip():
        exact = [r for r in rows if str(r["transaction_id"]) == transaction_id.strip()]
        return {"matches": exact} if exact else {"matches": [], "tip": "no such transaction id on this customer's accounts"}
    start = parse_when(on_or_after, today) if on_or_after.strip() else None
    end = parse_when(on_or_before, today) if on_or_before.strip() else None
    wanted = _money(amount) if amount.strip() else None
    exact_amount = wanted is not None and "." in amount

    def fits_amount(r: Mapping[str, Any]) -> bool:
        if wanted is None:
            return True
        have = abs(_money(r["amount"]) or 0.0)
        return abs(have - wanted) <= 0.01 if exact_amount else abs(have - wanted) <= max(1.0, 0.1 * wanted)

    def fits_dates(r: Mapping[str, Any]) -> bool:
        when = parse_when(str(r["date"]).split()[0]) if r["date"] else None
        if when is None:
            return start is None and end is None
        return (start is None or when >= start) and (end is None or when <= end)

    hits = [r for r in rows if merchant_fits(merchant, r["description"]) and fits_amount(r) and fits_dates(r)]
    out: dict[str, Any] = {"searched": how, "matches": hits[:MAX_ROWS]}
    if not hits:
        loose = [r for r in rows if merchant_fits(merchant, r["description"])] if merchant.strip() else []
        out["tip"] = "nothing fits every filter; the rows below fit the merchant alone, so check the dates and amount against them"
        out["merchant_only"] = loose[:MAX_ROWS]
    return out


def list_cards(db: Any, user_id: str) -> list[dict[str, Any]]:
    levels = {a["account_id"]: a.get("level") for a in db.accounts.data.values() if a.get("user_id") == user_id}
    keep = ("card_id", "last_4_digits", "status", "issue_reason")
    return [
        {**{k: c[k] for k in keep if c.get(k) is not None}, "account": levels.get(c.get("account_id")), "account_id": c.get("account_id")}
        for c in db.debit_cards.data.values()
        if c.get("user_id") == user_id
    ]


def debit_lookups(db: Any, user_id: str, today: date) -> list[ExtraTool]:
    def find(args: Mapping[str, Any]) -> str:
        found = find_transactions(
            db, user_id, today, str(args["card"]), str(args["merchant"]), str(args["amount"]),
            str(args["on_or_after"]), str(args["on_or_before"]), str(args["transaction_id"]),
        )
        return json.dumps(found, ensure_ascii=False, separators=(",", ":"))

    return [
        ExtraTool(
            _schema(
                FIND,
                "Search the verified customer's debit card transactions. Give the card the way the customer named it "
                "(its id, its last four digits, or the account it is on, like 'Evergreen') or '' for all, and whatever "
                "else they said: the merchant (any part of the name), a rough amount, a date range in any ordinary "
                "spelling, or a transaction id. Leave the rest as ''. Use this before you ask the customer anything about "
                "a charge. Rows show the account and the card's last four digits they came from.",
                {
                    "card": {**_TEXT, "description": "A card id, its last four digits, or its account ('Evergreen'), or ''."},
                    "merchant": {**_TEXT, "description": "Part of the merchant's name, or ''."},
                    "amount": {**_TEXT, "description": "The amount, rough is fine, or ''."},
                    "on_or_after": {**_TEXT, "description": "Earliest date, or ''."},
                    "on_or_before": {**_TEXT, "description": "Latest date, or ''."},
                    "transaction_id": {**_TEXT, "description": "A transaction id the customer gave, or ''."},
                },
            ),
            find,
        ),
        ExtraTool(
            _schema(CARDS, "List the verified customer's debit cards: card id, last four digits, status and the account each is on.", {}),
            lambda args: json.dumps(list_cards(db, user_id), ensure_ascii=False, separators=(",", ":")),
        ),
    ]
