"""Lookups a credit specialist must use before it asks the customer anything about a charge.

``look_up_customer_records`` returns whole sections, which a long history can push past what the model is shown
(the Marriott charge of task 031 was cut off, and the agent told the customer it could not find it). These tools
search instead: by merchant, a rough amount, a date in any ordinary spelling, or a transaction id, and they say
what they found and what they left out. They read the verified customer's own rows only.
"""

import json
import re
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Optional

from agent.llm_tools import ToolSchema
from agent.toolbox import ExtraTool

MAX_ROWS = 15
_FORMATS = ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%B %d %Y", "%b %d %Y", "%B %d, %Y", "%b %d, %Y", "%m-%d-%Y", "%m-%d-%y")


def parse_when(text: str, today: Optional[date] = None) -> Optional[date]:
    """A date written the way people write it: 11/07/25, 11/07/2025, November 7th of 2025, 2025-11-07."""
    cleaned = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", text.strip().lower())
    cleaned = re.sub(r"\bof\b|,", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    for fmt in _FORMATS:
        try:
            return datetime.strptime(cleaned.title() if "%B" in fmt or "%b" in fmt else cleaned, fmt).date()
        except ValueError:
            continue
    if today is not None:
        for fmt in ("%B %d", "%b %d", "%m/%d"):
            try:
                found = datetime.strptime(cleaned.title() if "%B" in fmt or "%b" in fmt else cleaned, fmt).date()
            except ValueError:
                continue
            return found.replace(year=today.year)
    return None


def _plain(text: Any) -> str:
    return "".join(c for c in str(text).lower() if c.isalnum())


# Words that say what kind of place it is, not which one: "a Marriott hotel" is about Marriott.
_GENERIC = frozenset(
    "a an the and of for from at my on in to charge charges card credit payment purchase transaction transactions "
    "hotel hotels store shop inc llc co ads airlines airline market markets foods food farmers group services service "
    "online corp company bank pay".split()
)
SIMILAR = 0.8


def merchant_fits(asked: str, merchant: Any) -> bool:
    """Does the merchant the customer named mean this charge's merchant? Misspellings count ("mariot hotel" for
    "Marriott Hotels"), and so do extra words ("a Marriott hotel"). An empty ask fits every merchant."""
    from difflib import SequenceMatcher

    if not asked.strip():
        return True
    name = str(merchant)
    squashed = _plain(name)
    words = re.findall(r"[a-z0-9]+", name.lower())
    wanted = re.findall(r"[a-z0-9]+", asked.lower())
    keep = [w for w in wanted if w not in _GENERIC] or wanted
    for word in keep:
        if len(word) < 3:
            continue
        if word in squashed or any(SequenceMatcher(None, word, w).ratio() >= SIMILAR for w in words):
            return True
    whole = _plain(asked)
    return bool(whole) and (whole in squashed or squashed in whole)


def _money(text: Any) -> Optional[float]:
    found = re.search(r"\d[\d,]*(?:\.\d+)?", str(text))
    return float(found.group(0).replace(",", "")) if found else None


def _row(t: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "transaction_id": t.get("transaction_id"),
        "date": t.get("transaction_date"),
        "merchant": t.get("merchant_name"),
        "amount": t.get("transaction_amount"),
        "card": t.get("credit_card_type"),
        "status": t.get("status"),
    }


def _card_types(db: Any, user_id: str, card: str) -> Optional[set[str]]:
    """The card types a customer's words mean: an account id, or part of a card's name ('silver'). None if none match."""
    accounts = [a for a in db.credit_card_accounts.data.values() if a.get("user_id") == user_id]
    named = card.strip()
    by_id = [a for a in accounts if a.get("account_id") == named]
    if by_id:
        return {str(by_id[0].get("card_type"))}
    words = [w for w in re.findall(r"[a-z0-9]+", named.lower()) if w not in ("card", "credit", "my", "the", "rewards")]
    found = {
        str(a.get("card_type"))
        for a in accounts
        if words and all(w in _plain(a.get("card_type")) for w in words)
    }
    return found or None


def find_transactions(
    db: Any, user_id: str, today: date, merchant: str, amount: str, on_or_after: str, on_or_before: str, transaction_id: str,
    card: str = "",
) -> dict[str, Any]:
    mine = [t for t in db.credit_card_transaction_history.data.values() if t.get("user_id") == user_id]
    if card.strip():
        types = _card_types(db, user_id, card)
        if types is None:
            return {"matches": [], "tip": "that card is not one of this customer's; use list_credit_cards to see theirs"}
        mine = [t for t in mine if str(t.get("credit_card_type")) in types]
    if transaction_id.strip():
        exact = [t for t in mine if str(t.get("transaction_id")) == transaction_id.strip()]
        return {"matches": [_row(t) for t in exact]} if exact else {"matches": [], "tip": "no such transaction id on this customer's cards"}
    start = parse_when(on_or_after, today) if on_or_after.strip() else None
    end = parse_when(on_or_before, today) if on_or_before.strip() else None
    wanted = _money(amount) if amount.strip() else None
    exact_amount = wanted is not None and "." in amount

    def fits_merchant(t: Mapping[str, Any]) -> bool:
        return merchant_fits(merchant, t.get("merchant_name"))

    def fits_amount(t: Mapping[str, Any]) -> bool:
        if wanted is None:
            return True
        have = _money(t.get("transaction_amount")) or 0.0
        return abs(have - wanted) <= 0.01 if exact_amount else abs(have - wanted) <= max(1.0, 0.1 * wanted)

    def fits_dates(t: Mapping[str, Any]) -> bool:
        when = parse_when(str(t.get("transaction_date")))
        if when is None:
            return start is None and end is None
        return (start is None or when >= start) and (end is None or when <= end)

    hits = [t for t in mine if fits_merchant(t) and fits_amount(t) and fits_dates(t)]
    out: dict[str, Any] = {"matches": [_row(t) for t in hits[:MAX_ROWS]]}
    if not hits:
        loose = [t for t in mine if fits_merchant(t)] if merchant.strip() else []
        out["tip"] = "nothing fits every filter; the rows below fit the merchant alone, so check the dates and amount against them"
        out["merchant_only"] = [_row(t) for t in loose[:MAX_ROWS]]
    return out


def list_cards(db: Any, user_id: str) -> list[dict[str, Any]]:
    keep = ("account_id", "card_type", "card_last_4_digits", "status", "date_of_account_open", "current_balance", "credit_limit")
    return [{k: a[k] for k in keep if a.get(k) is not None} for a in db.credit_card_accounts.data.values() if a.get("user_id") == user_id]


def dispute_history(db: Any, user_id: str) -> list[dict[str, Any]]:
    keep = ("dispute_id", "transaction_id", "dispute_reason", "status", "submitted_at", "eligible_for_provisional_credit")
    rows = getattr(db, "transaction_disputes").data.values()
    return [{k: r[k] for k in keep if r.get(k) is not None} for r in rows if r.get("user_id") == user_id]


def _schema(name: str, description: str, properties: dict[str, Any]) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=description,
        parameters={"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False},
    )


_TEXT = {"type": "string"}
FIND = "find_credit_transactions"
CARDS = "list_credit_cards"
DISPUTES = "credit_dispute_history"
NAMES = frozenset({FIND, CARDS, DISPUTES})


def credit_lookups(db: Any, user_id: str, today: date) -> list[ExtraTool]:
    def find(args: Mapping[str, Any]) -> str:
        found = find_transactions(
            db, user_id, today, str(args["merchant"]), str(args["amount"]), str(args["on_or_after"]),
            str(args["on_or_before"]), str(args["transaction_id"]), str(args["card"]),
        )
        return json.dumps(found, ensure_ascii=False, separators=(",", ":"))

    return [
        ExtraTool(
            _schema(
                FIND,
                "Search the verified customer's credit card transactions. Give whatever the customer said and leave the "
                "rest as an empty string: the card (its account id or part of its name, like 'silver'), the merchant "
                "(any part of the name), a rough amount, a date range in any ordinary spelling (11/07/25, November 7th), "
                "or a transaction id. Use this before you ask the customer anything about a charge. Rows show which "
                "card each charge was on.",
                {
                    "card": {**_TEXT, "description": "Part of the card's name ('silver'), its account id, or ''."},
                    "merchant": {**_TEXT, "description": "Part of the merchant's name, or ''."},
                    "amount": {**_TEXT, "description": "The amount, rough is fine ('around 160'), or ''."},
                    "on_or_after": {**_TEXT, "description": "Earliest date, or ''."},
                    "on_or_before": {**_TEXT, "description": "Latest date, or ''."},
                    "transaction_id": {**_TEXT, "description": "A transaction id the customer gave, or ''."},
                },
            ),
            find,
        ),
        ExtraTool(
            _schema(CARDS, "List the verified customer's credit cards: account id, type, last four digits, status, balance.", {}),
            lambda args: json.dumps(list_cards(db, user_id), ensure_ascii=False, separators=(",", ":")),
        ),
        ExtraTool(
            _schema(DISPUTES, "List the disputes this customer has already filed on credit card charges.", {}),
            lambda args: json.dumps(dispute_history(db, user_id), ensure_ascii=False, separators=(",", ":")),
        ),
    ]
