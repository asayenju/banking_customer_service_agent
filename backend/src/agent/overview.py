"""One tool for looking up the verified customer's own records, straight from the bank's database.

A specialist used to have to call several narrow read tools to find a transaction id or a card's last four digits,
and a model that skipped them asked the customer instead ("What is the transaction id?"), which no customer can
answer. This tool returns the customer's profile, accounts, cards, recent transactions and other records in one
call, in sections small enough to read. It only ever reads rows that belong to the customer, and it never returns a
PIN, a card security code or a card expiry.
"""

import json
from collections.abc import Mapping
from typing import Any

from agent.toolbox import ExtraTool
from agent.llm_tools import ToolSchema

SECTIONS = ("profile_and_accounts", "debit_cards", "bank_transactions", "credit_cards", "credit_transactions", "other_records")
NEVER = frozenset({"pin", "new_pin", "current_pin", "cvv", "expiration_date", "pin_attempts_remaining"})
ROWS = 25
# Tables searched for "other records": a row counts when any of its values is the customer's user id.
OTHER_TABLES = (
    "referrals",
    "debit_card_disputes",
    "transaction_disputes",
    "debit_card_orders",
    "credit_card_orders",
    "credit_card_applications",
    "credit_limit_increase_requests",
    "credit_card_account_flags",
    "cash_back_disputes",
    "payment_history",
    "human_transfer_requests",
)


def _clean(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in NEVER and v is not None}


def _mine(table: Any, user_id: str) -> list[dict[str, Any]]:
    return [_clean(r) for r in table.data.values() if isinstance(r, Mapping) and user_id in map(str, r.values())]


def section(db: Any, user_id: str, name: str) -> Any:
    accounts = [a for a in db.accounts.data.values() if a.get("user_id") == user_id]
    ids = {a["account_id"] for a in accounts}
    if name == "profile_and_accounts":
        profile = db.users.data.get(user_id, {})
        return {"profile": _clean(profile), "bank_accounts": [_clean(a) for a in accounts]}
    if name == "debit_cards":
        return [_clean(c) for c in db.debit_cards.data.values() if c.get("user_id") == user_id]
    if name == "bank_transactions":
        rows = [_clean(t) for t in db.bank_account_transaction_history.data.values() if t.get("account_id") in ids]
        return {"most_recent_last": rows[-ROWS * 2 :], "note": "ids, dates, merchants and amounts; use these ids, never ask the customer"}
    if name == "credit_cards":
        return [_clean(a) for a in db.credit_card_accounts.data.values() if a.get("user_id") == user_id]
    if name == "credit_transactions":
        rows = [_clean(t) for t in db.credit_card_transaction_history.data.values() if t.get("user_id") == user_id]
        return {"most_recent_last": rows[-ROWS * 2 :]}
    if name == "other_records":
        return {t: rows[-10:] for t in OTHER_TABLES if hasattr(db, t) and (rows := _mine(getattr(db, t), user_id))}
    return {"error": "unknown section"}


def overview_tool(db: Any, user_id: str) -> ExtraTool:
    def run(args: Mapping[str, Any]) -> str:
        name = str(args["section"])
        if name not in SECTIONS:
            return f"Error: section must be one of {', '.join(SECTIONS)}."
        return json.dumps(section(db, user_id, name), default=str, ensure_ascii=False, separators=(",", ":"))

    schema = ToolSchema(
        name="look_up_customer_records",
        description=(
            "Read the verified customer's own records straight from the bank's database: profile and bank accounts, "
            "debit cards (with last four digits), bank transactions (with ids), credit cards, credit card "
            "transactions, and other records (referrals, disputes, orders, applications). Use this to find ids, "
            "last four digits, dates and amounts yourself. Never ask the customer for something it returns."
        ),
        parameters={
            "type": "object",
            "properties": {"section": {"type": "string", "enum": list(SECTIONS)}},
            "required": ["section"],
            "additionalProperties": False,
        },
    )
    return ExtraTool(schema=schema, run=run)
