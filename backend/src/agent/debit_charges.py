"""A checking account's recent charges as lines of text with a "Dispute" button each, built from the bank's own record.

Used by the plain history read (``recipes/direct.py``) and by the debit specialist when a dispute is missing which charge
is meant. A button means exactly the charge on its line: the request is made from the record, not from the words.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from orchestrator.types import Choice, IntentKind


def _money(value: Any) -> str:
    try:
        amount = float(str(value).replace("$", "").replace(",", ""))
    except ValueError:
        return str(value)
    return f"-${abs(amount):,.2f}" if amount < 0 else f"${amount:,.2f}"


def label_of(row: Mapping[str, Any]) -> str:
    # "level" is set only when charges of several accounts are shown together, so each line says whose it is.
    where = f"  ({row['level']})" if row.get("level") else ""
    return f"{str(row.get('date'))[:10]}  {_money(row.get('amount'))}  {row.get('description', '')}{where}"


def dispute_choices(rows: Sequence[Mapping[str, Any]], account_id: str = "") -> tuple[Choice, ...]:
    """One Dispute button per row that has an id; the account is in the offer so no one is asked which card it was."""
    return tuple(
        Choice(
            id=f"dispute:{row.get('transaction_id')}",
            label=label_of(row),
            action="Dispute",
            kind=IntentKind.DISPUTE_TRANSACTION,
            summary=f"dispute the {_money(row.get('amount'))} {row.get('description', '')} charge on {str(row.get('date'))[:10]}",
            slots={
                "transaction_id": str(row.get("transaction_id")),
                "account_id": account_id or str(row.get("account_id")),
                "merchant": str(row.get("description", "")),
                "amount": _money(row.get("amount")),
                "date": str(row.get("date"))[:10],
            },
        )
        for row in rows
        if row.get("transaction_id")
    )
