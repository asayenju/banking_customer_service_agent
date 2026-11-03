"""Which of the customer's cards does a request mean? Settled in code, never by a model.

Shared by the debit specialist and the direct freeze/unfreeze recipes. A customer with one
card needs no question; with several they get a list written here (last four digits, status
and account, never an id) and may answer with the last four digits, which code maps back to
the card. Another customer's card never appears in a list and never resolves.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.types import Intent, Question

Card = Mapping[str, Any]


def owned_cards(db: Any, user_id: str, *, statuses: Optional[Sequence[str]] = None) -> list[Card]:
    """The customer's cards, oldest id first; closed ones left out unless asked for by status."""
    wanted = set(statuses) if statuses is not None else None
    cards = (
        c
        for c in db.debit_cards.data.values()
        if c.get("user_id") == user_id
        and (c.get("status") in wanted if wanted is not None else c.get("status") != "CLOSED")
    )
    return sorted(cards, key=lambda c: str(c.get("card_id")))


def describe(db: Any, card: Card) -> str:
    account = db.accounts.data.get(str(card.get("account_id")), {})
    where = f", on your {account['level']}" if account.get("level") else ""
    return f"the card ending {card.get('last_4_digits')} ({str(card.get('status')).lower()}{where})"


def which_card(intent: Intent, db: Any, cards: Sequence[Card]) -> RouteOutcome:
    """The list the customer chooses from."""
    listing = "\n".join(f"- {describe(db, c)}" for c in cards)
    return RouteOutcome(
        intent_id=intent.id,
        status=OutcomeStatus.NEEDS_INFO,
        questions=(Question(intent_id=intent.id, slot="card_id", text=f"Which card do you mean?\n{listing}"),),
    )


def resolve(named: str, cards: Sequence[Card]) -> Optional[Card]:
    """The one card a name means: its id, or its last four digits if only one card has them."""
    named = named.strip()
    if not named:
        return None
    by_id = [c for c in cards if c.get("card_id") == named]
    if by_id:
        return by_id[0]
    if named.isdigit() and len(named) == 4:
        by_last_four = [c for c in cards if c.get("last_4_digits") == named]
        if len(by_last_four) == 1:
            return by_last_four[0]
    return None
