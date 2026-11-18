"""Things the customer asked for that undo each other, said before the first of them is done.

Doing what the customer asked first can make the other thing impossible (ordering a replacement closes the card whose limit
they also want raised). The order is the customer's to choose, and they may have chosen it on purpose ("fraud first"). So
nothing here reorders anything: it only says, in the reply that shows the plan, what will follow from it.
"""

from collections.abc import Iterable

from orchestrator.types import IntentKind as K

_NOTES: tuple[tuple[K, K, str], ...] = (
    (
        K.ORDER_REPLACEMENT_CARD,
        K.CREDIT_LIMIT_REQUEST,
        "Heads-up: ordering the replacement closes your current card, so a limit increase can't be requested on it afterwards. "
        "If a higher limit on this card matters to you, say so before you confirm.",
    ),
    (
        K.CLOSE_CARD,
        K.CREDIT_LIMIT_REQUEST,
        "Heads-up: once the card is closed its limit can't be raised. If you want the increase first, say so before you confirm.",
    ),
    (
        K.DISPUTE_TRANSACTION,
        K.CREDIT_LIMIT_REQUEST,
        "Heads-up: a dispute that is still open can stop a limit increase from being approved, so the increase may be declined "
        "while the dispute is under review.",
    ),
)


def heads_up(planned: Iterable[K], still_open: Iterable[K]) -> list[str]:
    """The notes for the plans being shown, given the requests that are still open beside them. Each is said once."""
    planned_kinds, open_kinds = set(planned), set(still_open)
    return [text for first, later, text in _NOTES if first in planned_kinds and later in open_kinds]
