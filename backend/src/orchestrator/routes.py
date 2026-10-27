"""Where each kind of intent goes once it has been cleared to run.

DIRECT is a fixed tool sequence with no policy judgment (it may be a lookup
followed by an action). SUB_AGENT needs policy reasoning or branching, for
example three activation tools that depend on why the card was issued.
HAND_OFF is a human. tests/orchestrator/test_routes.py fails if an IntentKind
has no route.
"""

from enum import StrEnum

from orchestrator.types import IntentKind

K = IntentKind


class Route(StrEnum):
    DIRECT = "direct"
    SUB_AGENT = "sub_agent"
    HAND_OFF = "hand_off"


D, S, H = Route.DIRECT, Route.SUB_AGENT, Route.HAND_OFF

ROUTES: dict[IntentKind, Route] = {
    K.FREEZE_CARD: D,
    K.UNFREEZE_CARD: D,
    K.ACCOUNT_BALANCE_OR_HISTORY: D,
    K.LOOKUP_INFO: D,
    K.UPDATE_PROFILE: D,
    K.UNCLEAR_OR_GREETING: D,
    K.ACTIVATE_CARD: S,
    K.ORDER_REPLACEMENT_CARD: S,
    K.CHANGE_OR_RESET_PIN: S,
    K.CLOSE_CARD: S,
    K.CARD_CONTROLS: S,
    K.DISPUTE_TRANSACTION: S,
    K.CREDIT_LIMIT_REQUEST: S,
    K.FEE_OR_INTEREST_ADJUSTMENT: S,
    K.TRANSFER_FUNDS: S,
    K.PAY_CREDIT_CARD: S,
    K.OPEN_ACCOUNT: S,
    K.CLOSE_ACCOUNT: S,
    K.REPORT_FRAUD_OR_LOSS: S,
    K.REQUEST_HUMAN: H,
    K.OTHER: H,
}


def route_for(kind: IntentKind) -> Route:
    return ROUTES[kind]
