import pytest

from orchestrator.routes import ROUTES, Route, route_for
from orchestrator.types import IntentKind


def test_every_intent_kind_has_exactly_one_route():
    assert set(ROUTES) == set(IntentKind)


def test_every_route_value_is_a_route():
    assert all(isinstance(route, Route) for route in ROUTES.values())


@pytest.mark.parametrize(
    ("kind", "route"),
    [
        (IntentKind.FREEZE_CARD, Route.DIRECT),
        (IntentKind.UNFREEZE_CARD, Route.DIRECT),
        (IntentKind.ACCOUNT_BALANCE_OR_HISTORY, Route.DIRECT),
        (IntentKind.LOOKUP_INFO, Route.DIRECT),
        (IntentKind.UNCLEAR_OR_GREETING, Route.DIRECT),
        (IntentKind.DISPUTE_TRANSACTION, Route.SUB_AGENT),
        (IntentKind.CREDIT_LIMIT_REQUEST, Route.SUB_AGENT),
        (IntentKind.CLOSE_CARD, Route.SUB_AGENT),
        (IntentKind.TRANSFER_FUNDS, Route.SUB_AGENT),
        (IntentKind.REQUEST_HUMAN, Route.HAND_OFF),
        (IntentKind.OTHER, Route.HAND_OFF),
    ],
)
def test_known_routes(kind, route):
    assert route_for(kind) is route
