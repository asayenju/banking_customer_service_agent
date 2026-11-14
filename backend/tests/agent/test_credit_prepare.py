"""A credit card named by its place in the list just shown."""

from agent.cards import owned_credit_accounts
from agent.specialists.credit import prepare
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks
from orchestrator.executor import RouteOutcome
from orchestrator.types import Intent, IntentKind as K


def world():
    for task in load_tasks():
        db = BankingSession.from_task(task).database
        owners = {}
        for a in db.credit_card_accounts.data.values():
            owners.setdefault(a.get("user_id"), []).append(a)
        for user, accounts in owners.items():
            if len([a for a in accounts if str(a.get("status")).upper() != "CLOSED"]) > 1:
                return db, user
    raise AssertionError("no customer with two credit cards")


def test_the_second_one_picks_the_second_card_in_the_list_they_saw():
    db, user = world()
    shown = owned_credit_accounts(db, user)
    for text, index in (("the second one", 1), ("2", 1), ("the first", 0)):
        out = prepare(Intent(id="x", kind=K.CLOSE_CARD, summary="s", slots={"credit_card_account_id": text}), user, db)
        assert not isinstance(out, RouteOutcome) and out.slots["credit_card_account_id"] == shown[index]["account_id"]


def test_a_number_past_the_end_asks_again():
    db, user = world()
    out = prepare(Intent(id="x", kind=K.CLOSE_CARD, summary="s", slots={"credit_card_account_id": "9"}), user, db)
    assert isinstance(out, RouteOutcome)
