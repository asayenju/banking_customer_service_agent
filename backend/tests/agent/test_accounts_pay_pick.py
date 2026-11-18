"""Paying a card: the customer's reply picks the card however they put it, and only an account that can pay is offered."""

from banking_tools import BankingSession
from banking_tools.manifest import load_tasks

from agent.cards import describe_credit_detail
from agent.specialists.accounts import prepare
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.types import Intent, IntentKind as K

USER = "224959b99e"  # Yuki Nakamura (task_055): Platinum $75, Business Silver $125, one Green checking
BSILVER, PLAT = "cc_224959b99e_bsilver", "cc_224959b99e_plat"


def world():
    return BankingSession.from_task(next(t for t in load_tasks() if t["id"] == "task_055")).database


def run(slots, db=None):
    return prepare(Intent(id="x", kind=K.PAY_CREDIT_CARD, summary="s", slots=slots), USER, db or world())


def test_a_number_picks_the_card_in_the_order_it_was_listed():
    assert run({"credit_card_account_id": "1"}).slots["credit_card_account_id"] == BSILVER
    assert run({"credit_card_account_id": "2"}).slots["credit_card_account_id"] == PLAT
    assert run({"credit_card_account_id": "the second one"}).slots["credit_card_account_id"] == PLAT


def test_a_pasted_list_line_picks_the_card():
    db = world()
    line = describe_credit_detail(db.credit_card_accounts.data[PLAT])
    assert run({"credit_card_account_id": line}, db).slots["credit_card_account_id"] == PLAT
    assert run({"credit_card_account_id": "your Business Silver Rewards Card (credit card)"}).slots["credit_card_account_id"] == BSILVER


def test_a_number_past_the_end_asks_again_and_says_what_it_did_not_find():
    out = run({"credit_card_account_id": "9"})
    assert isinstance(out, RouteOutcome) and "Which card would you like to pay?" in out.questions[0].text


def test_the_list_tells_cards_apart_without_inventing_digits():
    out = run({})
    assert "balance $75.00" in out.questions[0].text and "opened Nov 2022" in out.questions[0].text
    assert "ending" not in out.questions[0].text


def test_the_one_checking_account_that_can_pay_is_used_without_asking():
    out = run({"credit_card_account_id": "platinum"})
    assert isinstance(out, Intent) and out.slots["checking_account_id"] == "05" and out.slots["balance_owed"] == "75.00"


def test_no_account_that_can_pay_says_the_numbers_instead_of_asking():
    db = world()
    db.accounts.data["05"]["current_holdings"] = "20.00"
    out = run({"credit_card_account_id": "platinum"}, db)
    assert out.status is OutcomeStatus.DONE and "$75.00" in out.text and "$20.00" in out.text


def _closure(db, balance_fix=None):
    from datetime import date

    from agent.workflows.base import WorkflowContext
    from agent.workflows.credit_closure import ClosureFacts, CreditClosure

    facts = ClosureFacts(credit_card_account_id=PLAT, reason=None, pay_from_checking_account_id=None, waiver_answer=None, credit_answer=None, keeps_card=False)
    return CreditClosure().decide(WorkflowContext(db, USER, date(2026, 11, 17), {}), facts)


def test_closing_a_card_with_a_balance_uses_the_one_account_that_can_pay_it():
    from agent.workflows.base import Plan

    out = _closure(world())
    assert isinstance(out, Plan) and out.steps[0].tool == "pay_credit_card_from_checking_9182"
    assert out.steps[0].args["checking_account_id"] == "05" and out.steps[0].args["amount"] == 75.0


def test_closing_a_card_when_no_account_can_pay_says_the_numbers():
    from agent.workflows.base import Reply

    db = world()
    db.accounts.data["05"]["current_holdings"] = "20.00"
    out = _closure(db)
    assert isinstance(out, Reply) and "$75.00" in out.text and "$20.00" in out.text
