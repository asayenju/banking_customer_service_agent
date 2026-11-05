"""Slice 7 of the accounts specialist: a transfer between the customer's own accounts and a credit card
payment from checking. No reference answer exercises either on its own, so these run on synthetic data."""

import asyncio
import json
from datetime import date

import pytest

from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist
from agent.specialists.accounts import ACCOUNTS_SPECIALIST, prepare
from agent.specialists.credit import CREDIT_SPECIALIST
from agent.tier1 import SegmentError, plan_and_execute, run_tier1
from agent.toolbox import SecretVault
from agent.workflows.account_move import PayCard, PayFacts, TransferFacts, TransferFunds
from agent.workflows.base import NeedInfo, Plan, Refuse, Reply, WorkflowContext
from banking_tools import BankingSession, Fidelity
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.decision import fingerprint
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.types import Intent, IntentKind as K

SAM = "125"  # checking 03 (Green Account, $3,400) and savings 04 (Silver Account, $900)
MIA = "123"  # checking 01 (Blue, six days old) and savings 02
CARD_OWNER = "lj82d4f1a9"  # checking chk_lj82d4f1a9 ($127,450) and chk_538bfb9cba ($1,847.50); a Silver card owing $842.15
CARD = "cc_lj82d4f1a9_silver"
TODAY = date(2025, 11, 14)
INDEX = KnowledgeIndex({"doc_accounts_x": Document("doc_accounts_x", "Internal: x", "account procedure")})
TASKS = {t["id"]: t for t in load_tasks()}


def world() -> BankingSession:
    return BankingSession(fidelity=Fidelity.TAU2)


def ctx(shared, user=SAM, **slots) -> WorkflowContext:
    return WorkflowContext(shared.database, user, TODAY, slots)


def move(shared, src="03", dst="04", amount=100.0, user=SAM, slots=None):
    return TransferFunds().decide(ctx(shared, user, **(slots or {})), TransferFacts(from_account_id=src, to_account_id=dst, amount=amount))


def pay(shared, card=CARD, checking="chk_lj82d4f1a9", amount=100.0, slots=None):
    return PayCard().decide(ctx(shared, CARD_OWNER, **(slots or {})), PayFacts(credit_card_account_id=card, checking_account_id=checking, amount=amount))


# ---------------------------------------------------------------- transfers

def test_a_transfer_between_the_customers_own_accounts_is_one_step_with_the_accounts_found_in_their_records():
    plan = move(world(), "Green checking", "Silver savings", 100.0)
    assert isinstance(plan, Plan)
    assert [(s.tool, dict(s.args)) for s in plan.steps] == [
        ("transfer_funds_between_bank_accounts_7291", {"source_account_id": "03", "destination_account_id": "04", "amount": 100.0})
    ]


def test_money_cannot_be_sent_to_someone_elses_account_or_a_made_up_one():
    for stranger in ("01", "someone elses account", "9999"):
        asks = move(world(), "03", stranger)
        assert isinstance(asks, NeedInfo) and [a.slot for a in asks.questions] == ["to_account_id"], stranger
        assert "Blue" not in asks.questions[0].text and "Bronze" not in asks.questions[0].text
        assert "1. your Silver Account (savings)" in asks.questions[0].text
    from_stranger = move(world(), "01", "04")
    assert isinstance(from_stranger, NeedInfo) and from_stranger.questions[0].slot == "from_account_id"


def test_a_transfer_to_the_same_account_asks_again_with_the_others():
    asks = move(world(), "03", "03")
    assert isinstance(asks, NeedInfo) and "Green Account" not in asks.questions[0].text


def test_a_closed_account_is_not_offered_and_does_not_resolve():
    shared = world()
    shared.database.accounts.data["04"]["status"] = "CLOSED"
    assert isinstance(move(shared, "03", "04"), NeedInfo)


def test_a_transfer_needs_a_positive_amount_that_the_account_can_cover():
    for bad in (0.0, -5.0):
        assert isinstance(move(world(), amount=bad), Refuse)
    short = move(world(), amount=5000.0)
    assert short == Reply("I can't move that money: the account the money comes from has $3,400.00, which is not enough for $5,000.00.")
    assert isinstance(move(world(), amount=3400.0), Plan), "exactly the balance is enough"


def test_the_amount_in_the_form_must_be_the_amount_the_customer_gave():
    assert isinstance(move(world(), amount=900.0, slots={"note": "x"}), Plan), "an unrelated slot changes nothing"
    refused = move(world(), amount=900.0, slots={"amount": "$100"})
    assert refused == Refuse("the customer gave the amount $100.00; fill in that exact amount")
    assert isinstance(move(world(), amount=100.0, slots={"amount": "$100.00"}), Plan)
    assert isinstance(move(world(), amount=100.0, slots={"amount": "1,000 or so"}), Plan), "an amount that cannot be read is not compared"


def test_the_yes_is_bound_to_the_amount_so_a_changed_amount_makes_it_stale():
    shared = world()
    asked = Intent(id="i1", kind=K.TRANSFER_FUNDS, summary="s", slots={"from_account_id": "03", "to_account_id": "04", "amount": "100"})
    changed = asked.model_copy(update={"slots": {**asked.slots, "amount": "900"}})
    proposal = _plan_with(shared, asked, {"destination": "my_own_account", "from_account_id": "03", "to_account_id": "04", "amount": 100.0})
    assert proposal.intent_fp == fingerprint(asked) != fingerprint(changed)
    assert "Move $100.00 from your Green Account (checking) to your Silver Account (savings)" in proposal.rendered
    assert proposal.steps[0].args["amount"] == 100.0


def _plan_with(shared, intent, facts, user=SAM, tool="start_transfer_funds"):
    turn = AssistantTurn(tool_calls=(ToolCall(id="c1", name=tool, arguments_json=json.dumps(facts)),), usage=Usage(input_tokens=1))
    specialist = Specialist(ACCOUNTS_SPECIALIST, llm=FakeToolLLM([turn]), index=INDEX, database=shared.database, vault=SecretVault())
    session = scoped(shared)(SessionRequest(intent.id, mode="plan", approved=None))
    out: RouteOutcome = asyncio.run(specialist.run(intent, RouteContext(intent_id=intent.id, verified_user_id=user, session=session, mode="plan")))
    assert out.status is OutcomeStatus.NEEDS_CONFIRMATION, out
    return out.proposal


# ---------------------------------------------------------------- the whole path, and a repeat

def run(shared, facts, *, slots=None, user=SAM, progress=(), tool="start_transfer_funds", kind=K.TRANSFER_FUNDS):
    turn = AssistantTurn(tool_calls=(ToolCall(id="c1", name=tool, arguments_json=json.dumps(facts)),), usage=Usage(input_tokens=1))
    specialist = Specialist(ACCOUNTS_SPECIALIST, llm=FakeToolLLM([turn]), index=INDEX, database=shared.database, vault=SecretVault())
    intent = Intent(id="i1", kind=kind, summary="s", slots=slots or {})
    plan_ctx = RouteContext(intent_id="i1", verified_user_id=user, session=scoped(shared)(SessionRequest("i1", mode="plan")), mode="plan")
    planned = asyncio.run(specialist.run(intent, plan_ctx))
    assert planned.status is OutcomeStatus.NEEDS_CONFIRMATION, planned
    execute = RouteContext(
        intent_id="i1", verified_user_id=user, mode="execute", approved=planned.proposal, progress=tuple(progress),
        session=scoped(shared)(SessionRequest("i1", progress=tuple(progress), mode="execute", approved=planned.proposal)),
    )
    return asyncio.run(specialist.run(intent, execute))


def test_the_transfer_moves_the_money_through_the_specialist():
    shared = world()
    done = run(shared, {"destination": "my_own_account", "from_account_id": "03", "to_account_id": "04", "amount": 100.0}, slots={"from_account_id": "03", "to_account_id": "04", "amount": "100"})
    assert done.status is OutcomeStatus.DONE
    assert shared.database.accounts.data["03"]["current_holdings"] == "$3300.00" and shared.database.accounts.data["04"]["current_holdings"] == "$1000.00"


def test_a_repeated_run_after_a_timeout_does_not_move_the_money_twice():
    shared = world()
    facts = {"destination": "my_own_account", "from_account_id": "03", "to_account_id": "04", "amount": 100.0}
    slots = {"from_account_id": "03", "to_account_id": "04", "amount": "100"}
    assert run(shared, facts, slots=slots).status is OutcomeStatus.DONE
    earlier = tuple(shared.completed_for("i1"))
    assert len(earlier) == 1
    again = run(shared, facts, slots=slots, progress=earlier)
    assert again.status is OutcomeStatus.DONE
    assert shared.database.accounts.data["04"]["current_holdings"] == "$1000.00", "the second run found the transfer already made"
    assert len([e for e in shared.audit_log if e.tool == "transfer_funds_between_bank_accounts_7291" and e.ok]) == 1


def test_a_model_that_changes_the_amount_cannot_get_it_approved():
    shared = world()
    with pytest.raises(SegmentError):
        run_wrong = Specialist(
            ACCOUNTS_SPECIALIST,
            llm=FakeToolLLM([AssistantTurn(tool_calls=(ToolCall(id="c1", name="start_transfer_funds", arguments_json=json.dumps({"destination": "my_own_account", "from_account_id": "03", "to_account_id": "04", "amount": 900.0, "destination": "my_own_account"})),), usage=Usage(input_tokens=1))]),
            index=INDEX, database=shared.database, vault=SecretVault(),
        )
        plan_and_execute(run_wrong, Intent(id="i1", kind=K.TRANSFER_FUNDS, summary="s", slots={"amount": "100"}), shared, SAM, label="move")
    assert shared.database.accounts.data["03"]["current_holdings"] == "3400"


# ---------------------------------------------------------------- paying a card

def test_a_card_payment_is_one_step_from_an_open_checking_account():
    plan = pay(world(), amount=100.0)
    assert isinstance(plan, Plan)
    assert [(s.tool, dict(s.args)) for s in plan.steps] == [
        ("pay_credit_card_from_checking_9182", {"user_id": CARD_OWNER, "checking_account_id": "chk_lj82d4f1a9", "credit_card_account_id": CARD, "amount": 100.0})
    ]


def test_a_payment_above_the_balance_or_the_funds_is_told_in_words_and_not_planned():
    assert pay(world(), amount=900.0) == Reply("I can't make that payment: the card's balance is $842.15, which is less than $900.00.")
    shared = world()
    shared.database.accounts.data["chk_lj82d4f1a9"]["current_holdings"] = "50.00"
    assert pay(shared, amount=100.0) == Reply("I can't make that payment: the checking account has $50.00, which is not enough.")
    assert isinstance(pay(world(), amount=842.15), Plan), "the whole balance can be paid"


def test_the_card_and_the_checking_account_must_be_the_customers_own():
    other = pay(world(), card="cc_76ad9cc60e_gold")
    assert isinstance(other, NeedInfo) and other.questions[0].slot == "credit_card_account_id"
    assert "Gold" not in other.questions[0].text and "Silver Rewards Card" in other.questions[0].text
    foreign = pay(world(), checking="03")
    assert isinstance(foreign, NeedInfo) and foreign.questions[0].slot == "checking_account_id"
    assert isinstance(pay(world(), amount=0.0), Refuse)


def test_a_savings_account_is_not_offered_as_the_place_to_pay_from():
    shared = world()
    shared.database.accounts.data["sv"] = {"account_id": "sv", "user_id": CARD_OWNER, "class": "saving", "level": "Silver Account", "date_opened": "01/01/2024", "status": "OPEN", "current_holdings": "9000"}
    asks = pay(shared, checking="sv")
    assert isinstance(asks, NeedInfo) and "Silver Account" not in asks.questions[0].text


def test_a_checking_account_opened_in_this_visit_cannot_pay_a_card_and_the_customer_is_told_why():
    shared = world()
    shared.call("open_bank_account_4821", {"user_id": CARD_OWNER, "account_type": "checking", "account_class": "Purple Account"})
    new = next(r["account_id"] for r in shared.database.accounts.data.values() if r.get("account_class") == "Purple Account")
    shared.database.accounts.data[new]["current_holdings"] = "500"
    reply = pay(shared, checking="Purple Account", amount=10.0)
    assert isinstance(reply, Reply) and "too new for the bank's payment system" in reply.text


def test_a_payment_runs_through_the_specialist():
    shared = world()
    facts = {"credit_card_account_id": CARD, "checking_account_id": "chk_lj82d4f1a9", "amount": 100.0}
    done = run(shared, facts, user=CARD_OWNER, tool="start_pay_card", kind=K.PAY_CREDIT_CARD, slots={"amount": "100", "checking_account_id": "chk_lj82d4f1a9"})
    assert done.status is OutcomeStatus.DONE
    assert shared.database.credit_card_accounts.data[CARD]["current_balance"] == "$742.15"


# ---------------------------------------------------------------- finding the card and the account

def intent(**slots) -> Intent:
    return Intent(id="i1", kind=K.PAY_CREDIT_CARD, summary="s", slots=slots)


def test_a_single_card_and_a_single_checking_account_need_no_question():
    shared = world()
    shared.database.accounts.data["chk_538bfb9cba"]["status"] = "CLOSED"
    out = prepare(intent(amount="100"), CARD_OWNER, shared.database)
    assert isinstance(out, Intent) and out.slots["credit_card_account_id"] == CARD and out.slots["checking_account_id"] == "chk_lj82d4f1a9"


def test_two_checking_accounts_are_a_question_with_the_customers_own_listed():
    out = prepare(intent(amount="100"), CARD_OWNER, world().database)
    assert isinstance(out, RouteOutcome) and out.questions[0].slot == "checking_account_id"
    assert out.questions[0].text.count("\n") == 2 and "Which checking account should the payment come from?" in out.questions[0].text


def test_an_account_id_slot_that_names_the_card_or_the_checking_account_is_understood():
    by_card = prepare(intent(account_id=CARD, checking_account_id="chk_538bfb9cba"), CARD_OWNER, world().database)
    assert isinstance(by_card, Intent) and by_card.slots["credit_card_account_id"] == CARD and by_card.slots["checking_account_id"] == "chk_538bfb9cba"
    by_checking = prepare(intent(account_id="chk_538bfb9cba"), CARD_OWNER, world().database)
    assert isinstance(by_checking, Intent) and by_checking.slots["checking_account_id"] == "chk_538bfb9cba"


def test_a_customer_with_no_card_or_no_checking_account_is_told_so():
    nothing = prepare(intent(), SAM, world().database)
    assert isinstance(nothing, RouteOutcome) and nothing.status is OutcomeStatus.DONE and "credit card" in nothing.text
    shared = world()
    shared.database.accounts.data["03"]["status"] = "CLOSED"
    shared.database.credit_card_accounts.data["c"] = {"account_id": "c", "user_id": SAM, "card_type": "Silver Rewards Card", "status": "ACTIVE", "current_balance": "$5.00"}
    no_checking = prepare(intent(), SAM, shared.database)
    assert isinstance(no_checking, RouteOutcome) and "checking account" in no_checking.text


# ---------------------------------------------------------------- the boundary with the credit specialist

@pytest.mark.parametrize("task_id", ["task_043", "task_045", "task_048", "task_051"])
def test_the_credit_flows_that_pay_a_card_are_unchanged(task_id):
    result = run_tier1(TASKS[task_id], CREDIT_SPECIALIST, index=INDEX)
    assert result.error is None and result.excluded is None, result
    assert result.passed or task_id == "task_048", result.mismatches


# ---------------------------------------------------------------- the plan names accounts, not ids

def test_the_plan_for_a_closing_names_the_accounts_in_the_customers_words():
    shared = world()
    asked = Intent(id="i1", kind=K.CLOSE_ACCOUNT, summary="s", slots={"account_id": "03"})
    facts = {"closures": [{"account_id": "03", "moves": [{"to_account": "04", "amount": None}], "reason": "Customer requested closure", "asks_to_waive_fee": False}]}
    proposal = _plan_with(shared, asked, facts, tool="start_close_account")
    assert "1. Move $3,400.00 from your Green Account (checking) to your Silver Account (savings)" in proposal.rendered
    assert "2. Close your Green Account (checking) (" in proposal.rendered and "cannot be undone" in proposal.rendered


def test_the_plan_for_an_opening_names_the_class_and_the_deposit_in_words():
    shared = world()
    asked = Intent(id="i1", kind=K.OPEN_ACCOUNT, summary="s", slots={})
    facts = {"accounts": [{"account_type": "savings", "account_class": "Silver Plus Account", "fund_now": "yes", "fund_from_account": "03", "fund_amount": 500.0}]}
    proposal = _plan_with(shared, asked, facts, tool="start_open_account")
    assert "1. Open a new Silver Plus Account (savings)" in proposal.rendered
    assert "2. Move $500.00 from your Green Account (checking) into your new Silver Plus Account as its opening deposit" in proposal.rendered


def test_the_words_shown_are_not_part_of_what_the_yes_is_bound_to():
    shared = world()
    asked = Intent(id="i1", kind=K.TRANSFER_FUNDS, summary="s", slots={"from_account_id": "03", "to_account_id": "04", "amount": "100"})
    one = _plan_with(shared, asked, {"destination": "my_own_account", "from_account_id": "03", "to_account_id": "04", "amount": 100.0})
    other = _plan_with(world(), asked, {"destination": "my_own_account", "from_account_id": "03", "to_account_id": "04", "amount": 100.0})
    assert one.plan_fp == other.plan_fp
    changed = _plan_with(world(), asked.model_copy(update={"slots": {**asked.slots, "amount": "101"}}), {"destination": "my_own_account", "from_account_id": "03", "to_account_id": "04", "amount": 101.0})
    assert changed.plan_fp != one.plan_fp
