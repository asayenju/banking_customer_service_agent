"""Slice 5 of the accounts specialist: closing accounts, funds first. The money moves to an account the
customer chooses from their own, the early-closure fee stays where it is and is never moved or waived, and
what only a person can decide is handed over."""

import json
from datetime import date

import pytest

from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist
from agent.specialists.accounts import ACCOUNTS_SPECIALIST
from agent.tier1 import SegmentError, plan_and_execute, run_tier1
from agent.toolbox import SecretVault
from agent.workflows.account_close import CloseAccount, CloseFacts, Closure
from agent.workflows.base import HandOff, NeedInfo, Plan, Refuse, Reply, WorkflowContext
from banking_tools import BankingSession, Fidelity
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex
from orchestrator.types import Intent, IntentKind as K

SAM = "125"  # checking 03 (Green Account, 2023, $3,400) and savings 04 (Silver Account, $900)
MIA = "123"  # checking 01 (Blue Account, 11/08/2025, $0; $25 inside 60 days) and savings 02 (Bronze, $129.02; $20 inside 60 days)
TODAY = date(2025, 11, 14)
INDEX = KnowledgeIndex({"doc_accounts_x": Document("doc_accounts_x", "Internal: x", "account procedure")})
TASKS = {t["id"]: t for t in load_tasks()}


def world() -> BankingSession:
    return BankingSession(fidelity=Fidelity.TAU2)


def put(shared, account_id, user, cls, level, opened, balance):
    shared.database.accounts.data[account_id] = {
        "account_id": account_id, "user_id": user, "class": cls, "level": level,
        "date_opened": opened, "status": "OPEN", "current_holdings": balance,
    }


def closing(account_id="03", to=None, reason="Customer requested closure", waive=False) -> Closure:
    return Closure(account_id=account_id, move_to_account=to, reason=reason, asks_to_waive_fee=waive)


def decide(shared, *closures, user=SAM):
    return CloseAccount().decide(WorkflowContext(shared.database, user, TODAY), CloseFacts(closures=list(closures)))


def tools(plan: Plan) -> list[tuple[str, dict]]:
    return [(s.tool, dict(s.args)) for s in plan.steps]


# ---------------------------------------------------------------- funds first

def test_an_empty_account_outside_the_fee_window_is_just_closed():
    shared = world()
    shared.database.accounts.data["03"]["current_holdings"] = "0"
    assert tools(decide(shared, closing())) == [("close_bank_account_7392", {"account_id": "03", "reason": "Customer requested closure"})]


def test_the_balance_moves_to_the_account_the_customer_chose_before_the_account_closes():
    plan = decide(world(), closing(to="Silver savings"))
    assert tools(plan) == [
        ("transfer_funds_between_bank_accounts_7291", {"source_account_id": "03", "destination_account_id": "04", "amount": 3400.0}),
        ("close_bank_account_7392", {"account_id": "03", "reason": "Customer requested closure"}),
    ]


def test_a_balance_with_no_destination_is_a_question_listing_only_their_other_accounts():
    asks = decide(world(), closing())
    assert isinstance(asks, NeedInfo) and [a.slot for a in asks.questions] == ["move_to_account"]
    text = asks.questions[0].text
    assert "has $3,400.00. Where should $3,400.00 go before it closes?" in text
    assert "1. your Silver Account (savings)" in text and "Green Account" not in text.split("\n", 1)[1]


def test_money_is_never_sent_to_someone_elses_account_or_to_a_made_up_one():
    for stranger in ("01", "someone elses account", "9999"):
        asks = decide(world(), closing(to=stranger))
        assert isinstance(asks, NeedInfo), stranger
        text = asks.questions[0].text
        assert "Blue" not in text and "Bronze" not in text, "Mia's accounts are never offered to Sam"


def test_money_is_not_sent_to_an_account_that_is_being_closed_too():
    shared = world()
    asks = decide(shared, closing("03", to="Silver savings"), closing("04", to="Green checking"))
    # each is the other's destination, and both are closing: neither is an option
    assert isinstance(asks, NeedInfo) and len(asks.questions) == 2
    assert all("no other open account" in a.text for a in asks.questions)


def test_a_customer_with_no_other_open_account_is_told_there_is_nowhere_to_send_it():
    shared = world()
    put(shared, "solo", "alone", "checking", "Green Account", "01/01/2024", "75.00")
    asks = decide(shared, closing("solo"), user="alone")
    assert "no other open account to move $75.00 to" in asks.questions[0].text


# ---------------------------------------------------------------- the early-closure fee

def test_inside_the_window_a_balance_equal_to_the_fee_closes_without_moving_anything():
    """Task 061: the customer deposited the $25 themselves; the fee stays in the account."""
    shared = world()
    shared.database.accounts.data["01"]["current_holdings"] = "25"
    assert [t for t, _ in tools(decide(shared, closing("01"), user=MIA))] == ["close_bank_account_7392"]


def test_inside_the_window_only_what_is_above_the_fee_moves():
    shared = world()
    shared.database.accounts.data["01"]["current_holdings"] = "100"
    plan = decide(shared, closing("01", to="Bronze savings"), user=MIA)
    assert tools(plan)[0] == ("transfer_funds_between_bank_accounts_7291", {"source_account_id": "01", "destination_account_id": "02", "amount": 75.0})
    assert [t for t, _ in tools(plan)] == ["transfer_funds_between_bank_accounts_7291", "close_bank_account_7392"]


def test_inside_the_window_with_less_than_the_fee_the_customer_is_told_how_much_to_add():
    reply = decide(world(), closing("01"), user=MIA)
    assert isinstance(reply, Reply)
    assert reply.text == (
        "Your Blue Account was opened less than 60 days ago, so closing it now costs an early-closure fee of $25.00, "
        "taken from its balance. It has $0.00, so it needs $25.00 more before it can close. You can add that by moving "
        "money in from another of your accounts, or by depositing a check yourself; then ask me to close it again."
    )


def test_the_savings_fee_the_bank_never_applies_is_applied_here():
    shared = world()
    shared.database.accounts.data["02"]["current_holdings"] = "0"
    reply = decide(shared, closing("02"), user=MIA)
    assert isinstance(reply, Reply) and "$20.00" in reply.text and "Bronze Account" in reply.text


def test_the_fee_is_never_waived_by_the_agent_and_a_request_to_waive_goes_to_a_person():
    handed = decide(world(), closing("01", waive=True), user=MIA)
    assert isinstance(handed, HandOff) and "waived" in handed.reason
    plan = decide(world(), closing("03", to="Silver savings"))
    assert all("waive" not in step.args for step in plan.steps)


def test_after_closing_the_customer_is_told_the_fee_came_out_of_the_balance_left():
    shared = world()
    shared.database.accounts.data["01"]["current_holdings"] = "25"
    plan = decide(shared, closing("01"), user=MIA)
    shared.call("close_bank_account_7392", {"account_id": "01", "reason": "x"})
    note = CloseAccount().after(WorkflowContext(shared.database, MIA, TODAY), plan.steps)
    assert note == "The $25.00 early-closure fee on your Blue Account was taken from the balance left in it."
    unfeed = world()
    unfeed.database.accounts.data["03"]["current_holdings"] = "0"
    assert CloseAccount().after(WorkflowContext(unfeed.database, SAM, TODAY), decide(unfeed, closing()).steps) == ""


# ---------------------------------------------------------------- what only a person can decide

def test_approval_tiers_missing_rules_and_unknown_balances_are_handed_over():
    shared = world()
    put(shared, "pl", SAM, "saving", "Platinum Account", "01/01/2020", "0")
    put(shared, "pp", SAM, "checking", "Purple Account", "01/01/2020", "0")
    put(shared, "nb", SAM, "checking", "Blue Account", "01/01/2020", None)
    assert "manager approval" in decide(shared, closing("pl")).reason
    assert "no closing rules on file for a Purple Account" in decide(shared, closing("pp")).reason
    assert "balance of the Blue Account is not on file" in decide(shared, closing("nb")).reason


def test_a_closed_unknown_or_someone_elses_account():
    shared = world()
    shared.database.accounts.data["04"]["status"] = "CLOSED"
    assert decide(shared, closing("04")) == Reply("Your Silver Account is already closed.")
    for bad in ("01", "nope"):
        assert isinstance(decide(shared, closing(bad)), Refuse)


# ---------------------------------------------------------------- the whole path

def run_with(shared: BankingSession, facts: dict, user: str = SAM, account: str = "03"):
    turn = AssistantTurn(tool_calls=(ToolCall(id="c1", name="start_close_account", arguments_json=json.dumps(facts)),), usage=Usage(input_tokens=1))
    specialist = Specialist(ACCOUNTS_SPECIALIST, llm=FakeToolLLM([turn]), index=INDEX, database=shared.database, vault=SecretVault())
    it = Intent(id="i1", kind=K.CLOSE_ACCOUNT, summary="close an account", slots={"account_id": account})
    return plan_and_execute(specialist, it, shared, user, label="close")


def form(account="03", to=None, waive=False) -> dict:
    return {"closures": [{"account_id": account, "move_to_account": to, "reason": "Customer requested closure", "asks_to_waive_fee": waive}]}


def test_the_money_moves_and_the_account_closes_through_the_specialist():
    shared = world()
    assert run_with(shared, form(to="Silver savings")) == 0
    assert shared.database.accounts.data["03"]["status"] == "CLOSED"
    assert shared.database.accounts.data["04"]["current_holdings"] == "$4300.00"


def test_inside_the_window_the_fee_is_left_in_the_closed_account_and_the_rest_moves():
    shared = world()
    shared.database.accounts.data["01"]["current_holdings"] = "100"
    assert run_with(shared, form("01", to="Bronze savings"), user=MIA, account="01") == 0
    assert shared.database.accounts.data["01"]["status"] == "CLOSED" and shared.database.accounts.data["01"]["current_holdings"] == "$25.00"
    assert shared.database.accounts.data["02"]["current_holdings"] == "$204.02"


def test_a_destination_that_is_someone_elses_cannot_move_anything():
    shared = world()
    with pytest.raises(SegmentError):
        run_with(shared, form(to="01"))
    assert shared.database.accounts.data["03"]["status"] == "OPEN" and shared.database.accounts.data["01"]["current_holdings"] == "0"


def test_a_model_cannot_waive_the_fee_or_close_with_money_in_the_account_by_going_around_the_workflow():
    shared = world()
    with pytest.raises(SegmentError):
        run_with(shared, form("01", waive=True), user=MIA, account="01")
    turn = AssistantTurn(
        tool_calls=(ToolCall(id="c1", name="propose_plan", arguments_json=json.dumps({"steps": [{"tool": "close_bank_account_7392", "arguments_json": json.dumps({"account_id": "03", "waive_early_closure_fee": True}), "description": "x"}]})),),
        usage=Usage(input_tokens=1),
    )
    specialist = Specialist(ACCOUNTS_SPECIALIST, llm=FakeToolLLM([turn]), index=INDEX, database=shared.database, vault=SecretVault())
    with pytest.raises(SegmentError):
        plan_and_execute(specialist, Intent(id="i1", kind=K.CLOSE_ACCOUNT, summary="s", slots={}), shared, SAM, label="close")
    assert shared.database.accounts.data["03"]["status"] == "OPEN"


def test_two_accounts_are_closed_in_the_order_given():
    shared = world()
    shared.database.accounts.data["03"]["current_holdings"] = "0"
    shared.database.accounts.data["04"]["current_holdings"] = "0"
    facts = {"closures": form("03")["closures"] + form("04")["closures"]}
    assert run_with(shared, facts) == 0
    assert shared.database.accounts.data["03"]["status"] == shared.database.accounts.data["04"]["status"] == "CLOSED"


# ---------------------------------------------------------------- the reference answers

@pytest.mark.parametrize("task_id", ["task_060", "task_061"])
def test_the_reference_answer_is_reproduced_through_the_close_workflow(task_id):
    result = run_tier1(TASKS[task_id], ACCOUNTS_SPECIALIST, index=INDEX)
    assert result.error is None and result.excluded is None, result
    assert result.passed, result.mismatches
    assert result.bypasses == 0
    assert [s.split(":")[0] for s in result.segments] == ["OPEN_ACCOUNT", "CLOSE_ACCOUNT"]
