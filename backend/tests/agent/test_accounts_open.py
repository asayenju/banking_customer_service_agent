"""Slice 4 of the accounts specialist: opening accounts. The model reports the type and exact class; code
decides what can be opened from the records, asks about the opening deposit of a savings account, and plans
the opening and the deposit. A customer's own claims about their accounts change nothing."""

import json
from datetime import date

import pytest

from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist
from agent.specialists.accounts import ACCOUNTS_SPECIALIST
from agent.tier1 import SegmentError, plan_and_execute, run_tier1
from agent.toolbox import SecretVault
from agent.workflows.account_open import FUNDING_DAYS, NewAccount, OpenAccount, OpenFacts, new_account_id
from agent.workflows.base import Ask, NeedInfo, Plan, Refuse, Reply, WorkflowContext
from banking_tools import BankingSession, Fidelity
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex
from orchestrator.types import Intent, IntentKind as K

SAM = "125"  # checking 03 (Green Account, 2023, $3,400) and savings 04 (Silver Account, $900)
MIA = "123"  # checking 01 (Blue Account, six days old) and savings 02 (Bronze Account)
TODAY = date(2025, 11, 14)
INDEX = KnowledgeIndex({"doc_accounts_x": Document("doc_accounts_x", "Internal: x", "account procedure")})
TASKS = {t["id"]: t for t in load_tasks()}


def world() -> BankingSession:
    return BankingSession(fidelity=Fidelity.TAU2)


def ctx(shared: BankingSession, user: str = SAM) -> WorkflowContext:
    return WorkflowContext(shared.database, user, TODAY)


def acct(type_="savings", cls="Silver Plus Account", fund="no", source=None, amount=None) -> NewAccount:
    return NewAccount(account_type=type_, account_class=cls, fund_now=fund, fund_from_account=source, fund_amount=amount)


def decide(shared, *accounts, user=SAM):
    return OpenAccount().decide(ctx(shared, user), OpenFacts(accounts=list(accounts)))


def asked(decision) -> dict[str, str]:
    assert isinstance(decision, NeedInfo), decision
    return {a.slot: a.text for a in decision.questions}


# ---------------------------------------------------------------- what can be opened

def test_an_eligible_account_with_no_deposit_is_a_single_opening():
    plan = decide(world(), acct("checking", "Purple Account"))
    assert isinstance(plan, Plan)
    assert [(s.tool, dict(s.args)) for s in plan.steps] == [
        ("open_bank_account_4821", {"account_type": "checking", "account_class": "Purple Account"})
    ]


def test_several_accounts_are_opened_in_the_order_given():
    plan = decide(world(), acct("checking", "Purple Account"), acct("savings", "Silver Plus Account"))
    assert [s.args["account_class"] for s in plan.steps] == ["Purple Account", "Silver Plus Account"]


def test_a_class_that_is_not_an_exact_name_gets_the_list_back_instead():
    for wrong in ("", "silver plus", "Silver Plus Account ", "Platinum Savings"):
        refusal = decide(world(), acct(cls=wrong))
        assert isinstance(refusal, Refuse)
        assert "Accounts this customer can open today (11/14/2025):" in refusal.reason and "- Silver Plus Account" in refusal.reason


def test_the_list_carries_the_running_promotion_and_leaves_out_the_expired_one():
    refusal = decide(world(), acct("business_checking", ""))
    assert "Sky Blue (November 2025 business checking promotion puts it number 1" in refusal.reason
    assert "Hunter Green" in refusal.reason and "October" not in refusal.reason


def test_an_account_the_records_say_cannot_be_opened_is_explained_from_the_records_and_never_planned():
    """Task 057: the customer says the checking account is fourteen days old; it is six."""
    shared = world()
    reply = decide(shared, acct("savings", "Silver Account"), user=MIA)
    assert isinstance(reply, Reply)
    assert reply.text == (
        "A Silver Account cannot be opened now: a savings account needs a checking account that has been open at "
        "least 14 days; your checking account has been open 6 days."
    )
    assert shared.database.accounts.data["01"]["date_opened"] == "11/08/2025", "nothing was changed"


def test_some_accounts_can_be_opened_and_the_customer_is_asked_about_the_rest():
    asks = asked(decide(world(), acct("checking", "Purple Account"), acct("savings", "Silver Account"), user=MIA))
    assert set(asks) == {"account_class"}
    assert "I can open a Purple Account now" in asks["account_class"] and "Shall I open just a Purple Account?" in asks["account_class"]
    assert "A Silver Account cannot be opened now" in asks["account_class"]


# ---------------------------------------------------------------- the opening deposit

def test_a_savings_account_asks_about_the_deposit_before_a_plan_is_made():
    asks = asked(decide(world(), acct(fund="not_asked")))
    assert set(asks) == {"fund_now"} and f"{FUNDING_DAYS} days" in asks["fund_now"]
    assert isinstance(decide(world(), acct("checking", "Purple Account", fund="not_asked")), Plan), "checking has no deposit step"


def test_a_deposit_needs_a_source_account_of_their_own_and_an_amount():
    asks = asked(decide(world(), acct(fund="yes")))
    assert set(asks) == {"fund_from_account", "fund_amount"}
    assert "1. your Green Account (checking)" in asks["fund_from_account"] and "2. your Silver Account (savings)" in asks["fund_from_account"]
    for stranger in ("01", "someone elses account"):
        again = asked(decide(world(), acct(fund="yes", source=stranger, amount=500.0)))
        assert set(again) == {"fund_from_account"} and "Blue" not in again["fund_from_account"] and "Bronze" not in again["fund_from_account"]
    assert set(asked(decide(world(), acct(fund="yes", source="Green checking", amount=0.0)))) == {"fund_amount"}
    assert set(asked(decide(world(), acct(fund="yes", source="Green checking", amount=None)))) == {"fund_amount"}


def test_a_deposit_is_planned_into_the_new_account_by_its_id():
    plan = decide(world(), acct(fund="yes", source="Green checking", amount=500.0))
    assert isinstance(plan, Plan)
    new_id = new_account_id(SAM, "savings", "Silver Plus Account")
    assert [s.tool for s in plan.steps] == ["open_bank_account_4821", "transfer_funds_between_bank_accounts_7291"]
    assert dict(plan.steps[1].args) == {"source_account_id": "03", "destination_account_id": new_id, "amount": 500.0}


def test_after_the_plan_runs_an_unfunded_savings_account_is_told_it_has_thirty_days():
    unfunded = decide(world(), acct(fund="no"))
    note = OpenAccount().after(ctx(world()), unfunded.steps)
    assert "Silver Plus Account needs its opening deposit within 30 days" in note and "will be closed" in note
    funded = decide(world(), acct(fund="yes", source="03", amount=500.0))
    assert OpenAccount().after(ctx(world()), funded.steps) == ""
    assert OpenAccount().after(ctx(world()), decide(world(), acct("checking", "Purple Account")).steps) == ""


# ---------------------------------------------------------------- the whole path: model, plan, yes, bank

def run_with(shared: BankingSession, facts: dict, user: str = SAM):
    turn = AssistantTurn(tool_calls=(ToolCall(id="c1", name="start_open_account", arguments_json=json.dumps(facts)),), usage=Usage(input_tokens=1))
    specialist = Specialist(ACCOUNTS_SPECIALIST, llm=FakeToolLLM([turn]), index=INDEX, database=shared.database, vault=SecretVault())
    it = Intent(id="i1", kind=K.OPEN_ACCOUNT, summary="open an account", slots={})
    return plan_and_execute(specialist, it, shared, user, label="open")


def form(**changes) -> dict:
    return {"accounts": [{"account_type": "savings", "account_class": "Silver Plus Account", "fund_now": "no", "fund_from_account": None, "fund_amount": None, **changes}]}


def test_the_account_is_opened_and_funded_through_the_specialist():
    shared = world()
    assert run_with(shared, form(fund_now="yes", fund_from_account="Green checking", fund_amount=500.0)) == 0
    new = shared.database.accounts.data[new_account_id(SAM, "savings", "Silver Plus Account")]
    assert new["account_class"] == "Silver Plus Account" and new["current_holdings"] == "$500.00"
    assert shared.database.accounts.data["03"]["current_holdings"] == "$2900.00"


def test_a_model_that_names_a_class_the_customer_cannot_have_cannot_open_it():
    shared = world()
    with pytest.raises(SegmentError):
        run_with(shared, form(account_class="Silver Plus Account"), user=MIA)
    assert not any(r.get("account_class") for r in shared.database.accounts.data.values())


def test_a_model_that_wants_to_skip_the_workflow_cannot_propose_a_free_plan():
    shared = world()
    turn = AssistantTurn(
        tool_calls=(ToolCall(id="c1", name="propose_plan", arguments_json=json.dumps({"steps": [{"tool": "open_bank_account_4821", "arguments_json": json.dumps({"account_type": "savings", "account_class": "Silver Plus Account"}), "description": "x"}]})),),
        usage=Usage(input_tokens=1),
    )
    specialist = Specialist(ACCOUNTS_SPECIALIST, llm=FakeToolLLM([turn]), index=INDEX, database=shared.database, vault=SecretVault())
    with pytest.raises(SegmentError):
        plan_and_execute(specialist, Intent(id="i1", kind=K.OPEN_ACCOUNT, summary="s", slots={}), shared, SAM, label="open")


# ---------------------------------------------------------------- the reference answers

@pytest.mark.parametrize("task_id", ["task_055", "task_057", "task_058", "task_059", "task_063", "task_064", "task_070", "task_071", "task_075", "task_076"])
def test_the_reference_answer_is_reproduced_through_the_open_workflow(task_id):
    result = run_tier1(TASKS[task_id], ACCOUNTS_SPECIALIST, index=INDEX)
    assert result.error is None and result.excluded is None, result
    assert result.passed, result.mismatches
    assert result.bypasses == 0
    assert any(s.startswith("OPEN_ACCOUNT") for s in result.segments)
