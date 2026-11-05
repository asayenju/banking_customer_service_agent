"""Slice 8 of the accounts specialist: requests the bank has no tool for (wires, ACH, bill pay, sending money
to someone else, depositing a check). A question is answered from the documents; a request to do one gets the
customer's own steps or a hand-off written by code, and no money moves. The eight "transfer-only" tasks are
about transfers to a human agent, and make no account change."""

import asyncio
import json
from datetime import date

import pytest

from agent.dispatcher import FIXED, Name
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist
from agent.specialists.accounts import ACCOUNTS_SPECIALIST
from agent.specialists.general import GENERAL_ANSWERER
from agent.tier1 import gold_steps
from agent.toolbox import SecretVault
from agent.workflows.account_move import CHECK_DEPOSIT_STEPS, TransferFacts, TransferFunds
from agent.workflows.base import HandOff, Reply, WorkflowContext
from banking_tools import BankingSession, Fidelity
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex, get_index
from knowledge.scopes import ACCOUNTS, GENERAL
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.executor import OutcomeStatus
from orchestrator.types import Intent, IntentKind as K

SAM = "125"
TODAY = date(2025, 11, 14)
INDEX = KnowledgeIndex({"doc_accounts_x": Document("doc_accounts_x", "Internal: x", "account procedure")})
TASKS = {t["id"]: t for t in load_tasks()}


def decide(destination, shared=None):
    shared = shared or BankingSession(fidelity=Fidelity.TAU2)
    facts = TransferFacts(destination=destination, from_account_id="03", to_account_id="a stranger", amount=250.0)
    return TransferFunds().decide(WorkflowContext(shared.database, SAM, TODAY), facts)


# ---------------------------------------------------------------- sending money outside the customer's own accounts

@pytest.mark.parametrize("means", ["wire", "ach", "bill_pay", "another_person_or_bank"])
def test_sending_money_to_someone_else_is_handed_to_a_person_with_a_reason_written_here(means):
    decision = decide(means)
    assert isinstance(decision, HandOff)
    assert decision.reason == f"the customer wants to send money by {means.replace('_', ' ')} to someone outside their own accounts, which the bank has no tool for"
    assert "250" not in decision.reason and "stranger" not in decision.reason, "nothing the customer said travels in the note"


@pytest.mark.parametrize("means", ["wire", "ach", "bill_pay", "another_person_or_bank"])
def test_nothing_moves_when_the_model_files_an_outside_transfer(means):
    shared = BankingSession(fidelity=Fidelity.TAU2)
    facts = {"destination": means, "from_account_id": "03", "to_account_id": "04", "amount": 250.0}
    turn = AssistantTurn(tool_calls=(ToolCall(id="c1", name="start_transfer_funds", arguments_json=json.dumps(facts)),), usage=Usage(input_tokens=1))
    specialist = Specialist(ACCOUNTS_SPECIALIST, llm=FakeToolLLM([turn]), index=INDEX, database=shared.database, vault=SecretVault())
    intent = Intent(id="i1", kind=K.TRANSFER_FUNDS, summary="s", slots={"from_account_id": "03", "to_account_id": "04", "amount": "250"})
    ctx = RouteContext(intent_id="i1", verified_user_id=SAM, session=scoped(shared)(SessionRequest("i1", mode="plan")), mode="plan")
    out = asyncio.run(specialist.run(intent, ctx))
    assert out.status is OutcomeStatus.FAILED and "human" in (out.error or "")
    assert shared.database.accounts.data["03"]["current_holdings"] == "3400" and not shared.audit_log


# ---------------------------------------------------------------- depositing a check is the customer's own step

def test_a_check_deposit_gets_the_customers_own_steps_and_never_a_tool_name():
    reply = decide("check_deposit")
    assert reply == Reply(CHECK_DEPOSIT_STEPS)
    assert "Mobile Check Deposit" in reply.text and "1 to 2 business days" in reply.text and "cannot" not in reply.text
    assert "deposit_check" not in reply.text and "3847" not in reply.text


def test_the_check_deposit_reply_is_the_outcome_and_changes_nothing():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    facts = {"destination": "check_deposit", "from_account_id": "03", "to_account_id": "04", "amount": 100.0}
    turn = AssistantTurn(tool_calls=(ToolCall(id="c1", name="start_transfer_funds", arguments_json=json.dumps(facts)),), usage=Usage(input_tokens=1))
    specialist = Specialist(ACCOUNTS_SPECIALIST, llm=FakeToolLLM([turn]), index=INDEX, database=shared.database, vault=SecretVault())
    intent = Intent(id="i1", kind=K.TRANSFER_FUNDS, summary="s", slots={})
    ctx = RouteContext(intent_id="i1", verified_user_id=SAM, session=scoped(shared)(SessionRequest("i1", mode="plan")), mode="plan")
    out = asyncio.run(specialist.run(intent, ctx))
    assert out.status is OutcomeStatus.DONE and out.text == CHECK_DEPOSIT_STEPS
    assert not shared.audit_log


# ---------------------------------------------------------------- questions are answered from the documents

QUESTIONS = {
    "wire transfer fee": ("wire",),
    "same day ACH": ("ach",),
    "how long until a deposited check is available": ("check", "deposit"),
    "overdraft protection": ("overdraft",),
    "hold on my paycheck direct deposit": ("hold", "direct deposit"),
    "pay a bill online": ("bill",),
}


@pytest.mark.parametrize("question", sorted(QUESTIONS))
def test_a_question_about_a_service_without_a_tool_finds_the_documents_for_it(question):
    index = get_index()
    for scope in (GENERAL, ACCOUNTS):
        hits = index.search(question, scope, k=5)
        assert hits, f"{question!r} found nothing in the {scope.name} scope"
        text = " ".join(h.text.lower() for h in hits)
        assert any(word in text for word in QUESTIONS[question]), question


def test_a_general_question_goes_to_the_answerer_that_must_cite_documents():
    assert FIXED[K.GENERAL_QUESTION] is Name.GENERAL
    assert GENERAL_ANSWERER.cite_documents is True
    assert not GENERAL_ANSWERER.workflows and GENERAL_ANSWERER.terminals == ("finish", "ask_customer", "escalate")


# ---------------------------------------------------------------- the eight tasks about transfers to a person

HUMAN_TRANSFER_TASKS = ["task_004", "task_008", "task_012", "task_014", "task_032", "task_033", "task_034", "task_035"]


@pytest.mark.parametrize("task_id", HUMAN_TRANSFER_TASKS)
def test_the_reference_for_a_transfer_to_a_person_makes_no_account_change(task_id):
    """These tasks never verify a customer and end in a hand-off, so tier-1 cannot run them (it needs a verified
    customer to replay against) and the state comparison does not apply. What is checked is the reference."""
    steps = gold_steps(TASKS[task_id])
    assert not [s for s in steps if s.kind == "write"], "a hand-off task has no domain write"
    assert any(s.name == "transfer_to_human_agents" or s.name == "request_human_agent_transfer" for s in steps)
    assert {s.name for s in steps if s.kind == "other"} <= {
        "transfer_to_human_agents", "initial_transfer_to_human_agent_0218", "initial_transfer_to_human_agent_1822",
        "emergency_credit_bureau_incident_transfer_1114", "give_discoverable_user_tool",
    }
