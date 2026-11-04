"""The workflows as the specialist offers them: a form to fill in, code that answers it."""

import asyncio
import json

from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist
from agent.specialists.debit import DEBIT_SPECIALIST
from agent.toolbox import SecretVault
from banking_tools import BankingSession, Fidelity
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.executor import OutcomeStatus
from orchestrator.types import Intent, IntentKind as K

TASK = next(t for t in load_tasks() if t["id"].endswith("084"))
USER = "a7c2f85d91"
INDEX = KnowledgeIndex({"doc_x_001": Document("doc_x_001", "X", "debit disputes")})
ONE = {
    "transaction_id": "btxn_e8c3b09d2c45",
    "category": "duplicate_charge",
    "discovery_date": "11/10/2025",
    "disputed_amount": 47.5,
    "transaction_type": "pin_purchase",
    "card_in_possession": True,
    "pin_compromised": "no",
    "contacted_merchant": False,
    "police_report_filed": False,
    "written_statement_provided": True,
}


def turn(name, **arguments):
    call = ToolCall(id=f"c{name}{len(arguments)}{id(arguments) % 997}", name=name, arguments_json=json.dumps(arguments))
    return AssistantTurn(tool_calls=(call,), usage=Usage(input_tokens=10))


def run(script, kind=K.DISPUTE_TRANSACTION):
    session = BankingSession.from_task(TASK)
    llm = FakeToolLLM(script)
    specialist = Specialist(DEBIT_SPECIALIST, llm=llm, index=INDEX, database=session.database, vault=SecretVault())
    intent = Intent(id="x", kind=kind, summary="dispute", slots={})
    ctx = RouteContext(intent_id="x", verified_user_id=USER, session=scoped(session)(SessionRequest("x")), mode="plan")
    return asyncio.run(specialist.run(intent, ctx)), llm


def test_the_model_is_offered_the_forms_and_each_form_is_a_strict_schema():
    _, llm = run([turn("start_debit_dispute", disputes=[ONE])])
    offered = set(llm.calls[0]["tools"])
    assert "start_debit_dispute" in offered and "start_debit_replacement" not in offered
    _, llm = run([turn("escalate", reason="x")], kind=K.ORDER_REPLACEMENT_CARD)
    offered = set(llm.calls[0]["tools"])
    assert "start_debit_replacement" in offered and "start_debit_dispute" not in offered


def test_filled_in_form_becomes_a_plan_with_everything_else_worked_out_by_code():
    outcome, _ = run([turn("start_debit_dispute", disputes=[ONE])])
    assert outcome.status is OutcomeStatus.NEEDS_CONFIRMATION and outcome.proposal is not None
    [step] = outcome.proposal.steps
    assert step.tool == "file_debit_card_transaction_dispute_6281"
    assert step.args["customer_max_liability_amount"] == 50.0
    assert step.args["card_action"] == "keep_active" and step.args["provisional_credit_eligible"] is True
    assert step.args["disputed_amount"] == 47.5 and step.args["transaction_date"] == "11/06/2025"


def test_a_fraud_dispute_gets_the_card_closure_added_by_code():
    fraud = {**ONE, "transaction_id": "btxn_c2g7f43h6g89", "category": "card_present_fraud",
             "discovery_date": "11/14/2025", "disputed_amount": 412.88, "pin_compromised": "unknown"}
    outcome, _ = run([turn("start_debit_dispute", disputes=[fraud])])
    tools = [s.tool for s in outcome.proposal.steps]
    assert tools == ["file_debit_card_transaction_dispute_6281", "close_debit_card_4721"]
    assert outcome.proposal.steps[0].args["customer_max_liability_amount"] == 500.0


def test_a_refusal_goes_back_to_the_model_in_words_and_the_corrected_form_is_accepted():
    too_much = {**ONE, "disputed_amount": 500.0}
    outcome, llm = run([turn("start_debit_dispute", disputes=[too_much]), turn("start_debit_dispute", disputes=[ONE])])
    assert outcome.status is OutcomeStatus.NEEDS_CONFIRMATION
    told = llm.calls[1]["messages"][-1]["content"]
    assert told.startswith("Not accepted: the disputed amount must be between $1.00 and the transaction's $47.50")


def test_a_free_form_plan_is_refused_for_a_request_a_workflow_covers():
    plan = turn("propose_plan", steps=[{"tool": "close_debit_card_4721",
                                         "arguments_json": json.dumps({"card_id": "dbc_f4a72c9e55", "reason": "lost"}),
                                         "description": "x"}])
    outcome, llm = run([plan, turn("start_debit_dispute", disputes=[ONE])])
    assert outcome.status is OutcomeStatus.NEEDS_CONFIRMATION
    assert "has a workflow; end with start_debit_dispute" in llm.calls[1]["messages"][-1]["content"]


def test_a_transaction_that_is_not_the_customers_is_refused_not_planned():
    other = {**ONE, "transaction_id": "btxn_bd30d42f67a2"}
    outcome, llm = run([turn("start_debit_dispute", disputes=[other])] * 5)
    assert outcome.status is OutcomeStatus.FAILED
    assert "not a transaction on this customer's accounts" in llm.calls[1]["messages"][-1]["content"]


def test_a_malformed_form_is_sent_back():
    bad = {**ONE, "category": "made_up"}
    outcome, llm = run([turn("start_debit_dispute", disputes=[bad]), turn("start_debit_dispute", disputes=[ONE])])
    assert outcome.status is OutcomeStatus.NEEDS_CONFIRMATION
    assert "fill in every field with an allowed value" in llm.calls[1]["messages"][-1]["content"]
