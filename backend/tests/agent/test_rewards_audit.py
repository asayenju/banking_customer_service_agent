"""A rewards complaint that names no purchase: the model is never allowed to ask which purchase. It ends with the workflow, which
finds the wrong purchases itself and plans the cash back disputes for the customer to approve with one yes."""

import asyncio
import json

from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist
from agent.specialists.fees import FEES_SPECIALIST
from agent.toolbox import SecretVault
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.executor import OutcomeStatus
from orchestrator.types import Intent, IntentKind as K

TASK = next(t for t in load_tasks() if t["id"] == "task_028")
CUSTOMER = "6680a37184"  # Fatima: one Silver Rewards Card, five purchases, three of them wrong
INDEX = KnowledgeIndex({"doc_x": Document("doc_x", "Internal: rewards", "rewards")})


def call(name, **arguments):
    return ToolCall(id=f"c-{name}", name=name, arguments_json=json.dumps(arguments))


def run(script, summary="some of my cash back rewards might be wrong"):
    shared = BankingSession.from_task(TASK)
    llm = FakeToolLLM([AssistantTurn(tool_calls=(c,), usage=Usage(input_tokens=1)) for c in script])
    spec = Specialist(FEES_SPECIALIST, llm=llm, index=INDEX, database=shared.database, vault=SecretVault())
    intent = Intent(id="x", kind=K.FEE_OR_INTEREST_ADJUSTMENT, summary=summary, slots={})
    ctx = RouteContext(intent_id="x", verified_user_id=CUSTOMER, session=scoped(shared)(SessionRequest("x")), mode="plan")
    return asyncio.run(spec.run(intent, ctx)), llm, shared


def test_asking_which_purchase_is_refused_and_the_workflow_then_plans_the_disputes():
    outcome, llm, shared = run(
        [
            call("ask_customer", details=["purchase"], question="Which purchases do you think earned the wrong rewards?"),
            call("start_rewards_correction", purchases=[]),
        ]
    )
    assert "do not ask the customer which purchase" in llm.calls[1]["messages"][-1]["content"]
    assert outcome.status is OutcomeStatus.NEEDS_CONFIRMATION and outcome.proposal is not None
    tools = {step.tool for step in outcome.proposal.steps}
    assert tools == {"submit_cash_back_dispute_0589"}
    assert len(outcome.proposal.steps) == 3
    assert not shared.database.cash_back_disputes.data, "nothing is submitted until the customer says yes"


def test_asking_for_something_else_about_a_rewards_complaint_is_not_refused():
    outcome, _, _ = run([call("ask_customer", details=["card_type"], question="Which card?")])
    assert outcome.status is OutcomeStatus.NEEDS_INFO
