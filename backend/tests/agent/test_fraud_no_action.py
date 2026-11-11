"""Slice 6 of the fraud composer: a report with no tool behind it gets the knowledge base's guidance, a plain
statement of what cannot be done and an offer of a person; never a flag, a score or an invented action."""

import asyncio
import json

import pytest

from agent.dispatcher import Name, RecordKinds, SubAgentDispatcher
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist
from agent.specialists.fraud import FRAUD_SPECIALIST, answer_check
from agent.toolbox import SecretVault
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.executor import OutcomeStatus
from orchestrator.types import Intent, IntentKind as K

TASK = next(t for t in load_tasks() if t["id"] == "task_078")
INDEX = KnowledgeIndex({"doc_x": Document("doc_x", "Internal: security hold", "a security hold cannot be cleared by chat")})
GOOD = (
    "I can't change your online banking password or lock your account from here, and I can't see inside a "
    "security review. Would you like me to connect you with a person on the security team?"
)


def finish(message):
    return AssistantTurn(tool_calls=(ToolCall(id="c", name="finish", arguments_json=json.dumps({"message": message})),), usage=Usage(input_tokens=1))


def run(script):
    shared = BankingSession.from_task(TASK)
    llm = FakeToolLLM(script)
    spec = Specialist(FRAUD_SPECIALIST, llm=llm, index=INDEX, database=shared.database, vault=SecretVault())
    it = Intent(id="r", kind=K.REPORT_FRAUD_OR_LOSS, summary="someone got into my account", slots={})
    session = scoped(shared)(SessionRequest(it.id, mode="plan", approved=None))
    ctx = RouteContext(intent_id=it.id, verified_user_id="mc78a5b9d2", session=session, mode="plan", approved=None)
    outcome = asyncio.run(SubAgentDispatcher({Name.FRAUD: spec}, RecordKinds(shared.database)).run(it, ctx))
    return outcome, shared, llm


def test_a_good_answer_is_given_and_nothing_is_changed():
    outcome, shared, _ = run([finish(GOOD)])
    assert outcome.status is OutcomeStatus.DONE and outcome.text == GOOD and shared.audit_log == ()


@pytest.mark.parametrize(
    "bad",
    [
        "Your card was flagged for fraud by our system. Would you like a person to call you?",
        "Your account has a fraud score of 12, which is why it is on hold. Want a person to review it?",
        "I've reported this to the fraud team and locked your account. Would you like a person?",
        "I have changed your password for you.",
        "There is nothing more I can do here.",
    ],
)
def test_an_answer_that_discloses_invents_or_does_not_offer_a_person_is_refused_and_the_model_retries(bad):
    outcome, _, llm = run([finish(bad), finish(GOOD)])
    assert outcome.status is OutcomeStatus.DONE and outcome.text == GOOD, "the first answer never reached the customer"
    assert len(llm.calls) == 2, "the model was told why and tried again"


def test_the_composer_cannot_hand_over_by_itself():
    escalate = AssistantTurn(tool_calls=(ToolCall(id="e", name="escalate", arguments_json=json.dumps({"reason": "x"})),), usage=Usage(input_tokens=1))
    outcome, _, _ = run([escalate, finish(GOOD)])
    assert outcome.status is OutcomeStatus.DONE and outcome.text == GOOD
    assert "escalate" not in FRAUD_SPECIALIST.terminals


def test_the_check_itself_accepts_a_plain_offer_and_rejects_each_break():
    assert answer_check(GOOD) is None
    for broken in (
        "It was flagged. Want a person?",
        "We will block it. Want a human?",
        "I will lock the account. Want a person?",
        "I'll change your password now. Want a person?",
        "It has a high risk score. Want a person?",
        "I can help with that.",
    ):
        assert answer_check(broken), broken
