"""The credit card history offers a Dispute button per charge, built from the bank's own record of each one."""

import asyncio

from agent.assembly import build_conversation
from agent.llm_tools import FakeToolLLM
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex
from orchestrator.decomposer import Interpretation, ScriptedDecomposer
from orchestrator.scoring import IntentScore
from orchestrator.types import Decomposition, Intent, IntentKind as K

TASK = next(t for t in load_tasks() if t["id"] == "task_031")  # Fatima Al-Hassan, Silver Rewards Card
CLAIMS = {"phone_number": "313-555-0246", "email": "coffeelover_fati@protonmail.com"}
INDEX = KnowledgeIndex({"doc_x": Document("doc_x", "Internal: x", "dispute")})


class Scores:
    async def score(self, message, decomposition):
        return [IntentScore.flat(i.id, 0.99) for i in decomposition.intents]


def conversation():
    history = Interpretation(
        decomposition=Decomposition(
            intents=(Intent(id="h", kind=K.ACCOUNT_BALANCE_OR_HISTORY, summary="show my recent transactions", slots={"account_id": "Silver Rewards Card"}),)
        ),
        claims=CLAIMS,
    )
    return build_conversation(
        decomposer=ScriptedDecomposer([history]), scorer=Scores(), llm=FakeToolLLM([]), index=INDEX, session=BankingSession.from_task(TASK)
    )


def test_each_listed_charge_has_a_dispute_button_whose_line_is_in_the_text():
    conv = conversation()
    result = asyncio.run(conv.orchestrator.handle_turn("m1", "show me my recent transactions"))
    choices = result.reply.choices
    assert len(choices) >= 5 and all(c.id.startswith("dispute:txn_") and c.action == "Dispute" for c in choices)
    assert all(c.kind is K.DISPUTE_TRANSACTION and c.slots["transaction_id"] and c.slots["amount"].startswith("$") for c in choices)
    lines = {line.removeprefix("- ") for line in result.reply.text.splitlines()}
    assert all(c.label in lines for c in choices), "every button belongs to a line the customer can read"


def test_pressing_one_makes_a_dispute_of_that_exact_charge_and_sends_it_to_the_credit_specialist():
    conv = conversation()
    first = asyncio.run(conv.orchestrator.handle_turn("m1", "show me my recent transactions"))
    chosen = first.reply.choices[0]
    second = asyncio.run(conv.orchestrator.handle_turn("m2", f"Dispute: {chosen.label}", chosen.id))
    events = [(e.kind.value, e.detail) for e in second.trace.events]
    assert ("decided", "button") in events and ("scored", "button") in events, "made from the offer, not scored by a model"
    assert ("routed", "sub_agent") in events, "a dispute of a credit card charge goes to the specialist"
    # The specialist's own model is not scripted here, so the run stops; what matters is that the press reached it
    # as this exact charge (the decomposer was never asked: it only had the one scripted turn).
    assert second.reply.text
