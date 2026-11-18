"""What has been done in the conversation is known to the specialists, and a plan says what it will undo."""

import asyncio
import json
from datetime import date

from agent.handoff import describe, done_events
from agent.workflows.base import Reply, WorkflowContext
from agent.workflows.credit_limit import CreditLimit, LimitFacts
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks
from banking_tools.types import AuditEntry, Reversibility
from datetime import datetime
from orchestrator.conflicts import heads_up
from orchestrator.decision import CONFIRM_SLOT, Action, IntentDecision
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.orchestrator import Orchestrator
from orchestrator.reply import compose_reply
from orchestrator.routes import Route
from orchestrator.types import Intent, IntentKind as K, Question
from orchestrator.decomposer import ScriptedDecomposer
from scenario_kit import GOOD_CLAIMS, I, ScriptedDirectory, RECORD, SequenceScorer, S, interp


def test_the_replacement_sentence_says_the_old_card_closes():
    db = BankingSession.from_task(next(t for t in load_tasks() if t["id"] == "task_055")).database
    entry = AuditEntry(
        audit_id="a1", intent_id="i", tool="order_replacement_credit_card_7291", reversibility=Reversibility.IRREVERSIBLE,
        arguments={"credit_card_account_id": "cc_224959b99e_plat", "reason": "fraud_suspected"}, ok=True, timestamp=datetime(2025, 11, 14),
    )
    said = describe(db, entry)
    assert "replacement was ordered" in said and "the old card closes" in said
    assert done_events(db, [entry]) == [said]


class Spy:
    def __init__(self):
        self.done = []

    async def run(self, intent, ctx):
        self.done.append(ctx.done_so_far)
        return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="ok")


def test_the_orchestrator_hands_each_route_what_has_been_done_so_far():
    spy = Spy()
    lines = ["A dispute was filed for a CloudSync Storage charge (unauthorized fraudulent charge)."]
    orch = Orchestrator(
        decomposer=ScriptedDecomposer([interp(I("x", K.FREEZE_CARD, {"card_id": "c1"}), claims=GOOD_CLAIMS)]),
        scorer=SequenceScorer([S("x", 0.99)]),
        directory=ScriptedDirectory(RECORD),
        runners={Route.DIRECT: spy, Route.SUB_AGENT: spy},
        ledger=lambda: lines,
    )
    asyncio.run(orch.handle_turn("m1", "freeze my card c1"))
    assert spy.done == [tuple(lines)]


def test_the_specialist_is_shown_what_has_been_done():
    from agent.specialist import Specialist
    from agent.specialists.credit import CREDIT_SPECIALIST
    from agent.llm_tools import FakeToolLLM
    from knowledge.index import Document, KnowledgeIndex
    from agent.toolbox import SecretVault

    session = BankingSession()
    spec = Specialist(CREDIT_SPECIALIST, llm=FakeToolLLM([]), index=KnowledgeIndex({"d": Document("d", "t", "x")}), database=session.database, vault=SecretVault())
    intent = Intent(id="i", kind=K.CREDIT_LIMIT_REQUEST, summary="raise my limit")
    body = json.loads(spec._request(intent, verified=True, done=("A replacement was ordered for the Gold card; the old card closes when a replacement is ordered.",)))
    assert body["already done in this conversation"] == ["A replacement was ordered for the Gold card; the old card closes when a replacement is ordered."]
    assert "already done in this conversation" not in json.loads(spec._request(intent, verified=True))


def limit_world(close=True, replacement=True):
    session = BankingSession.from_task(next(t for t in load_tasks() if t["id"] == "task_050"))
    db = session.database
    card = next(a for a in db.credit_card_accounts.data.values() if a["user_id"] == "224959b99e" or "Gold" in a["card_type"])
    if close:
        card["status"] = "CLOSED"
    if replacement:
        db.credit_card_orders.data["o1"] = {"order_id": "o1", "credit_card_account_id": card["account_id"], "user_id": card["user_id"], "status": "ORDERED", "order_date": "11/14/2025"}
    return db, card


def test_a_card_closed_by_its_replacement_is_explained_not_just_called_closed():
    db, card = limit_world()
    ctx = WorkflowContext(db, card["user_id"], date(2025, 11, 14), {})
    said = CreditLimit().decide(ctx, LimitFacts(credit_card_account_id=card["account_id"], requested_increase_amount=400))
    assert isinstance(said, Reply) and "closed when its replacement was ordered" in said.text and "new card" in said.text


def test_a_card_closed_for_another_reason_keeps_the_plain_answer():
    db, card = limit_world(replacement=False)
    ctx = WorkflowContext(db, card["user_id"], date(2025, 11, 14), {})
    said = CreditLimit().decide(ctx, LimitFacts(credit_card_account_id=card["account_id"], requested_increase_amount=400))
    assert isinstance(said, Reply) and said.text == "That card account is closed, so its limit cannot be raised."


def test_a_plan_says_what_it_will_do_to_something_else_the_customer_asked_for():
    assert heads_up([K.ORDER_REPLACEMENT_CARD], [K.CREDIT_LIMIT_REQUEST])[0].startswith("Heads-up: ordering the replacement closes your current card")
    assert heads_up([K.DISPUTE_TRANSACTION], [K.CREDIT_LIMIT_REQUEST])[0].startswith("Heads-up: a dispute that is still open")
    assert heads_up([K.ORDER_REPLACEMENT_CARD], []) == [] and heads_up([K.FREEZE_CARD], [K.CREDIT_LIMIT_REQUEST]) == []
    assert heads_up([K.ORDER_REPLACEMENT_CARD, K.DISPUTE_TRANSACTION], [K.CREDIT_LIMIT_REQUEST]).__len__() == 2


def test_the_reply_that_shows_the_plan_carries_the_heads_up_before_it():
    intents = {
        "r": Intent(id="r", kind=K.ORDER_REPLACEMENT_CARD, summary="send me a new card"),
        "l": Intent(id="l", kind=K.CREDIT_LIMIT_REQUEST, summary="raise my limit"),
    }
    plan = Question(intent_id="r", slot=CONFIRM_SLOT, text="Here is exactly what I will do:\n1. Order a replacement\nShall I go ahead? (yes or no)", kind="yes_no")
    decisions = [
        IntentDecision(intent_id="r", action=Action.ASK, reason="plan", questions=(plan,)),
        IntentDecision(intent_id="l", action=Action.ASK, reason="held", questions=(Question(intent_id="l", slot="clarification"),)),
    ]
    reply = compose_reply(intents, decisions=decisions)
    assert reply.text.index("Heads-up") < reply.text.index("Here is exactly what I will do")
    assert [q.slot for q in reply.questions] == [CONFIRM_SLOT], "the question about the increase still waits its turn"
    alone = compose_reply({"r": intents["r"]}, decisions=decisions[:1])
    assert "Heads-up" not in alone.text, "no other request is open, so there is nothing to warn about"
