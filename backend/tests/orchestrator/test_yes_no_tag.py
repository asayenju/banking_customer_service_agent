"""Yes and No buttons come from a tag set where the question is written, never from how the question reads."""

from datetime import date

from fastapi.testclient import TestClient

import main
from agent.workflows.base import NeedInfo, WorkflowContext
from agent.workflows.credit_closure import ClosureFacts, CreditClosure
from agent.workflows.credit_limit import CreditLimit, LimitFacts
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks
from orchestrator.types import Question

USER = "224959b99e"  # Yuki Nakamura (task_055): Platinum $75, Business Silver $125, one Green checking
BSILVER = "cc_224959b99e_bsilver"


def world():
    return BankingSession.from_task(next(t for t in load_tasks() if t["id"] == "task_055")).database


def test_a_question_is_untagged_unless_the_code_that_asks_it_says_yes_or_no():
    assert Question(intent_id="x", slot="s").kind == ""
    assert Question(intent_id="x", slot="s", kind="yes_no").kind == "yes_no"


def test_the_retention_offer_is_a_yes_or_no_and_the_closure_reason_is_not():
    db = world()
    db.credit_card_accounts.data[BSILVER]["current_balance"] = "$0.00"  # nothing owed: the offers come next
    base = dict(credit_card_account_id=BSILVER, pay_from_checking_account_id=None, waiver_answer=None, credit_answer=None, keeps_card=False)
    ctx = WorkflowContext(db, USER, date(2026, 11, 17), {})
    reason = CreditClosure().decide(ctx, ClosureFacts(reason=None, **base))
    assert isinstance(reason, NeedInfo) and [(a.slot, a.kind) for a in reason.questions] == [("closure_reason", "")]
    offer = CreditClosure().decide(ctx, ClosureFacts(reason="not_using_card", **base))
    assert isinstance(offer, NeedInfo) and [(a.slot, a.kind) for a in offer.questions] == [("credit_answer", "yes_no")]


def test_how_much_would_you_like_is_never_a_yes_or_no():
    db = world()
    db.credit_card_accounts.data["cc_224959b99e_plat"]["credit_limit"] = "$5,000.00"
    out = CreditLimit().decide(
        WorkflowContext(db, USER, date(2026, 11, 17), {}), LimitFacts(credit_card_account_id="cc_224959b99e_plat", requested_increase_amount=10_000_000)
    )
    assert isinstance(out, NeedInfo) and "Would you like to request" in out.questions[0].text
    assert out.questions[0].kind == "", "an amount, or 'this or a smaller one', has no Yes or No answer"


def test_the_api_sends_each_question_with_its_kind():
    from agent.assembly import build_conversation
    from agent.llm_tools import FakeToolLLM
    from knowledge.index import Document, KnowledgeIndex
    from orchestrator.decomposer import Interpretation, ScriptedDecomposer
    from orchestrator.scoring import IntentScore
    from orchestrator.types import Decomposition, Intent, IntentKind as K

    class Scores:
        async def score(self, message, decomposition):
            return [IntentScore.flat(i.id, 0.4) for i in decomposition.intents]  # unsure: the code asks "please confirm"

    task = next(t for t in load_tasks() if t["id"] == "task_078")
    index = KnowledgeIndex({"doc_x": Document("doc_x", "Internal: x", "lost card protocol")})
    freeze = Interpretation(decomposition=Decomposition(intents=(Intent(id="f", kind=K.FREEZE_CARD, summary="freeze it"),)))
    main.app.state.factory = lambda: build_conversation(
        decomposer=ScriptedDecomposer([freeze]), scorer=Scores(), llm=FakeToolLLM([]), index=index, session=BankingSession.from_task(task)
    )
    main.app.state.conversations = {}
    body = TestClient(main.app).post("/conversations/k1/messages", json={"message_id": "m1", "message": "freeze it"}).json()
    assert body["asks"] and all(set(a) == {"slot", "kind"} for a in body["asks"])
