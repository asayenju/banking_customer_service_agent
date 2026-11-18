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


def test_questions_that_gather_information_or_a_preference_carry_no_tag():
    db = world()
    db.credit_card_accounts.data[BSILVER]["current_balance"] = "$0.00"  # nothing owed: the offers come next
    base = dict(credit_card_account_id=BSILVER, pay_from_checking_account_id=None, waiver_answer=None, credit_answer=None, keeps_card=False)
    ctx = WorkflowContext(db, USER, date(2026, 11, 17), {})
    reason = CreditClosure().decide(ctx, ClosureFacts(reason=None, **base))
    assert isinstance(reason, NeedInfo) and [(a.slot, a.kind) for a in reason.questions] == [("closure_reason", "")]
    offer = CreditClosure().decide(ctx, ClosureFacts(reason="not_using_card", **base))
    assert isinstance(offer, NeedInfo) and [(a.slot, a.kind) for a in offer.questions] == [("credit_answer", "")]


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
            return [IntentScore.flat(i.id, 0.4) for i in decomposition.intents]  # unsure: the code asks "tell me more"

    task = next(t for t in load_tasks() if t["id"] == "task_078")
    index = KnowledgeIndex({"doc_x": Document("doc_x", "Internal: x", "lost card protocol")})
    freeze = Interpretation(decomposition=Decomposition(intents=(Intent(id="f", kind=K.FREEZE_CARD, summary="freeze it"),)))
    main.app.state.factory = lambda: build_conversation(
        decomposer=ScriptedDecomposer([freeze]), scorer=Scores(), llm=FakeToolLLM([]), index=index, session=BankingSession.from_task(task)
    )
    main.app.state.conversations = {}
    body = TestClient(main.app).post("/conversations/k1/messages", json={"message_id": "m1", "message": "freeze it"}).json()
    assert body["asks"] and all(set(a) == {"slot", "kind"} for a in body["asks"])
    assert [a["kind"] for a in body["asks"]] == [""], "a 'tell me more' question is free text, never a yes or no"


def test_the_reply_keeps_the_kind_of_a_question_it_rewords_or_repeats():
    """The bug: the reply was rebuilt from slot and text only, so a tagged question reached the chat untagged."""
    from orchestrator.decision import Action, IntentDecision
    from orchestrator.reply import compose_reply
    from orchestrator.types import Intent, IntentKind as K

    intent = Intent(id="x", kind=K.OPEN_ACCOUNT, summary="open a savings account")
    tagged = Question(intent_id="x", slot="fund_now", text="Would you like me to move the opening deposit?", kind="yes_no")
    plain = Question(intent_id="x", slot="fund_amount", text="How much would you like to move?")
    for question, kind in ((tagged, "yes_no"), (plain, "")):
        decision = IntentDecision(intent_id="x", action=Action.ASK, reason="r", questions=(question,))
        assert [q.kind for q in compose_reply({"x": intent}, decisions=[decision]).questions] == [kind]
    # a question with no text of its own is worded by the reply, and still keeps its kind
    bare = Question(intent_id="x", slot="fund_now", kind="yes_no")
    decision = IntentDecision(intent_id="x", action=Action.ASK, reason="r", questions=(bare,))
    (only,) = compose_reply({"x": intent}, decisions=[decision]).questions
    assert only.kind == "yes_no" and only.text


def test_the_please_confirm_question_reaches_the_chat_as_a_yes_or_no():
    """Through the whole orchestrator: a write the customer has not said yes to is confirmed with a tagged question."""
    from orchestrator.types import IntentKind as K
    from scenario_kit import GOOD_CLAIMS, I, S, World, interp

    w = World([interp(I("x", K.CLOSE_CARD, {"card_id": "c1"}), claims=GOOD_CLAIMS)], [[S("x", 0.99)]])
    result = w.turn("close my card")
    assert [(q.slot, q.kind) for q in result.reply.questions] == [("confirmation", "yes_no")]


def test_no_workflow_question_is_tagged_so_buttons_belong_only_to_confirmations_that_run_something():
    """The rule: a question that gathers a detail, a preference or an agreement to a charge is typed. Buttons come only
    from the plan card and the 'please confirm' question that gates a write, both made by the orchestrator."""
    import pathlib

    workflows = pathlib.Path(__file__).resolve().parents[2] / "src" / "agent" / "workflows"
    tagged = [p.name for p in workflows.glob("*.py") if 'kind="yes_no"' in p.read_text()]
    assert tagged == []
