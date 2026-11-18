"""The decomposer proposes and the code decides: a plain yes, what it is told, and what it reads of a long reply."""

import asyncio
import json

from banking_tools import BankingSession, Fidelity
from orchestrator.context import scoped
from orchestrator.decision import fingerprint
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.llm import FakeLLMClient
from orchestrator.llm_decomposer import LLMDecomposer
from orchestrator.orchestrator import HISTORY_CHARS, _shorten
from orchestrator.plan_preview import PlanPreview, PlanStep
from orchestrator.types import Intent, IntentKind as K
from scenario_kit import GOOD_CLAIMS, I, S, World, interp

CARD = "dbc_lj82d4f1a9_bluest"


def step(tool, args, text):
    return PlanStep(index=1, tool=tool, args=args, reversible=False, description=text)


class Plans:
    """A route that proposes one irreversible plan per request and runs it once approved."""

    proposes_plans = True

    def __init__(self):
        self.executed = []

    async def run(self, intent, ctx):
        if ctx.mode == "plan":
            plan = PlanPreview.build(intent.id, fingerprint(intent), (step("close_debit_card_4721", {"card_id": CARD, "reason": intent.id}, f"Do {intent.id}"),))
            return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.NEEDS_CONFIRMATION, proposal=plan)
        self.executed.append(intent.id)
        return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text=f"Done: {intent.id}")


def test_a_plain_yes_confirms_every_plan_on_screen_even_with_a_clarification_open_and_nothing_listed_by_the_decomposer():
    """Sofia: a dispute plan, a replacement plan and a question about a limit increase were on screen. "Yes, go ahead." (the plan
    card's own button) confirmed nothing, because the decomposer listed no ids, and she had to say it again."""
    plans = Plans()
    w = World(
        [
            interp(I("d", K.DISPUTE_TRANSACTION, {"transaction_id": "t1"}), I("r", K.ORDER_REPLACEMENT_CARD, {"card_id": CARD}), I("l", K.CREDIT_LIMIT_REQUEST, {}), claims=GOOD_CLAIMS),
            interp(),  # the decomposer lists nothing as confirmed
        ],
        [[S("d", 0.93), S("r", 0.85), S("l", 0.59)], [S("d", 0.93), S("r", 0.85)]],
        sub=plans,
        session_for=scoped(BankingSession(fidelity=Fidelity.TAU2)),
    )
    first = w.turn("dispute that charge, order a new card, and later a higher limit")
    assert [q.slot for q in first.reply.questions].count("confirmation") == 2 and plans.executed == []
    w.turn("Yes, go ahead.")
    assert sorted(plans.executed) == ["d", "r"], "both plans on screen were confirmed by one plain yes"


def test_the_decomposer_is_told_which_requests_have_a_plan_waiting_for_a_yes():
    plans = Plans()
    w = World([interp(I("d", K.DISPUTE_TRANSACTION, {"transaction_id": "t1"}), claims=GOOD_CLAIMS), interp()], [[S("d", 0.93)], [S("d", 0.93)]], sub=plans, session_for=scoped(BankingSession(fidelity=Fidelity.TAU2)))
    w.turn("dispute that charge")
    w.turn("yes")
    assert w.decomposer.awaiting[0] == () and w.decomposer.awaiting[1] == ("d",)


def test_the_model_sees_awaiting_yes_and_the_instruction_for_it():
    client = FakeLLMClient([json.dumps({"intents": [], "conflicts": [], "confirmed": [], "declined": [], "answers": [], "claims": []})])
    open_one = Intent(id="d", kind=K.DISPUTE_TRANSACTION, summary="dispute it")
    asyncio.run(LLMDecomposer(client).interpret("yes", [open_one], (), (), ("d",)))
    sent = client.calls[0]["messages"]
    assert json.loads(sent[1]["content"])["awaiting_yes"] == ["d"]
    assert "awaiting_yes" in sent[0]["content"] and "confirmed" in sent[0]["content"]


def test_a_long_reply_keeps_its_end_in_the_history_not_just_its_start():
    reply = "To go on, I need a little more information. " + "Here is what I found. " * 80 + "Shall I go ahead and close it?"
    kept = _shorten(reply)
    assert len(kept) <= HISTORY_CHARS and kept.startswith("To go on") and kept.endswith("Shall I go ahead and close it?")
    assert _shorten("short") == "short"


# ---------------------------------------------------------------- one thing at a time, and Jev sees the whole reading

class Mixed:
    """The dispute asks for a date; the replacement proposes a plan: Sofia's first two requests."""

    proposes_plans = True

    def __init__(self):
        self.executed, self.dispute_slots = [], []

    async def run(self, intent, ctx):
        if intent.id == "d":
            self.dispute_slots.append(dict(intent.slots))
            if not intent.slots.get("issue_noticed_date"):
                from orchestrator.types import Question

                q = Question(intent_id="d", slot="issue_noticed_date", text="When did you first notice the charge?")
                return RouteOutcome(intent_id="d", status=OutcomeStatus.NEEDS_INFO, questions=(q,))
            return RouteOutcome(intent_id="d", status=OutcomeStatus.DONE, text="Done: disputed")
        if ctx.mode == "plan":
            plan = PlanPreview.build(intent.id, fingerprint(intent), (step("close_debit_card_4721", {"card_id": CARD}, "Order the replacement"),))
            return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.NEEDS_CONFIRMATION, proposal=plan)
        self.executed.append(intent.id)
        return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="Done: replacement ordered")


def sofia_world(route):
    return World(
        [interp(I("d", K.DISPUTE_TRANSACTION, {"transaction_id": "t1"}), I("r", K.ORDER_REPLACEMENT_CARD, {"card_id": CARD}), claims=GOOD_CLAIMS), interp(), interp()],
        [[S("d", 0.93), S("r", 0.85)], [S("d", 0.93), S("r", 0.85)], [S("d", 0.93)]],
        sub=route,
        session_for=scoped(BankingSession(fidelity=Fidelity.TAU2)),
    )


def test_a_plan_and_a_question_are_never_in_the_same_reply_the_action_goes_first():
    route = Mixed()
    w = sofia_world(route)
    first = w.turn("dispute the charge and send me a new card")
    assert [q.slot for q in first.reply.questions] == ["confirmation"], "only the plan: the date question waits"
    assert "When did you first notice" not in first.reply.text
    assert w.orch.ask_counts.get("d", 0) == 0, "a question that waited was not asked, so it has used none of its asks"
    second = w.turn("Yes, go ahead.")
    assert route.executed == ["r"], "the plain yes confirmed the plan on screen"
    assert [q.slot for q in second.reply.questions] == ["issue_noticed_date"], "now the question is asked, alone"
    assert all(not slots.get("issue_noticed_date") for slots in route.dispute_slots), "the yes was not taken as the date"


def test_a_held_question_can_still_be_answered_while_the_plan_waits():
    route = Mixed()
    w = sofia_world(route)
    w.turn("dispute the charge and send me a new card")
    w.turn("I first noticed it today")
    assert route.dispute_slots[-1].get("issue_noticed_date") == "I first noticed it today"


def test_jev_is_shown_the_customers_other_requests_and_what_this_one_waits_for():
    from orchestrator.jev import _state
    from orchestrator.scoring import ScoreContext

    limit = Intent(id="l", kind=K.CREDIT_LIMIT_REQUEST, summary="raise my limit for a renovation", depends_on=("d",))
    dispute = Intent(id="d", kind=K.DISPUTE_TRANSACTION, summary="dispute the $487.50 CloudSync charge")
    context = ScoreContext(requests=tuple((i.id, i.kind.value, i.summary, i.depends_on) for i in (dispute, limit)))
    state = _state("a message with three requests", limit, context)
    assert "dispute a charge" not in state["the_customers_other_requests"] and "dispute transaction: dispute the" in state["the_customers_other_requests"]
    assert "dispute transaction" in state["the_customer_wants_this_after"]
    assert "the_customers_other_requests" not in _state("one request", limit, ScoreContext(requests=((limit.id, limit.kind.value, limit.summary, ()),)))


# ---------------------------------------------------------------- a held question does not climb the repair ladder unseen

def test_a_question_held_back_does_not_move_up_the_repair_ladder_until_it_is_really_shown():
    from orchestrator.decision import Action as A, IntentDecision as D
    from orchestrator.repair import RepairLadder
    from orchestrator.types import Question

    intent = Intent(id="l", kind=K.CREDIT_LIMIT_REQUEST, summary="raise my limit")
    ask = D(intent_id="l", action=A.ASK, reason="r", questions=(Question(intent_id="l", slot="clarification", text="To be sure: raise your limit?"),))
    ladder = RepairLadder()
    for _ in range(4):  # held behind a plan for four turns: not shown, so the wording does not move
        (held,) = ladder.apply({"l": intent}, [ask], lambda i, q: "unused", set())
        assert held.questions[0].text == "To be sure: raise your limit?"
    (first,) = ladder.apply({"l": intent}, [ask], lambda i, q: "unused", {("l", "clarification")})
    assert first.questions[0].text == "To be sure: raise your limit?", "the first time it is shown it is asked as written"
    (second,) = ladder.apply({"l": intent}, [ask], lambda i, q: "unused", {("l", "clarification")})
    assert second.questions[0].text != first.questions[0].text and "still need an answer" not in second.questions[0].text


def test_the_last_rung_only_talks_about_a_card_when_the_question_is_about_one():
    from orchestrator.repair import offer_person

    assert "last four digits of the card" in offer_person("which card?", "card_or_account_id")
    neutral = offer_person("what do you mean?", "clarification")
    assert "last four digits" not in neutral and "say 'person'" in neutral


# ---------------------------------------------------------------- the decomposer is told what is already finished

def test_the_decomposer_is_told_which_requests_are_already_done_so_it_does_not_find_them_again():
    from orchestrator.mocks import ScriptedRoute

    w = World(
        [interp(I("x", K.FREEZE_CARD, {"card_id": "c1"}, summary="freeze my card"), claims=GOOD_CLAIMS), interp()],
        [[S("x", 0.99)], [S("x", 0.99)]],
        direct=ScriptedRoute(),
    )
    w.turn("freeze my card")
    w.turn("anything else?")
    assert w.decomposer.finished[0] == () and w.decomposer.finished[1] == (("FREEZE_CARD", "freeze my card"),)


def test_a_greeting_or_a_question_is_not_listed_as_finished_work():
    from orchestrator.mocks import ScriptedRoute

    w = World([interp(I("q", K.GENERAL_QUESTION, summary="what are your hours"), claims=GOOD_CLAIMS), interp()], [[S("q", 0.99)], [S("q", 0.99)]], direct=ScriptedRoute(), sub=ScriptedRoute())
    w.turn("what are your hours")
    w.turn("thanks for asking again")
    assert w.decomposer.finished[1] == ()


def test_the_model_sees_what_is_already_done_and_the_instruction_not_to_make_it_again():
    client = FakeLLMClient([json.dumps({"intents": [], "conflicts": [], "confirmed": [], "declined": [], "answers": [], "claims": []})])
    asyncio.run(LLMDecomposer(client).interpret("and the limit?", [], (), (), (), (("ORDER_REPLACEMENT_CARD", "order a replacement card"),)))
    sent = client.calls[0]["messages"]
    assert json.loads(sent[1]["content"])["already_done"] == [{"kind": "ORDER_REPLACEMENT_CARD", "summary": "order a replacement card"}]
    assert "already_done" in sent[0]["content"] and "Never make a new intent out of one of them" in sent[0]["content"]
