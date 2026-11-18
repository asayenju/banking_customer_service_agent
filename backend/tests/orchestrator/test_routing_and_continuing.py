"""The fixes from the Yuki conversation: a card named up front is understood, a plain yes cannot turn an offer into a
hand-off, and a closure that paid a balance goes on as the same request."""

from agent.dispatcher import Name, RecordKinds
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.types import IntentKind as K, Question
from scenario_kit import GOOD_CLAIMS, I, S, World, interp

USER = "224959b99e"  # Yuki Nakamura (task_055): Platinum $75, Business Silver $125, Green checking, a debit card ending 8765


def records():
    session = BankingSession.from_task(next(t for t in load_tasks() if t["id"] == "task_055"))
    return RecordKinds(session.database)


def test_a_credit_card_named_in_words_in_any_card_slot_is_routed_to_the_credit_specialist():
    r = records()
    for slot in ("card_id", "credit_card_account_id", "account_id", "card_or_account_id"):
        assert r.products({slot: "Platinum Rewards Card"}, USER) == {Name.CREDIT}, slot
        assert r.products({slot: "your Business Silver Rewards Card (credit card)"}, USER) == {Name.CREDIT}, slot


def test_a_name_that_fits_two_cards_or_none_is_still_asked_not_guessed():
    r = records()
    assert r.products({"card_id": "rewards card"}, USER) == frozenset(), "Platinum and Business Silver both fit"
    assert r.products({"card_id": "Mystery Card"}, USER) == frozenset()
    assert r.products({"card_id": "Platinum Rewards Card"}, "someone-else") == frozenset(), "never another customer's card"


class Asking:
    """Asks for the waiver answer once, then runs with whatever it was given."""

    def __init__(self):
        self.seen = []

    async def run(self, intent, ctx):
        self.seen.append((intent.kind, dict(intent.slots)))
        if not intent.slots.get("waiver_answer"):
            q = Question(intent_id=intent.id, slot="waiver_answer", text="I can waive the annual fee for a year. Would you like that?")
            return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.NEEDS_INFO, questions=(q,))
        return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="Done: waived")


def test_a_thank_you_yes_to_an_offer_is_its_answer_even_when_the_decomposer_calls_it_a_hand_off():
    route = Asking()
    handoff = I("x", K.REQUEST_HUMAN, {"waiver_answer": "yes thank you"})  # the open request, re-sent as something else
    w = World(
        [interp(I("x", K.FREEZE_CARD, {}), claims=GOOD_CLAIMS), interp(handoff)],
        [[S("x", 0.99)], [S("x", 0.99)]],
        direct=route,
        sub=route,
    )
    w.turn("freeze my card")
    result = w.turn("yes thank you")
    assert not result.reply.escalated_intent_ids, "nobody asked for a person"
    assert route.seen[-1][0] is K.FREEZE_CARD and route.seen[-1][1].get("waiver_answer") == "yes"


def test_a_hand_off_replaces_an_open_request_only_when_the_customer_mentions_a_person():
    route = Asking()
    w = World(
        [interp(I("x", K.FREEZE_CARD, {}), claims=GOOD_CLAIMS), interp(I("x", K.REQUEST_HUMAN, {}))],
        [[S("x", 0.99)], [S("x", 0.99)]],
        direct=route,
        sub=route,
    )
    w.turn("freeze my card")
    assert not w.turn("what does that cost me?").reply.escalated_intent_ids

    route2 = Asking()
    w2 = World(
        [interp(I("x", K.FREEZE_CARD, {}), claims=GOOD_CLAIMS), interp(I("x", K.REQUEST_HUMAN, {}))],
        [[S("x", 0.99)], [S("x", 0.99)]],
        direct=route2,
        sub=route2,
    )
    w2.turn("freeze my card")
    assert w2.turn("I want to talk to a person").reply.escalated_intent_ids == ("x",)


class PayThenAsk:
    """Pays a balance and says what comes next, like the closure of a card that owed money."""

    proposes_plans = False

    def __init__(self):
        self.seen = []

    async def run(self, intent, ctx):
        self.seen.append(dict(intent.slots))
        if not intent.slots.get("closure_reason"):
            follow = Question(intent_id=intent.id, slot="closure_reason", text="Why would you like to close this card?")
            return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="The balance is paid.", follow_up=follow)
        return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="Done: closed")


def test_a_step_that_asks_what_comes_next_keeps_the_same_request_open_with_what_it_knows():
    route = PayThenAsk()
    w = World(
        [interp(I("x", K.FREEZE_CARD, {"card_id": "platinum"}), claims=GOOD_CLAIMS), interp()],
        [[S("x", 0.99)], [S("x", 0.99)]],
        direct=route,
        sub=route,
    )
    first = w.turn("freeze my platinum card")
    assert "The balance is paid." in first.reply.text and first.reply.questions[0].slot == "closure_reason"
    assert w.orch.active_ids == ("x",) and w.orch.done_ids == (), "the request is not over"
    second = w.turn("it costs too much")
    assert route.seen[-1].get("card_id") == "platinum", "the card it already knew is still there"
    assert route.seen[-1].get("closure_reason") == "it costs too much"
    assert w.orch.done_ids == ("x",) and "closed" in second.reply.text


# ---------------------------------------------------------------- a request that goes on must not be blamed for its first step

def test_a_request_that_goes_on_is_not_blamed_for_the_step_an_earlier_plan_already_made():
    """The bug: a card closure paid a balance, asked why, and its second plan was refused as "a change that was not in the
    approved plan", because the alarm compared every write of the request with only the latest plan."""
    from banking_tools import BankingSession, Fidelity
    from orchestrator.context import scoped
    from orchestrator.decision import fingerprint
    from orchestrator.plan_preview import PlanPreview, PlanStep

    card = "dbc_lj82d4f1a9_bluest"
    block = PlanStep(index=1, tool="set_debit_card_recurring_block_7382", args={"card_id": card, "block_recurring": True}, reversible=True, description="Block recurring charges")
    freeze = PlanStep(index=1, tool="freeze_debit_card_3892", args={"card_id": card}, reversible=True, description="Freeze the card")

    class TwoPlans:
        proposes_plans = True

        async def run(self, intent, ctx):
            asked_why = bool(intent.slots.get("closure_reason"))
            if ctx.mode == "plan":
                steps = (freeze,) if asked_why else (block,)
                return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.NEEDS_CONFIRMATION, proposal=PlanPreview.build(intent.id, fingerprint(intent), steps))
            for step in ctx.approved.steps:
                ctx.session.call(step.tool, step.args)
            if asked_why:
                return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="Done: froze it")
            follow = Question(intent_id=intent.id, slot="closure_reason", text="Why would you like to close this card?")
            return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="Done: blocked", follow_up=follow)

    shared = BankingSession(fidelity=Fidelity.TAU2)
    w = World(
        [interp(I("x", K.CLOSE_CARD, {"card_id": card}), claims=GOOD_CLAIMS), interp()],
        [[S("x", 0.99)], [S("x", 0.99)]],
        sub=TwoPlans(),
        session_for=scoped(shared),
    )
    first = w.turn("block my card")
    assert first.reply.questions[0].slot == "closure_reason" and w.orch.active_ids == ("x",)
    second = w.turn("it costs too much")
    assert [o.status for o in second.outcomes] == [OutcomeStatus.DONE], [(o.status, o.error) for o in second.outcomes]
    assert [e.tool for e in shared.audit_log if e.ok] == ["set_debit_card_recurring_block_7382", "freeze_debit_card_3892"]
    assert "stale" not in " ".join(d.reason for d in second.decisions), "the finished step's plan is not kept around to go stale"


# ---------------------------------------------------------------- the waiver offer answered with a restatement

class Offering:
    """Like the credit specialist: plans, and asks the waiver question before it has a plan."""

    proposes_plans = True

    def __init__(self):
        self.seen = []

    async def run(self, intent, ctx):
        self.seen.append((intent.kind, dict(intent.slots)))
        if not intent.slots.get("waiver_answer"):
            q = Question(intent_id=intent.id, slot="waiver_answer", text="I can waive the annual fee for a year. Would you like that?")
            return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.NEEDS_INFO, questions=(q,))
        return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="Done: waived")


def test_a_yes_that_restates_the_offer_is_its_answer_and_makes_no_second_request():
    route = Offering()
    fee_talk = I("t2-f", K.FEE_OR_INTEREST_ADJUSTMENT, {"note": "waive the annual fee"})
    w = World(
        [interp(I("x", K.CLOSE_CARD, {}), claims=GOOD_CLAIMS), interp(fee_talk)],
        [[S("x", 0.99)], [S("x", 0.99)]],
        sub=route,
    )
    first = w.turn("close my card")
    assert first.reply.questions[0].slot == "waiver_answer", "the offer is on screen"
    w.turn("yes, I would love that")
    assert "t2-f" not in w.orch.active_ids, "the restated offer is not a second request"
    assert route.seen[-1][1].get("waiver_answer") == "yes, I would love that"
    assert all(kind is not K.FEE_OR_INTEREST_ADJUSTMENT for kind, _ in route.seen)


def test_a_yes_to_a_plan_is_not_also_the_answer_to_a_tell_me_more_question():
    route = Asking()
    w = World([interp(I("x", K.FREEZE_CARD, {}), claims=GOOD_CLAIMS)], [[S("x", 0.99)]], direct=route, sub=route)
    w.turn("freeze my card")
    w.orch._clarifying["x"] = Question(intent_id="x", slot="clarification", text="To be sure I have this right: x?")
    w.orch._pending_confirm["x"] = "plan-fingerprint"
    from orchestrator.decomposer import Interpretation
    from orchestrator.types import Decomposition

    assert w.orch._clarification("yes", Decomposition(intents=()), frozenset()) is None
    del w.orch._pending_confirm["x"]
    assert w.orch._clarification("yes", Decomposition(intents=()), frozenset()) is not None, "without a plan waiting, the yes does answer it"


def test_a_reply_that_opens_with_yes_cannot_change_the_kind_of_the_request_it_answers():
    """The conversation that failed: the waiver offer was on screen, the customer said "yes, I would love that", the decomposer
    re-sent the open request as a fee review, and the closure was lost to a request nobody made."""
    route = Offering()
    relabelled = I("x", K.FEE_OR_INTEREST_ADJUSTMENT, {"waiver_answer": "yes, I would love that"})
    w = World(
        [interp(I("x", K.CLOSE_CARD, {}), claims=GOOD_CLAIMS), interp(relabelled)],
        [[S("x", 0.99)], [S("x", 0.82)]],
        sub=route,
    )
    w.turn("close my card")
    second = w.turn("yes, I would love that")
    assert w.orch.done_ids == ("x",) or w.orch.active_ids == ("x",)
    assert all(kind is K.CLOSE_CARD for kind, _ in route.seen), "the request is still the closure, whatever the decomposer called it"
    assert route.seen[-1][1].get("waiver_answer") == "yes, I would love that"
    assert not second.reply.escalated_intent_ids and not second.reply.failed_intent_ids


def test_a_reply_without_a_yes_can_still_change_what_a_request_is():
    route = Offering()
    w = World(
        [interp(I("x", K.CLOSE_CARD, {}), claims=GOOD_CLAIMS), interp(I("x", K.FEE_OR_INTEREST_ADJUSTMENT, {}))],
        [[S("x", 0.99)], [S("x", 0.99)]],
        sub=route,
    )
    w.turn("close my card")
    w.turn("actually, I meant a fee on my savings")
    assert route.seen[-1][0] is K.FEE_OR_INTEREST_ADJUSTMENT, "a real change of mind is still honoured"


def test_a_short_answer_to_a_waiting_question_is_not_turned_into_a_general_question():
    """The dispute asked "why are you disputing it?", the customer answered, and the decomposer re-sent the dispute as a general
    question at 0.59: the request that was being worked on was gone and the conversation lost its thread."""
    route = Offering()
    relabelled = I("x", K.GENERAL_QUESTION, {"waiver_answer": "I never used that service"})
    w = World(
        [interp(I("x", K.CLOSE_CARD, {}), claims=GOOD_CLAIMS), interp(relabelled)],
        [[S("x", 0.99)], [S("x", 0.59)]],
        sub=route,
    )
    first = w.turn("close my card")
    assert first.reply.questions[0].slot == "waiver_answer"
    w.turn("I never used that service")
    assert all(kind is K.CLOSE_CARD for kind, _ in route.seen), "it is still the closure it was"
    assert route.seen[-1][1].get("waiver_answer") == "I never used that service"


def test_a_real_change_of_mind_to_an_action_is_still_honoured_while_a_question_is_open():
    route = Offering()
    w = World(
        [interp(I("x", K.CLOSE_CARD, {}), claims=GOOD_CLAIMS), interp(I("x", K.FREEZE_CARD, {"card_id": "c1"}))],
        [[S("x", 0.99)], [S("x", 0.99)]],
        sub=route,
        direct=route,
    )
    w.turn("close my card")
    w.turn("actually just freeze it")
    assert route.seen[-1][0] is K.FREEZE_CARD, "a change to another action is a change of mind, not a mislabelled answer"
