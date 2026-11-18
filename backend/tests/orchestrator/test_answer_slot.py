"""A reply to the one open question lands in the slot that was asked, even when the decomposer re-sends the request.
(A freeze needs no yes, so the route runs and asks at once; the mechanism is the same for any kind.)"""

from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.types import Question
from scenario_kit import GOOD_CLAIMS, I, K, S, World, interp


class AskingRoute:
    """Asks for the card until it has it, and remembers what it was given."""

    def __init__(self):
        self.seen = []

    async def run(self, intent, ctx):
        self.seen.append(dict(intent.slots))
        if not intent.slots.get("card_id"):
            q = Question(intent_id=intent.id, slot="card_id", text="Which card do you mean?")
            return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.NEEDS_INFO, questions=(q,))
        return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="ok")


def test_a_reply_is_kept_under_the_asked_slot_when_the_decomposer_resends_other_slots():
    route = AskingRoute()
    w = World(
        [interp(I("x", K.FREEZE_CARD, {"note": "75"}), claims=GOOD_CLAIMS), interp(I("x", K.FREEZE_CARD, {"which": "green"}))],
        [[S("x", 0.99)], [S("x", 0.99)]],
        direct=route, sub=route,
    )
    w.turn("freeze my card")
    w.turn("the green one")
    assert route.seen[-1].get("card_id") == "the green one"
    assert route.seen[-1].get("note") == "75"  # what was known is not erased



def test_a_polite_yes_to_the_question_on_screen_adds_nothing():
    from orchestrator.llm_decomposer import confirms_understanding

    for said in ("yes that would be great", "yes sounds good", "yes that works", "sure, perfect"):
        assert confirms_understanding(said), said
    assert not confirms_understanding("yes the silver card as well")
    assert not confirms_understanding("yes but pay it from savings")


def test_a_pick_is_not_overwritten_with_yes_when_the_decomposer_also_calls_the_reply_confirmed():
    """The reply to "which card?" is the answer. A decomposer that also lists the request as confirmed must not
    turn the answer into "yes" (the customer would be asked the same question again, and again)."""
    route = AskingRoute()
    w = World(
        [interp(I("x", K.FREEZE_CARD, {}), claims=GOOD_CLAIMS), interp(confirmed=("x",))],
        [[S("x", 0.99)], [S("x", 0.99)]],
        direct=route,
        sub=route,
    )
    w.turn("freeze my card")
    w.turn("the green one")
    assert route.seen[-1].get("card_id") == "the green one"


def test_a_plain_yes_to_a_question_is_still_its_answer():
    route = AskingRoute()
    w = World(
        [interp(I("x", K.FREEZE_CARD, {}), claims=GOOD_CLAIMS), interp(confirmed=("x",))],
        [[S("x", 0.99)], [S("x", 0.99)]],
        direct=route,
        sub=route,
    )
    w.turn("freeze my card")
    w.turn("yes")
    assert route.seen[-1].get("card_id") == "yes"


def test_only_a_reply_that_opens_with_a_yes_counts_as_agreeing():
    from orchestrator.llm_decomposer import says_yes

    for said in ("yes", "Yes, I am sure", "sure, go ahead", "ok that works"):
        assert says_yes(said), said
    for said in ("the green one", "your EcoCard (credit card)", "1", "no thanks", "not sure which"):
        assert not says_yes(said), said
