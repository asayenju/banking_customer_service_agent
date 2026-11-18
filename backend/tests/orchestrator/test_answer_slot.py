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
