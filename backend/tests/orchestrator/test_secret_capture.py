import asyncio

from scenario_kit import GOOD_CLAIMS, I, S, World, interp

from orchestrator.decision import CONFIRM_SLOT
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.orchestrator import SECRET_SLOTS
from orchestrator.types import IntentKind as K
from orchestrator.types import Question

PIN = "2589"


class Capture:
    """A capture hook that records what it was told."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, *, expecting_secret=False):
        self.calls.append((text, expecting_secret))
        return text.replace(PIN, "{{secret:pin}}") if expecting_secret else text


class AsksForThePin:
    def __init__(self):
        self.asked = False

    async def run(self, intent, ctx):
        if not intent.slots.get("new_pin"):
            return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.NEEDS_INFO, questions=(Question(intent_id=intent.id, slot="new_pin"),))
        return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="done")


def world(capture, route):
    from orchestrator.auth import ScriptedDirectory
    from orchestrator.decomposer import ScriptedDecomposer
    from orchestrator.mocks import ScriptedRoute
    from orchestrator.orchestrator import Orchestrator
    from orchestrator.routes import Route
    from scenario_kit import RECORD, SequenceScorer

    # UPDATE_PROFILE needs no yes, so the route runs at once and its question is the pending one.
    script = [interp(I("p", K.UPDATE_PROFILE, {"card_id": "c1"}), claims=GOOD_CLAIMS), interp(I("p", K.UPDATE_PROFILE, {"card_id": "c1", "new_pin": "{{secret:pin}}"}))]
    decomposer = ScriptedDecomposer(script)
    orch = Orchestrator(
        decomposer=decomposer,
        scorer=SequenceScorer([S("p", 0.99)], [S("p", 0.99)]),
        directory=ScriptedDirectory(RECORD),
        runners={Route.DIRECT: route, Route.SUB_AGENT: ScriptedRoute()},
        capture_secrets=capture,
    )
    return orch, decomposer


def test_the_message_after_a_question_about_a_pin_is_captured_as_one_even_with_no_cue_word():
    capture = Capture()
    orch, decomposer = world(capture, AsksForThePin())
    first = asyncio.run(orch.handle_turn("m1", "reset my pin"))
    assert "new_pin" in first.reply.text.replace(" ", "_") or first.reply.questions
    assert capture.calls[0] == ("reset my pin", False), "nothing was expected on the first message"
    asyncio.run(orch.handle_turn("m2", f"Set it to {PIN}"))
    assert capture.calls[1] == (f"Set it to {PIN}", True), "a PIN was just asked for"
    assert all(PIN not in text for text, _ in [(c, 0) for c in [m for m, _ in decomposer.calls]]), "the decomposer never saw the digits"


def test_a_question_about_anything_else_does_not_make_the_next_digits_secret():
    capture = Capture()

    class AsksForTheCard:
        async def run(self, intent, ctx):
            return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.NEEDS_INFO, questions=(Question(intent_id=intent.id, slot="card_id"),))

    orch, _ = world(capture, AsksForTheCard())
    asyncio.run(orch.handle_turn("m1", "reset my pin"))
    asyncio.run(orch.handle_turn("m2", "the card ending 4821"))
    assert capture.calls[1] == ("the card ending 4821", False)


def test_the_secret_slots_are_exactly_the_ones_the_audit_log_redacts():
    from banking_tools.session import REDACTED_ARGUMENTS

    assert SECRET_SLOTS == REDACTED_ARGUMENTS


def test_asking_for_a_yes_does_not_count_as_asking_for_a_pin():
    assert CONFIRM_SLOT not in SECRET_SLOTS
