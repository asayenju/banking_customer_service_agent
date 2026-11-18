"""Thanks, with nothing waiting for an answer, ends the conversation kindly and starts no request."""

import pytest

from orchestrator.unclear import CLOSING_REPLY, is_closing
from scenario_kit import GOOD_CLAIMS, I, K, S, World, interp


@pytest.mark.parametrize(
    "said",
    ["thanks", "Thank you!", "thank you so much", "ah no problem thank you.", "that's all, thanks", "ok great thanks a lot", "perfect, thank you", "thanks bye", "nothing else, I appreciate it"],
)
def test_thanks_and_goodbyes_are_closings(said):
    assert is_closing(said), said


@pytest.mark.parametrize(
    "said",
    ["thanks, now freeze my card", "thank you but my balance is wrong", "no", "yes please", "hi", "can you raise my limit", "no problem, close it", ""],
)
def test_a_request_tucked_into_a_thank_you_or_a_plain_answer_is_not_a_closing(said):
    assert not is_closing(said), said


def test_thanks_with_nothing_waiting_gets_a_kind_goodbye_and_no_model_is_asked():
    w = World([interp(I("x", K.FREEZE_CARD, {"card_id": "c1"}), claims=GOOD_CLAIMS)], [[S("x", 0.99)]])
    w.turn("freeze my card c1")
    asked_before = len(w.decomposer.histories)
    result = w.turn("ah no problem thank you.")
    assert result.reply.text == CLOSING_REPLY and result.decisions == () and result.outcomes == ()
    assert len(w.decomposer.histories) == asked_before and len(w.scorer.asked) == 1, "nothing was interpreted or scored"
    assert w.orch.active_ids == (), "and no request was made of the thanks"


def test_a_no_thank_you_to_an_open_question_is_an_answer_not_a_goodbye():
    from orchestrator.mocks import ScriptedRoute

    w = World(
        [interp(I("x", K.FREEZE_CARD, missing=("card_id",)), claims=GOOD_CLAIMS), interp()],
        [[S("x", 0.96)], [S("x", 0.96)]],
        direct=ScriptedRoute(),
    )
    first = w.turn("freeze my card")
    assert first.reply.questions, "a question is open"
    answer = w.turn("no thank you")
    assert answer.reply.text != CLOSING_REPLY, "the open question decides what this is"
    assert len(w.decomposer.histories) == 2, "it went to the decomposer like any other reply"


def test_thanks_closes_a_stale_what_do_you_mean_and_it_is_not_asked_again():
    from orchestrator.decision import CLARIFY_SLOT
    from orchestrator.types import IntentKind
    from orchestrator.scoring import IntentScore

    unsure = IntentScore(intent_id="d", kind_confidence=0.8, slots_confidence=1.0, scope_confidence=0.99)
    dispute = I("d", IntentKind.DISPUTE_TRANSACTION, {"merchant": "Zoom Video"}, summary="dispute the Zoom Video charge of $15.83")
    w = World([interp(dispute, claims=GOOD_CLAIMS), interp()], [[unsure]])
    first = w.turn("i think there is a zoom charge")
    assert [q.slot for q in first.reply.questions] == [CLARIFY_SLOT], "an unsure reading is open as a question"
    result = w.turn("thank you")
    assert result.reply.text == CLOSING_REPLY and not result.reply.questions
    assert w.orch.active_ids == (), "the stale question is closed, so it never comes back"


def test_a_no_thanks_to_a_what_do_you_mean_is_still_an_answer():
    from orchestrator.decision import CLARIFY_SLOT
    from orchestrator.types import IntentKind
    from orchestrator.scoring import IntentScore

    unsure = IntentScore(intent_id="d", kind_confidence=0.8, slots_confidence=1.0, scope_confidence=0.99)
    dispute = I("d", IntentKind.DISPUTE_TRANSACTION, {"merchant": "Zoom Video"}, summary="dispute the Zoom Video charge of $15.83")
    w = World([interp(dispute, claims=GOOD_CLAIMS), interp()], [[unsure], [unsure]])
    w.turn("i think there is a zoom charge")
    assert w.turn("no thanks").reply.text != CLOSING_REPLY
