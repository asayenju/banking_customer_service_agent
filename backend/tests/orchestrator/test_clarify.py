"""A reply to "to be sure I have this right..." is more of the same request, read with the whole story (the Zoom charge)."""

from scenario_kit import GOOD_CLAIMS, I, S, World, interp

from orchestrator.decision import CLARIFY_SLOT
from orchestrator.types import IntentKind as K

ASK = "the zoom video one seems wrong. Can u file a dispute there"
REPLY = "the charge in 10/21/2025 feels wrong. I never interacted with zoom before"


def dispute():
    return I("d", K.DISPUTE_TRANSACTION, {"merchant": "Zoom Video"}, summary="dispute the Zoom Video charge of $15.83")


def start(second, scores):
    w = World([interp(dispute(), claims=GOOD_CLAIMS), interp(second)], scores)
    first = w.turn(ASK)
    return w, first


def test_the_first_unsure_dispute_asks_to_be_sure_with_the_request_in_words():
    w, first = start(dispute(), [[S("d", 0.9)]])
    assert [q.slot for q in first.reply.questions] == [CLARIFY_SLOT]
    assert "dispute the Zoom Video charge" in first.reply.questions[0].text and "Is that what you want?" in first.reply.questions[0].text


def test_a_reply_the_decomposer_made_a_vague_new_request_of_is_folded_into_the_open_one_not_escalated():
    vague = I("n1", K.OTHER, summary="the charge in 10/21/2025 feels wrong")
    w, _ = start(vague, [[S("d", 0.9)], [S("d", 0.97)]])
    second = w.turn(REPLY)
    assert "human agent" not in second.reply.text and all(e.kind.value != "escalated" for e in second.trace.events)
    assert w.orch.active_ids == ("d",), "no second request was made"
    assert w.scorer.asked == [["d"], ["d"]], "the open request was scored again, not the vague reply"


def test_the_request_is_scored_on_the_whole_story_not_the_reply_alone():
    vague = I("n1", K.OTHER, summary="feels wrong")
    w, _ = start(vague, [[S("d", 0.9)], [S("d", 0.97)]])
    w.turn(REPLY)
    assert w.scorer.messages == [ASK, f"{ASK}\n{REPLY}"]


def test_a_reply_read_as_a_fraud_report_is_still_the_same_dispute_and_adds_its_details():
    sibling = I("n1", K.REPORT_FRAUD_OR_LOSS, {"merchant": "Zoom Video", "date": "10/21/2025"}, summary="unauthorized Zoom Video charge on 10/21/2025, never used Zoom")
    w, _ = start(sibling, [[S("d", 0.9)], [S("d", 0.97)]])
    w.turn(REPLY)
    kept = w.orch._active.get("d") or w.orch._active.get("n1")
    assert w.orch.active_ids == ("d",) and kept.slots["date"] == "10/21/2025" and "never used Zoom" in kept.summary
    assert "Customer: " + REPLY[:30] in kept.slots["conversation_so_far"]


def test_a_real_new_request_in_the_reply_is_not_swallowed():
    freeze = I("n1", K.FREEZE_CARD, {"card_id": "c1"})
    w, _ = start(freeze, [[S("d", 0.9)], [S("n1", 0.97)]])
    w.turn("actually freeze card c1 right now")
    assert "n1" in w.orch.active_ids or "n1" in w.orch.done_ids


def test_a_yes_to_the_question_settles_what_the_request_is_so_it_goes_ahead_to_the_exact_plan():
    w = World([interp(dispute(), claims=GOOD_CLAIMS), interp(confirmed=["d"])], [[S("d", 0.9)]])
    w.turn(ASK)
    second = w.turn("yes")
    assert w.scorer.asked == [["d"]], "a yes is not scored alone"
    assert all(e.kind.value != "escalated" for e in second.trace.events)
    assert "human agent" not in second.reply.text
    assert [q.slot for q in second.reply.questions] != [CLARIFY_SLOT], "it moves on instead of asking to be sure again"
