"""The conversations that went wrong in live use, scripted from their transcripts (no model, no Jev call).

Each one used to end with a hand-off or a loop. Here a low or hesitant score asks what the customer means, the same
question is never shown twice, and a person comes only after the repairs fail.
"""

from scenario_kit import GOOD_CLAIMS, I, S, World, decisions, interp

from orchestrator.decision import Action
from orchestrator.orchestrator import MAX_ASKS
from orchestrator.scoring import IntentScore
from orchestrator.types import IntentKind as K


def test_fatima_silver_card_read_scored_0_53_is_asked_about_not_handed_to_a_person():
    # "I want to see my silver reward card" scored 0.53 and went to a person.
    w = World([interp(I("see", K.ACCOUNT_BALANCE_OR_HISTORY, {"card": "silver"}), claims=GOOD_CLAIMS)], [[S("see", 0.53)]])
    r = w.turn("I want to see my silver reward card")
    assert decisions(r)["see"].action is Action.ASK
    assert r.reply.escalated_intent_ids == () and "human agent" not in r.reply.text


def test_fatima_with_a_close_second_reading_is_asked_either_or():
    score = IntentScore(
        intent_id="see", kind_confidence=0.5, slots_confidence=1.0, scope_confidence=0.95, alternative="DISPUTE_TRANSACTION"
    )
    w = World([interp(I("see", K.ACCOUNT_BALANCE_OR_HISTORY, {"card": "silver"}), claims=GOOD_CLAIMS)], [[score]])
    r = w.turn("I want to see my silver reward card")
    assert "Do you want me to see your balances or recent transactions, or dispute a charge?" in r.reply.text


def test_a_hedged_lost_wallet_report_is_asked_about_never_escalated():
    # "I think I might have lost my wallet" scored 0.75 to 0.92 and the reply to "tell me more" was handed to a person.
    report = I("lost", K.REPORT_FRAUD_OR_LOSS, {"what": "wallet"})
    w = World([interp(report, claims=GOOD_CLAIMS)] + [interp() for _ in range(MAX_ASKS)], [[S("lost", 0.75)]] * (MAX_ASKS + 1))
    results = [w.turn("I think I might have lost my wallet")] + [w.turn("yes I did") for _ in range(MAX_ASKS - 1)]
    assert [decisions(r)["lost"].action for r in results] == [Action.ASK] * MAX_ASKS
    assert all(r.reply.escalated_intent_ids == () for r in results)


def test_a_short_reply_scored_alone_asks_instead_of_handing_over():
    # The reply to "tell me more" (the Zoom dispute) was scored with no context at 0.4 and escalated.
    w = World([interp(I("zoom", K.DISPUTE_TRANSACTION, {"merchant": "Zoom"}), claims=GOOD_CLAIMS)], [[S("zoom", 0.4)]])
    r = w.turn("it was a Zoom charge I never made")
    assert decisions(r)["zoom"].action is Action.ASK and r.reply.escalated_intent_ids == ()


def test_the_scorer_being_down_asks_what_they_mean_and_never_hands_over():
    w = World([interp(I("see", K.ACCOUNT_BALANCE_OR_HISTORY), claims=GOOD_CLAIMS)], [[IntentScore.unscored("see", "scorer failed: TimeoutError")]])
    r = w.turn("show me my balance")
    assert decisions(r)["see"].action is Action.ASK and r.reply.escalated_intent_ids == ()


def test_an_email_change_shows_the_exact_change_and_waits_for_a_yes_at_any_score():
    w = World([interp(I("mail", K.UPDATE_PROFILE, {"new_email": "a@b.co"}), claims=GOOD_CLAIMS)], [[S("mail", 0.99)]])
    r = w.turn("change my email to a@b.co")
    assert decisions(r)["mail"].action is Action.ASK and w.direct.calls == []
    assert [q.slot for q in r.reply.questions] == ["confirmation"]
