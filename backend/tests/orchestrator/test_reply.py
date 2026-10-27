import pytest
from pydantic import ValidationError

from orchestrator.auth import IDENTITY_SLOT
from orchestrator.decision import CLARIFY_SLOT, CONFIRM_SLOT, Action, IntentDecision
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.plan import Waiting
from orchestrator.reply import Reply, compose_reply
from orchestrator.types import Intent, IntentKind, Question

K = IntentKind
FALLBACK = "Is there anything I can help you with?"


def I(id, summary, kind=K.LOOKUP_INFO, slots=None):
    return Intent(id=id, kind=kind, summary=summary, slots=slots or {})


INTENTS = {
    "bal": I("bal", "check my balance"),
    "frz": I("frz", "freeze my card", K.FREEZE_CARD),
    "fee": I("fee", "sort out my fees", K.FEE_OR_INTEREST_ADJUSTMENT),
    "cls": I("cls", "close my old card", K.CLOSE_CARD),
}


def done(id, text=""):
    return RouteOutcome(intent_id=id, status=OutcomeStatus.DONE, text=text)


def failed(id, error="route failed: RuntimeError"):
    return RouteOutcome(intent_id=id, status=OutcomeStatus.FAILED, error=error)


def ask(id, *slots):
    qs = tuple(Question(intent_id=id, slot=s) for s in slots)
    return IntentDecision(intent_id=id, action=Action.ASK, reason="r", questions=qs)


def escalate(id, reason="confidence below the escalation threshold"):
    return IntentDecision(intent_id=id, action=Action.ESCALATE, reason=reason)


def test_finished_work_is_reported_with_the_routes_own_words():
    reply = compose_reply(INTENTS, outcomes=[done("bal", "Your balance is $120."), done("frz")])
    assert "Your balance is $120." in reply.text
    assert "Done: freeze my card" in reply.text
    assert reply.questions == () and reply.failed_intent_ids == ()


def test_partial_work_says_what_was_done():
    partial = RouteOutcome(
        intent_id="cls", status=OutcomeStatus.PARTIAL, text="I logged the reason", error="step 2 failed"
    )
    reply = compose_reply(INTENTS, outcomes=[partial])
    assert "only finished part of this (close my old card): I logged the reason" in reply.text
    assert reply.failed_intent_ids == ("cls",)


def test_a_failure_names_the_request_offers_a_human_and_hides_the_internal_error():
    reply = compose_reply(INTENTS, outcomes=[failed("fee"), failed("cls")])
    assert "sort out my fees" in reply.text and "close my old card" in reply.text
    assert reply.text.count("human agent") == 1
    assert "RuntimeError" not in reply.text and "route failed" not in reply.text
    assert reply.failed_intent_ids == ("fee", "cls")


def test_a_skipped_dependent_counts_as_a_problem_too():
    skipped = RouteOutcome(
        intent_id="cls", status=OutcomeStatus.SKIPPED, error="dependency 'fee' did not complete"
    )
    reply = compose_reply(INTENTS, outcomes=[failed("fee"), skipped])
    assert reply.failed_intent_ids == ("fee", "cls")


def test_an_escalation_says_a_human_is_taking_it_over():
    reply = compose_reply(INTENTS, decisions=[escalate("fee")])
    assert "handing this to a human agent: sort out my fees" in reply.text
    assert reply.escalated_intent_ids == ("fee",)


def test_questions_come_as_one_batch_tied_to_their_intents_with_wording_filled_in():
    reply = compose_reply(INTENTS, decisions=[ask("frz", "card_id"), ask("fee", CLARIFY_SLOT)])
    assert [(q.intent_id, q.slot) for q in reply.questions] == [("frz", "card_id"), ("fee", CLARIFY_SLOT)]
    assert all(q.text for q in reply.questions)
    assert reply.questions[0].text == "What is the card id for: freeze my card?"
    assert reply.questions[1].text == "Can you tell me more about: sort out my fees?"
    assert reply.text.count("I need a little more information") == 1


def test_a_confirmation_question_says_what_will_happen():
    reply = compose_reply(INTENTS, decisions=[ask("cls", CONFIRM_SLOT)])
    assert reply.questions[0].text == "Please confirm: close my old card (yes or no)?"


def test_a_slot_the_customer_already_gave_is_not_asked_again():
    intents = {"frz": I("frz", "freeze my card", K.FREEZE_CARD, {"card_id": "c1"})}
    reply = compose_reply(intents, decisions=[ask("frz", "card_id", "reason")])
    assert [q.slot for q in reply.questions] == ["reason"]


def test_when_every_question_is_already_answered_there_is_no_question_block():
    intents = {"frz": I("frz", "freeze my card", K.FREEZE_CARD, {"card_id": "c1"})}
    reply = compose_reply(intents, decisions=[ask("frz", "card_id")])
    assert reply.questions == () and "I need a little more information" not in reply.text


def test_the_same_question_twice_is_asked_once():
    reply = compose_reply(INTENTS, decisions=[ask("frz", "card_id"), ask("frz", "card_id")])
    assert len(reply.questions) == 1


def test_identity_questions_for_several_intents_collapse_into_one():
    decisions = [ask("bal", IDENTITY_SLOT), ask("frz", IDENTITY_SLOT), ask("cls", IDENTITY_SLOT)]
    reply = compose_reply(INTENTS, decisions=decisions)
    assert len(reply.questions) == 1 and reply.questions[0].slot == IDENTITY_SLOT
    assert reply.text.count("verify your identity") == 1


def test_something_waiting_says_what_it_is_waiting_for():
    waiting = [Waiting(intent_id="cls", blocked_by=("fee",)), Waiting(intent_id="bal", blocked_by=("gone",))]
    reply = compose_reply(INTENTS, waiting=waiting)
    assert "close my old card once sort out my fees is done" in reply.text
    assert "check my balance once the earlier step is done" in reply.text


def test_one_message_covers_results_problems_escalations_and_questions_in_that_order():
    reply = compose_reply(
        INTENTS,
        outcomes=[done("bal", "Your balance is $120."), failed("frz")],
        decisions=[escalate("fee"), ask("cls", CONFIRM_SLOT)],
    )
    text = reply.text
    positions = [
        text.index("Your balance is $120."),
        text.index("ran into a problem"),
        text.index("handing this to a human agent"),
        text.index("Please confirm"),
    ]
    assert positions == sorted(positions)


def test_with_nothing_to_say_it_falls_back_to_an_open_question():
    assert compose_reply(INTENTS).text == FALLBACK


def test_reply_is_frozen_and_strict():
    reply = compose_reply(INTENTS)
    with pytest.raises(ValidationError):
        reply.text = "changed"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Reply(text="t", questions=(), escalated_intent_ids=(), failed_intent_ids=(), surprise="x")
