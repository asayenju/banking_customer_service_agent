"""A controller decides what to do with a request (and may ask Jev); the old threshold gate is the fallback."""

import asyncio

import pytest
from scenario_kit import GOOD_CLAIMS, I, S, SequenceScorer, World, interp

from orchestrator.controller import Choice, ControlChoice, ScriptedController
from orchestrator.orchestrator import Orchestrator
from orchestrator.types import IntentKind as K

RUN = ControlChoice(Choice.RUN, reason="clear from the conversation")


def world(script, controller, scores=((),)):
    w = World(script, [list(c) for c in scores])
    w.orch._controller = controller
    return w


def freeze():
    return I("f", K.FREEZE_CARD, {"card_id": "c1"})


def test_a_run_choice_goes_ahead_without_asking_the_scorer_at_all():
    controller = ScriptedController([RUN])
    w = world([interp(freeze(), claims=GOOD_CLAIMS)], controller)
    r = w.turn("freeze my card c1")
    assert w.direct.calls == ["f"] and w.scorer.asked == [], "no score was needed"
    assert any(e.detail == "controller" for e in r.trace.events)


def test_the_controller_is_shown_the_request_the_conversation_and_how_risky_it_is():
    controller = ScriptedController([RUN, RUN])
    w = world([interp(freeze(), claims=GOOD_CLAIMS), interp(I("g", K.ACCOUNT_BALANCE_OR_HISTORY))], controller)
    w.turn("freeze my card c1")
    w.turn("what is my balance")
    first, second = controller.cases
    assert first.request_kind == "FREEZE_CARD" and "reversible" in first.safety and first.details == {"card_id": "c1"}
    assert second.safety == "read only" and [who for who, _ in second.recent] == ["customer", "agent"]
    assert "Freeze a debit card" in first.what_it_can_do


def test_an_ask_choice_puts_the_controllers_own_question_on_screen_and_remembers_it():
    controller = ScriptedController([ControlChoice(Choice.ASK, "Do you want the balance or the recent transactions?", "vague")])
    w = world([interp(I("a", K.ACCOUNT_BALANCE_OR_HISTORY), claims=GOOD_CLAIMS)], controller)
    r = w.turn("show me my card")
    assert [q.text for q in r.reply.questions] == ["Do you want the balance or the recent transactions?"]
    assert w.orch._asked_log["a"] == ["Do you want the balance or the recent transactions?"]


def test_the_same_question_is_never_asked_twice_and_the_gate_decides_instead():
    q = ControlChoice(Choice.ASK, "Which card?", "unsure")
    controller = ScriptedController([q, q])
    w = world([interp(I("a", K.ACCOUNT_BALANCE_OR_HISTORY), claims=GOOD_CLAIMS), interp(I("a", K.ACCOUNT_BALANCE_OR_HISTORY, {"x": "y"}))], controller, [[S("a", 0.99)]])
    w.turn("show me my card")
    second = w.turn("the silver one")
    assert [d.action.value for d in second.decisions] != ["ask"] or second.reply.questions[0].text != "Which card?"
    assert w.scorer.asked, "the threshold gate was used for the repeat"
    assert controller.cases[1].already_asked == ("Which card?",)


def test_handing_to_a_person_on_first_sight_becomes_a_question_not_a_hand_off():
    controller = ScriptedController([ControlChoice(Choice.ESCALATE, "Do you mean the Silver card or the Gold card?", "not sure")])
    w = world([interp(I("a", K.ACCOUNT_BALANCE_OR_HISTORY), claims=GOOD_CLAIMS)], controller)
    r = w.turn("I want to see my card")
    assert r.decisions[0].action.value == "ask" and "human agent" not in r.reply.text
    assert r.reply.questions[0].text == "Do you mean the Silver card or the Gold card?"


def test_handing_to_a_person_is_allowed_once_something_has_been_asked_and_for_a_request_for_a_person():
    ask = ControlChoice(Choice.ASK, "Which one?", "")
    controller = ScriptedController([ask, ControlChoice(Choice.ESCALATE, "", "still unclear after asking")])
    w = world([interp(I("a", K.ACCOUNT_BALANCE_OR_HISTORY), claims=GOOD_CLAIMS), interp(I("a", K.ACCOUNT_BALANCE_OR_HISTORY, {"z": "1"}))], controller)
    w.turn("show me my thing")
    second = w.turn("that one")
    assert second.decisions[0].action.value == "escalate"


@pytest.mark.parametrize("failure", [RuntimeError("model down"), None])
def test_a_controller_that_fails_leaves_the_decision_to_the_thresholds(failure):
    controller = ScriptedController([failure])
    w = world([interp(freeze(), claims=GOOD_CLAIMS)], controller, [[S("f", 0.99)]])
    r = w.turn("freeze my card c1")
    assert w.direct.calls == ["f"] and w.scorer.asked == [["f"]]
    assert any(e.kind.value == "failed" and "controller failed" in (e.reason or "") for e in r.trace.events)


def test_a_slow_controller_is_given_up_on():
    class Slow:
        async def decide(self, case):
            await asyncio.sleep(1)

    w = World([interp(freeze(), claims=GOOD_CLAIMS)], [[S("f", 0.99)]])
    w.orch._controller, w.orch._controller_timeout = Slow(), 0.05
    w.turn("freeze my card c1")
    assert w.direct.calls == ["f"], "fell back to the threshold gate"


def test_a_run_choice_still_asks_for_a_missing_detail_and_for_the_yes_to_an_irreversible_change():
    controller = ScriptedController([RUN, RUN])
    w = world(
        [interp(I("m", K.FREEZE_CARD, missing=("card_id",)), claims=GOOD_CLAIMS), interp(I("c", K.CLOSE_ACCOUNT, {"account_id": "a1"}))],
        controller,
    )
    first = w.turn("freeze my card")
    assert first.decisions[0].action.value == "ask" and [q.slot for q in first.reply.questions] == ["card_id"]
    second = w.turn("close account a1")
    assert [q.slot for q in second.reply.questions][-1:] == ["confirmation"] and w.sub.calls == [] and w.direct.calls == []


def test_jev_is_consulted_only_when_the_controller_asks_and_what_it_says_is_put_in_words():
    seen = []

    class Asks:
        async def decide(self, case):
            seen.append(await case.consult_jev())
            return RUN

    w = World([interp(freeze(), claims=GOOD_CLAIMS)], [[S("f", 0.53)]])
    w.orch._controller = Asks()
    w.turn("freeze my card c1")
    assert w.scorer.asked == [["f"]] and "0.53" in seen[0] and "short summary" in seen[0]
    assert w.direct.calls == ["f"], "a low Jev number alone did not stop it"


def test_without_a_controller_nothing_changes():
    w = World([interp(freeze(), claims=GOOD_CLAIMS)], [[S("f", 0.99)]])
    assert w.orch._controller is None
    w.turn("freeze my card c1")
    assert w.scorer.asked == [["f"]]
