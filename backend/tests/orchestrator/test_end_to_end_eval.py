import asyncio

from orchestrator.decision import Action
from orchestrator.decomposer import ScriptedDecomposer
from orchestrator.evaluation.end_to_end import render, run_end_to_end, summarize
from orchestrator.evaluation.tasks import EvalCase
from orchestrator.scoring import IntentScore
from orchestrator.types import Decomposition, Intent, IntentKind

from scenario_kit import interp

K = IntentKind


class Sure:
    """Scores every intent the same."""

    def __init__(self, confidence):
        self.confidence = confidence

    async def score(self, message, decomposition):
        return [IntentScore.flat(i.id, self.confidence) for i in decomposition.intents]


def intent(id, kind, slots=None):
    return Intent(id=id, kind=kind, summary=id, slots=slots or {"card_id": "c1"})


def go(cases, scripts, confidence=0.99):
    by_message = {c.message: s for c, s in zip(cases, scripts, strict=True)}

    class ByMessage:
        async def interpret(self, message, open_intents):
            return await ScriptedDecomposer([by_message[message]]).interpret(message, open_intents)

    found, not_found = asyncio.run(
        run_end_to_end(cases, ByMessage, lambda: Sure(confidence), concurrency=1)
    )
    return found, not_found, summarize(cases, found, not_found)


def case(id, stated):
    return EvalCase(task_id=id, message=f"message {id}", stated=tuple(stated), expected=tuple(stated))


def test_an_invented_reversible_write_is_counted_as_having_run_with_no_yes():
    cases = [case("quiet", [])]
    found, _, s = go(cases, [interp(intent("a", K.FREEZE_CARD))])
    assert [(o.kind, o.requested, o.action) for o in found] == [(K.FREEZE_CARD, False, Action.RUN)]
    assert s.unrequested_ran == 1 and s.messages_with_unrequested_ran == 1
    assert s.unrequested_skipping_the_yes == 1
    assert s.irreversible_ran_without_yes == 0
    assert dict(s.ran_by_kind) == {K.FREEZE_CARD: 1}


def test_an_invented_irreversible_write_asks_and_never_runs():
    cases = [case("quiet", [])]
    _, _, s = go(cases, [interp(intent("a", K.CLOSE_CARD))])
    assert s.unrequested_asked == 1 and s.unrequested_ran == 0
    assert s.unrequested_skipping_the_yes == 0
    assert s.irreversible_ran_without_yes == 0


def test_a_requested_write_is_counted_apart_and_a_missed_one_is_reported():
    cases = [case("both", [K.FREEZE_CARD, K.CLOSE_CARD])]
    _, not_found, s = go(cases, [interp(intent("a", K.FREEZE_CARD))])
    assert s.requested_ran == 1 and s.unrequested_ran == 0
    assert s.requested_not_found == 1 and not_found == {"both": 1}


def test_a_write_below_95_percent_asks_before_it_runs():
    cases = [case("unsure", [K.FREEZE_CARD])]
    _, _, s = go(cases, [interp(intent("a", K.FREEZE_CARD))], confidence=0.9)
    assert s.requested_asked == 1 and s.requested_ran == 0


def test_reads_are_not_counted_as_writes_and_the_report_names_the_invariant():
    cases = [case("read", [])]
    found, _, s = go(cases, [interp(intent("a", K.GENERAL_QUESTION))])
    assert found == [] and s.writes == 0
    assert "must be 0" in render(s)
