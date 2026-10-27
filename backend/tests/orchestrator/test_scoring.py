import asyncio

import pytest
from pydantic import ValidationError

from orchestrator.bands import Band, band
from orchestrator.scoring import IntentScore, ScriptedScorer, score_intents
from orchestrator.types import Decomposition, Intent, IntentKind


def decomposition(*ids):
    return Decomposition(
        intents=tuple(
            Intent(id=i, kind=IntentKind.FREEZE_CARD, summary="freeze my card") for i in ids
        )
    )


def score(intent_id="a", kind=0.9, slots=0.9, scope=0.9, error=None):
    return IntentScore(
        intent_id=intent_id,
        kind_confidence=kind,
        slots_confidence=slots,
        scope_confidence=scope,
        error=error,
    )


def run(scorer, ids=("a",), timeout=1.0):
    return asyncio.run(score_intents(scorer, "msg", decomposition(*ids), timeout=timeout))


@pytest.mark.parametrize(
    ("kind", "slots", "scope", "expected"),
    [(0.5, 0.9, 0.95, 0.5), (0.99, 0.7, 0.95, 0.7), (0.99, 0.9, 0.6, 0.6)],
)
def test_confidence_is_the_weakest_of_the_three_answers(kind, slots, scope, expected):
    assert score(kind=kind, slots=slots, scope=scope).confidence == expected


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_each_answer_must_be_a_probability(bad):
    with pytest.raises(ValidationError):
        score(slots=bad)


def test_score_is_frozen_and_strict():
    s = score()
    with pytest.raises(ValidationError):
        s.intent_id = "b"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        IntentScore(
            intent_id="a",
            kind_confidence=1,
            slots_confidence=1,
            scope_confidence=1,
            surprise="x",
        )


def test_flat_sets_all_three_answers():
    s = IntentScore.flat("a", 0.92)
    assert (s.kind_confidence, s.slots_confidence, s.scope_confidence) == (0.92, 0.92, 0.92)
    assert band(s.confidence) is Band.PROCEED


def test_a_score_with_an_error_has_zero_confidence_whatever_the_numbers_say():
    s = score(kind=1.0, slots=1.0, scope=1.0, error="timed out")
    assert s.confidence == 0.0
    assert band(s.confidence) is Band.ESCALATE


def test_unscored_escalates_and_says_why():
    s = IntentScore.unscored("a", "no answer")
    assert (s.confidence, s.error) == (0.0, "no answer")


def test_scores_come_back_one_per_intent_in_the_decompositions_order():
    scorer = ScriptedScorer([score("b", kind=0.7), score("a", kind=0.95)])
    result = run(scorer, ids=("a", "b"))
    assert [(s.intent_id, s.kind_confidence) for s in result] == [("a", 0.95), ("b", 0.7)]


def test_a_scorer_that_raises_escalates_every_intent_without_raising():
    result = run(ScriptedScorer(error=RuntimeError("boom with card 4111111111111111")), ids=("a", "b"))
    assert [s.confidence for s in result] == [0.0, 0.0]
    assert all(s.error == "scorer failed: RuntimeError" for s in result)


def test_a_slow_scorer_times_out_and_escalates_every_intent():
    result = run(ScriptedScorer([score()], delay=0.5), ids=("a",), timeout=0.01)
    assert result[0].confidence == 0.0
    assert result[0].error == "scorer timed out after 0.01s"


def test_an_intent_the_scorer_skipped_is_escalated_and_the_rest_are_kept():
    result = run(ScriptedScorer([score("a", kind=0.95)]), ids=("a", "b"))
    assert result[0].confidence == 0.9 and result[0].error is None
    assert result[1].confidence == 0.0
    assert result[1].error == "scorer returned no score for this intent"


def test_two_scores_for_one_intent_are_not_trusted():
    result = run(ScriptedScorer([score("a"), score("a", kind=1.0)]), ids=("a",))
    assert result[0].confidence == 0.0
    assert result[0].error == "scorer returned more than one score for this intent"


def test_scores_for_intents_that_do_not_exist_are_ignored():
    result = run(ScriptedScorer([score("a"), score("ghost")]), ids=("a",))
    assert [s.intent_id for s in result] == ["a"]


def test_a_scorer_returning_something_else_escalates():
    result = run(ScriptedScorer(["not a score"]), ids=("a",))
    assert result[0].confidence == 0.0
    assert result[0].error == "scorer failed: TypeError"


def test_no_intents_means_no_scores_and_no_call():
    scorer = ScriptedScorer(error=RuntimeError("must not be called"))
    assert run(scorer, ids=()) == ()
