"""The richer way of asking Jev (facts, a Choice over kinds, signals), all behind switches and never deciding alone."""

import asyncio
import types

from orchestrator import jev_state
from orchestrator.jev import KIND_QUESTION, SCOPE_QUESTION, JevScorer, _state
from orchestrator.scoring import ScoreContext
from orchestrator.types import Decomposition, Intent, IntentKind as K


def I(kind=K.ACCOUNT_BALANCE_OR_HISTORY, **slots):  # noqa: E741
    return Intent(id="a", kind=kind, summary="see my silver card", slots=slots)


class Jev:
    def __init__(self, reading=None, human=0.1, answers=0.2, frustration=1.0, broken_extras=False):
        self.reading, self.human, self.answers, self.frustration, self.broken = reading, human, answers, frustration, broken_extras
        self.calls = []

    async def system_one(self, state, questions, *, model=None, timeout=None):
        self.calls.append((state, questions))
        n = lambda p: types.SimpleNamespace(noul=p)  # noqa: E731
        nouls = {KIND_QUESTION: n(0.5), SCOPE_QUESTION: n(0.9)}
        if self.broken:
            return types.SimpleNamespace(nouls=nouls, choices={}, scores={})
        nouls |= {jev_state.SIGNAL_HUMAN: n(self.human), jev_state.SIGNAL_ANSWER: n(self.answers)}
        choices = {jev_state.READING: types.SimpleNamespace(probabilities=self.reading or {})}
        scores = {jev_state.SIGNAL_FRUSTRATION: types.SimpleNamespace(score=self.frustration)}
        return types.SimpleNamespace(nouls=nouls, choices=choices, scores=scores)


def score(client, intent=None, **flags):
    d = Decomposition(intents=(intent or I(),))
    return asyncio.run(JevScorer(client, **flags).score("see my silver card", d))[0]


def test_nothing_extra_is_asked_unless_switched_on():
    fake = Jev()
    s = score(fake, choice=False, signals=False)
    assert set(fake.calls[0][1]) == {KIND_QUESTION, SCOPE_QUESTION}
    assert s.alternative is None and s.signals == ()


def test_the_choice_question_lists_every_kind_once_from_the_capability_text():
    q = jev_state.choice_question()
    assert q["type"] == "choice" and set(q["criteria"]) == {k.value for k in K}


def test_a_sibling_reading_lifts_the_kind_answer_so_history_versus_lookup_does_not_ask():
    s = score(Jev({"ACCOUNT_BALANCE_OR_HISTORY": 0.4, "LOOKUP_INFO": 0.5, "DISPUTE_TRANSACTION": 0.1}), choice=True, signals=False)
    assert abs(s.kind_confidence - 0.9) < 1e-9
    assert s.alternative is None, "the runner-up is too unlikely to ask about"


def test_a_different_reading_that_is_likely_enough_becomes_the_alternative():
    s = score(Jev({"ACCOUNT_BALANCE_OR_HISTORY": 0.5, "DISPUTE_TRANSACTION": 0.4}), choice=True, signals=False)
    assert s.kind_confidence == 0.5 and s.alternative == "DISPUTE_TRANSACTION"


def test_a_write_is_never_the_same_reading_as_another_kind():
    assert not jev_state.same_reading(K.FREEZE_CARD, K.UNFREEZE_CARD)
    assert not jev_state.same_reading(K.CLOSE_CARD, K.CLOSE_ACCOUNT)
    assert jev_state.same_reading(K.ACCOUNT_BALANCE_OR_HISTORY, K.LOOKUP_INFO)


def test_signals_are_returned_for_the_log_and_trouble_with_them_changes_nothing():
    s = score(Jev(human=0.9, answers=0.3, frustration=3.0), choice=False, signals=True)
    assert dict(s.signals) == {jev_state.SIGNAL_HUMAN: 0.9, jev_state.SIGNAL_ANSWER: 0.3, jev_state.SIGNAL_FRUSTRATION: 3.0}
    broken = score(Jev(broken_extras=True), choice=True, signals=True)
    assert broken.kind_confidence == 0.5 and broken.signals == () and broken.alternative is None


def test_the_facts_hold_only_words_and_flags_never_a_digit_or_an_id():
    facts = jev_state.build(
        verified=True, question_slot="card_id", plan_waiting=True, other_open_kinds=[K.CLOSE_CARD, K.FREEZE_CARD], asks_so_far=2
    )
    assert dict(facts)["times_this_request_was_already_asked_about"] == "several times"
    assert not any(ch.isdigit() for name, value in facts for ch in name + value)
    assert "dbc_" not in str(facts)
    state = _state("hello", I(), ScoreContext(facts=facts))
    assert "caller identity verified: yes" in state["situation"]
