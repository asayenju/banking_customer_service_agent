import pytest

from orchestrator.grounding import drop_unasked_handoff
from orchestrator.types import Conflict, Decomposition, Intent, IntentKind

K = IntentKind


def I(id, kind, depends_on=()):  # noqa: E741
    return Intent(id=id, kind=kind, summary=id, depends_on=depends_on)


def kinds(d):
    return [i.kind for i in d.intents]


def test_a_hand_off_with_no_word_for_a_person_is_dropped_when_other_intents_remain() -> None:
    d = Decomposition(intents=(I("a", K.OPEN_ACCOUNT), I("b", K.REQUEST_HUMAN)))
    out = drop_unasked_handoff("I want a checking account with no overdraft fees.", d)
    assert kinds(out) == [K.OPEN_ACCOUNT]


@pytest.mark.parametrize(
    "message",
    [
        "Can I speak to a human?",
        "I really need someone to investigate this.",
        "Please put me through to your manager.",
        "Can you call me back about this?",
        "Transfer me to an agent.",
    ],
)
def test_any_word_that_could_ask_for_a_person_keeps_it(message) -> None:
    d = Decomposition(intents=(I("a", K.OPEN_ACCOUNT), I("b", K.REQUEST_HUMAN)))
    assert drop_unasked_handoff(message, d) == d


def test_a_lone_hand_off_is_never_dropped() -> None:
    d = Decomposition(intents=(I("a", K.REQUEST_HUMAN),))
    assert drop_unasked_handoff("I am so stressed about everything", d) == d


def test_references_to_the_dropped_intent_are_removed() -> None:
    d = Decomposition(
        intents=(I("a", K.REQUEST_HUMAN), I("b", K.CLOSE_CARD, depends_on=("a",)), I("c", K.OPEN_ACCOUNT)),
        conflicts=(Conflict(intent_ids=("a", "b"), reason="r"), Conflict(intent_ids=("b", "c"), reason="r2")),
    )
    out = drop_unasked_handoff("close my card and open an account", d)
    assert kinds(out) == [K.CLOSE_CARD, K.OPEN_ACCOUNT]
    assert out.intents[0].depends_on == ()
    assert [c.reason for c in out.conflicts] == ["r2"]


def test_other_kinds_are_untouched() -> None:
    d = Decomposition(intents=(I("a", K.CLOSE_CARD), I("b", K.FREEZE_CARD)))
    assert drop_unasked_handoff("freeze and close it", d) == d
