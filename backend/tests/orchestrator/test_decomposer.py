import asyncio

import pytest
from pydantic import ValidationError

from orchestrator.decomposer import Interpretation, ScriptedDecomposer
from orchestrator.types import Decomposition, Intent, IntentKind


def intent(id="a"):
    return Intent(id=id, kind=IntentKind.FREEZE_CARD, summary="freeze my card")


def ask(decomposer, message="m", open_intents=()):
    return asyncio.run(decomposer.interpret(message, open_intents))


def test_an_interpretation_defaults_to_nothing_found():
    i = Interpretation()
    assert i.decomposition.intents == () and i.confirmed == () and i.declined == ()
    assert i.claims == {}


def test_an_interpretation_carries_what_the_customer_said_this_turn():
    i = Interpretation(
        decomposition=Decomposition(intents=(intent(),)),
        confirmed=("a",),
        declined=("b",),
        claims={"email": "x@example.com"},
    )
    assert i.confirmed == ("a",) and i.declined == ("b",) and i.claims["email"] == "x@example.com"


def test_an_interpretation_is_frozen_and_strict():
    with pytest.raises(ValidationError):
        Interpretation().confirmed = ("a",)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Interpretation(surprise="x")


def test_the_scripted_decomposer_returns_its_interpretations_in_order():
    first = Interpretation(decomposition=Decomposition(intents=(intent("a"),)))
    second = Interpretation(confirmed=("a",))
    decomposer = ScriptedDecomposer([first, second])
    assert ask(decomposer) is first and ask(decomposer) is second


def test_it_records_each_message_and_the_open_intents_it_was_given():
    decomposer = ScriptedDecomposer([Interpretation(), Interpretation()])
    ask(decomposer, "one")
    ask(decomposer, "two", (intent("a"), intent("b")))
    assert decomposer.calls == [("one", ()), ("two", ("a", "b"))]


def test_a_scripted_exception_is_raised():
    decomposer = ScriptedDecomposer([RuntimeError("boom")])
    with pytest.raises(RuntimeError):
        ask(decomposer)


def test_running_out_of_script_is_loud():
    with pytest.raises(AssertionError, match="more often than scripted"):
        ask(ScriptedDecomposer([]))
