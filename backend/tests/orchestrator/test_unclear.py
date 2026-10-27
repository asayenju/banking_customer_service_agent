import inspect

import pytest
from pydantic import ValidationError

from orchestrator.decision import Action
from orchestrator.types import Intent, IntentKind
from orchestrator.unclear import (
    HELP_MENU,
    MAX_MENUS,
    HelpState,
    handle_unclear,
    is_unclear,
    offer_help,
    precheck,
)

UNCLEAR = [
    "",
    "   \n\t ",
    "hi",
    "Hello!",
    "hey there",
    "good morning",
    "help",
    "I need help",
    "can you help me?",
    "hello, can you help",
    "I have a question",
    "can I ask something",
    "thanks",
    "???",
    "😀",
    "12345",
    "asdfghjkl",
    "qwerty uiop",
    "lkjhgf",
    "aaaaaaa",
    "Привет",
    "你好",
    "مرحبا",
    "こんにちは",
]
REAL = [
    "hi, please freeze my card",
    "help me freeze my card",
    "I need help with my card",
    "what is my balance",
    "freeze it",
    "hello, my card was stolen",
    "I want to talk to a human",
    "I want to ask about fees",
    "my cvv was exposed",
    "yes",
    "no",
]


@pytest.mark.parametrize("message", UNCLEAR)
def test_unclear_messages_are_caught(message):
    assert is_unclear(message) is True


@pytest.mark.parametrize("message", REAL)
def test_real_requests_are_left_for_the_decomposer(message):
    assert is_unclear(message) is False


@pytest.mark.parametrize("message", UNCLEAR)
def test_precheck_turns_an_unclear_message_into_one_unclear_intent(message):
    d = precheck(message)
    assert d is not None and len(d.intents) == 1
    assert d.intents[0].kind is IntentKind.UNCLEAR_OR_GREETING


@pytest.mark.parametrize("message", REAL)
def test_precheck_returns_nothing_for_a_real_request(message):
    assert precheck(message) is None


def test_the_unclear_intent_does_not_echo_the_message():
    d = precheck("asdf 4111 1111 1111 1111 qwer")
    assert d is not None
    assert "4111" not in d.intents[0].summary


def test_precheck_takes_no_scorer():
    # Unclear messages never reach Jev: there is nothing to score them with here.
    assert list(inspect.signature(precheck).parameters) == ["message"]


def test_the_menu_is_one_fixed_text_for_everyone():
    assert offer_help() == HELP_MENU == offer_help()
    assert list(inspect.signature(offer_help).parameters) == []


@pytest.mark.parametrize(
    "mentions", ["card", "dispute", "transfer", "balance", "human", "verify your identity"]
)
def test_the_menu_names_what_we_can_do(mentions):
    assert mentions in HELP_MENU.lower()


def unclear_intent(id="a"):
    return Intent(id=id, kind=IntentKind.UNCLEAR_OR_GREETING, summary="unclear or greeting")


def test_the_first_three_unclear_turns_each_get_the_menu():
    state = HelpState()
    for shown in range(1, MAX_MENUS + 1):
        decision, state = handle_unclear(unclear_intent(), state)
        assert decision.action is Action.RUN and decision.reason == "showing the help menu"
        assert state.menus_shown == shown


def test_the_next_unclear_turn_after_three_menus_escalates():
    state = HelpState(menus_shown=MAX_MENUS)
    decision, after = handle_unclear(unclear_intent(), state)
    assert decision.action is Action.ESCALATE
    assert decision.reason == "still unclear after 3 help menus"
    assert after == state


def test_a_fresh_state_starts_the_count_over():
    # A turn with a real intent is handled by replacing the state with HelpState().
    assert HelpState().menus_shown == 0


def test_help_state_is_frozen_and_strict():
    with pytest.raises(ValidationError):
        HelpState().menus_shown = 2
    with pytest.raises(ValidationError, match="extra_forbidden"):
        HelpState(surprise="x")
    with pytest.raises(ValidationError):
        HelpState(menus_shown=-1)
