import pytest

from orchestrator.evidence import unsupported_ids
from orchestrator.types import IntentKind

K = IntentKind
MESSAGE = "I need my card frozen. I really need someone to look at this, it’s urgent."


def bad(*rows, message=MESSAGE):
    return unsupported_ids(message, rows)


def test_a_quote_found_in_the_message_is_supported() -> None:
    assert bad(("a", K.FREEZE_CARD, "my card frozen"), ("b", K.GENERAL_QUESTION, "look at this")) == frozenset()


def test_quote_matching_ignores_case_spacing_and_curly_quotes() -> None:
    row = ("b", K.GENERAL_QUESTION, "IT'S   urgent")
    assert bad(("a", K.FREEZE_CARD, "my card frozen"), row) == frozenset()


def test_a_quote_not_in_the_message_is_unsupported_when_another_intent_remains() -> None:
    assert bad(("a", K.FREEZE_CARD, "my card frozen"), ("b", K.CLOSE_CARD, "close it")) == {"b"}


def test_an_empty_quote_is_unsupported() -> None:
    assert bad(("a", K.FREEZE_CARD, "my card frozen"), ("b", K.CLOSE_CARD, "")) == {"b"}


def test_the_only_intent_is_never_dropped() -> None:
    assert bad(("a", K.CLOSE_CARD, "not in the message")) == frozenset()


def test_every_intent_unsupported_drops_none() -> None:
    assert bad(("a", K.CLOSE_CARD, "close it"), ("b", K.FREEZE_CARD, "freeze it")) == frozenset()


@pytest.mark.parametrize(
    "message, quote",
    [
        ("Please get these sorted out, I need help.", "I need help"),
        ("I absolutely need paper statements mailed to me.", "paper statements mailed to me"),
    ],
)
def test_a_hand_off_or_profile_change_needs_words_that_can_ask_for_it(message, quote) -> None:
    for kind in (K.REQUEST_HUMAN, K.UPDATE_PROFILE):
        rows = (("a", K.OPEN_ACCOUNT, message.split()[0]), ("b", kind, quote))
        assert unsupported_ids(message, rows) == {"b"}


def test_a_hand_off_quote_with_a_person_word_is_supported() -> None:
    rows = (("a", K.FREEZE_CARD, "my card frozen"), ("b", K.REQUEST_HUMAN, "need someone to look at this"))
    assert bad(*rows) == frozenset()


def test_a_profile_quote_with_a_contact_word_is_supported() -> None:
    message = "Change my email to a@b.com and close the card."
    rows = (("a", K.CLOSE_CARD, "close the card"), ("b", K.UPDATE_PROFILE, "Change my email"))
    assert unsupported_ids(message, rows) == frozenset()


@pytest.mark.parametrize(
    "quote",
    ["my card was stolen", "I lost my debit card", "a charge I didn\u2019t make", "unauthorized transactions"],
)
def test_a_fraud_or_loss_quote_with_a_loss_word_is_supported(quote) -> None:
    message = f"Close my old account. Also {quote}."
    rows = (("a", K.CLOSE_CARD, "Close my old account"), ("b", K.REPORT_FRAUD_OR_LOSS, quote))
    assert unsupported_ids(message, rows) == frozenset()


def test_a_blocked_card_is_not_a_fraud_or_loss_report() -> None:
    message = "Open a savings account. My cards are locked at the ATM."
    rows = (("a", K.OPEN_ACCOUNT, "Open a savings account"), ("b", K.REPORT_FRAUD_OR_LOSS, "cards are locked at the ATM"))
    assert unsupported_ids(message, rows) == {"b"}


def test_a_quote_with_one_word_changed_is_still_supported() -> None:
    message = "I'm looking to open a second checking account and a savings account."
    rows = (
        ("a", K.OPEN_ACCOUNT, "I want to open a second checking account and a savings account."),
        ("b", K.GENERAL_QUESTION, "nothing like this was said anywhere here"),
    )
    assert unsupported_ids(message, rows) == {"b"}


def test_a_quote_of_words_never_written_is_unsupported_even_if_a_few_are_common() -> None:
    message = "Please freeze my card and tell me the balance."
    rows = (("a", K.FREEZE_CARD, "freeze my card"), ("b", K.CLOSE_CARD, "close the account for good"))
    assert unsupported_ids(message, rows) == {"b"}
