import pytest

from orchestrator.request_type import CommunicationType as T
from orchestrator.request_type import unrequested_writes
from orchestrator.types import IntentKind as K


def test_an_action_request_keeps_its_write_kind() -> None:
    assert unrequested_writes([("a", K.CLOSE_CARD, T.ACTION_REQUEST, "")]) == (frozenset(), {})


@pytest.mark.parametrize("kind", [K.ACCOUNT_BALANCE_OR_HISTORY, K.LOOKUP_INFO, K.GENERAL_QUESTION])
def test_reads_are_left_alone_whatever_the_type(kind) -> None:
    for t in T:
        assert unrequested_writes([("a", kind, t, "")]) == (frozenset(), {})


def test_other_is_left_alone() -> None:
    assert unrequested_writes([("a", K.OTHER, T.UNCLEAR, "")]) == (frozenset(), {})


def test_a_write_that_was_not_asked_for_is_dropped_when_another_intent_remains() -> None:
    rows = [("a", K.GENERAL_QUESTION, T.INFORMATION_REQUEST, ""), ("b", K.FEE_OR_INTEREST_ADJUSTMENT, T.COMPLAINT, "")]
    assert unrequested_writes(rows) == ({"b"}, {})


@pytest.mark.parametrize(
    ("kind_type", "read"),
    [
        (T.PROBLEM_STATEMENT, K.ACCOUNT_BALANCE_OR_HISTORY),
        (T.COMPLAINT, K.ACCOUNT_BALANCE_OR_HISTORY),
        (T.PRODUCT_SHOPPING, K.GENERAL_QUESTION),
        (T.INFORMATION_REQUEST, K.GENERAL_QUESTION),
        (T.UNCLEAR, K.OTHER),
    ],
)
def test_when_nothing_was_asked_the_only_write_becomes_the_read_that_fits(kind_type, read) -> None:
    assert unrequested_writes([("a", K.OPEN_ACCOUNT, kind_type, "")]) == (frozenset(), {"a": read})


def test_several_unrequested_writes_all_become_reads_rather_than_leaving_nothing() -> None:
    rows = [("a", K.OPEN_ACCOUNT, T.PRODUCT_SHOPPING, ""), ("b", K.CLOSE_CARD, T.PROBLEM_STATEMENT, "")]
    assert unrequested_writes(rows) == (frozenset(), {"a": K.GENERAL_QUESTION, "b": K.ACCOUNT_BALANCE_OR_HISTORY})


def test_every_communication_type_has_a_read_to_fall_back_to() -> None:
    for t in T:
        if t is not T.ACTION_REQUEST:
            assert unrequested_writes([("a", K.CLOSE_CARD, t, "")])[1]


def test_telling_the_bank_a_card_is_lost_is_a_report_even_as_a_problem_statement() -> None:
    lost = "I think I lost my wallet"
    assert unrequested_writes([("a", K.REPORT_FRAUD_OR_LOSS, T.PROBLEM_STATEMENT, lost)]) == (frozenset(), {})
    assert unrequested_writes([("a", K.REPORT_FRAUD_OR_LOSS, T.COMPLAINT, lost)])[1]


def test_a_locked_card_is_not_a_loss_report_even_as_a_problem_statement() -> None:
    rows = [("a", K.REPORT_FRAUD_OR_LOSS, T.PROBLEM_STATEMENT, "my cards are locked")]
    assert unrequested_writes(rows) == (frozenset(), {"a": K.ACCOUNT_BALANCE_OR_HISTORY})
