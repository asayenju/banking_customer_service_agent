import pytest
from pydantic import ValidationError

from orchestrator.auth import (
    IDENTITY_SLOT,
    MAX_FAILED_ATTEMPTS,
    AuthState,
    CustomerRecord,
    requires_verification,
    verify,
    gate,
)
from orchestrator.types import Decomposition, HandOff, Intent, IntentKind

RECORD = CustomerRecord(
    user_id="u_kenji01",
    date_of_birth="03/14/1990",
    email="kenji.tanaka@example.com",
    phone_number="206-555-0293",
    address="12 Pine St, Seattle, WA 98101",
)
OTHER_RECORD = CustomerRecord(
    user_id="u_fatima02",
    date_of_birth="12/05/1993",
    email="fatima@example.com",
    phone_number="313-555-0246",
    address="1923 Oak Park Boulevard, Detroit, MI 48226",
)
RIGHT_EMAIL = {"email": "kenji.tanaka@example.com"}
RIGHT_PHONE = {"phone_number": "206-555-0293"}
RIGHT_DOB = {"date_of_birth": "03/14/1990"}
WRONG_EMAIL = {"email": "someone.else@example.com"}
OPEN = (
    IntentKind.UNCLEAR_OR_GREETING,
    IntentKind.GENERAL_QUESTION,
    IntentKind.REQUEST_HUMAN,
    IntentKind.OTHER,
)


def intent(id, kind):
    return Intent(id=id, kind=kind, summary="s")


def decomposition(*pairs):
    return Decomposition(intents=tuple(intent(i, k) for i, k in pairs))


def locked_state():
    state = AuthState()
    for _ in range(MAX_FAILED_ATTEMPTS):
        state = verify(state, RECORD, WRONG_EMAIL)
    return state


@pytest.mark.parametrize("kind", OPEN)
def test_greetings_hand_offs_and_unknowns_need_no_verification(kind):
    assert requires_verification(kind) is False


@pytest.mark.parametrize("kind", [k for k in IntentKind if k not in OPEN])
def test_every_other_kind_needs_verification(kind):
    assert requires_verification(kind) is True


def test_two_matching_fields_verify_the_caller():
    state = verify(AuthState(), RECORD, {**RIGHT_EMAIL, **RIGHT_PHONE})
    assert state.verified and state.verified_user_id == "u_kenji01"
    assert state.failed_attempts == 0


def test_one_matching_field_is_not_enough_but_is_not_a_failure():
    state = verify(AuthState(), RECORD, RIGHT_EMAIL)
    assert not state.verified
    assert state.failed_attempts == 0


def test_matches_add_up_across_turns():
    state = verify(AuthState(), RECORD, RIGHT_EMAIL)
    state = verify(state, RECORD, RIGHT_DOB)
    assert state.verified


def test_the_same_field_given_twice_counts_once():
    state = verify(AuthState(), RECORD, RIGHT_EMAIL)
    state = verify(state, RECORD, RIGHT_EMAIL)
    assert not state.verified


def test_two_right_and_one_wrong_still_verifies():
    state = verify(AuthState(), RECORD, {**RIGHT_EMAIL, **RIGHT_DOB, "phone_number": "999"})
    assert state.verified


def test_a_wrong_field_is_a_failed_attempt():
    state = verify(AuthState(), RECORD, WRONG_EMAIL)
    assert not state.verified and state.failed_attempts == 1


def test_three_failures_lock_and_a_correct_answer_afterwards_does_not_unlock():
    state = locked_state()
    assert state.locked and state.failed_attempts == MAX_FAILED_ATTEMPTS
    state = verify(state, RECORD, {**RIGHT_EMAIL, **RIGHT_PHONE})
    assert state.locked and not state.verified


def test_a_verified_caller_stays_verified():
    state = verify(AuthState(), RECORD, {**RIGHT_EMAIL, **RIGHT_PHONE})
    assert verify(state, OTHER_RECORD, {"email": "fatima@example.com"}) == state


@pytest.mark.parametrize(
    "claimed",
    [
        {"phone_number": "(206) 555 0293"},
        {"phone_number": "+1 206 555 0293"},
        {"phone_number": "2065550293"},
        {"email": "  Kenji.Tanaka@EXAMPLE.com "},
        {"date_of_birth": "March 14, 1990"},
        {"date_of_birth": "1990-03-14"},
        {"date_of_birth": "3/14/1990"},
        {"address": "12 pine st seattle wa 98101"},
        {"address": "12  PINE ST., Seattle, WA   98101"},
    ],
)
def test_formatting_differences_still_match(claimed):
    # Pair the variant under test with a correct answer for a different field.
    companion = next(c for c in (RIGHT_EMAIL, RIGHT_PHONE, RIGHT_DOB) if not c.keys() & claimed.keys())
    state = verify(AuthState(), RECORD, {**claimed, **companion})
    assert state.verified


@pytest.mark.parametrize(
    "claimed",
    [
        {"date_of_birth": "03/15/1990"},
        {"date_of_birth": "sometime in March"},
        {"phone_number": "206-555-0294"},
        {"address": "13 Pine St, Seattle, WA 98101"},
    ],
)
def test_near_misses_do_not_match(claimed):
    state = verify(AuthState(), RECORD, claimed)
    assert state.failed_attempts == 1 and not state.matched


def test_a_bypass_claim_is_ignored_and_costs_nothing():
    claimed = {
        "bypass_code": "9K2X7M4P1N8Q3R5T6A",
        "supervisor_override": "true",
        "verified": "true",
        "user_id": "u_kenji01",
    }
    state = verify(AuthState(), RECORD, claimed)
    assert not state.verified
    assert state.failed_attempts == 0 and not state.matched


def test_a_bypass_claim_next_to_one_real_field_still_needs_a_second():
    state = verify(AuthState(), RECORD, {**RIGHT_EMAIL, "bypass_code": "9K2X"})
    assert not state.verified


def test_blank_values_are_ignored_not_counted_as_wrong():
    state = verify(AuthState(), RECORD, {"email": "   ", "phone_number": ""})
    assert state.failed_attempts == 0 and not state.matched


def test_a_different_customer_record_starts_the_matches_over_but_keeps_the_failures():
    state = verify(AuthState(), RECORD, WRONG_EMAIL)
    state = verify(state, RECORD, RIGHT_EMAIL)
    state = verify(state, OTHER_RECORD, {"email": "fatima@example.com"})
    assert state.candidate_user_id == "u_fatima02"
    assert len(state.matched) == 1
    assert state.failed_attempts == 1


def test_state_is_frozen_and_strict():
    state = AuthState()
    with pytest.raises(ValidationError):
        state.failed_attempts = 5
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AuthState(surprise="x")


def test_gate_lets_open_intents_through_and_blocks_the_rest_with_a_question_each():
    d = decomposition(
        ("a", IntentKind.UNCLEAR_OR_GREETING),
        ("b", IntentKind.FREEZE_CARD),
        ("c", IntentKind.ACCOUNT_BALANCE_OR_HISTORY),
    )
    result = gate(AuthState(), d)
    assert result.allowed == ("a",)
    assert result.blocked == ("b", "c")
    assert [(q.intent_id, q.slot) for q in result.questions] == [
        ("b", IDENTITY_SLOT),
        ("c", IDENTITY_SLOT),
    ]
    assert result.hand_off is None


def test_gate_lets_everything_through_once_verified():
    state = verify(AuthState(), RECORD, {**RIGHT_EMAIL, **RIGHT_PHONE})
    d = decomposition(("a", IntentKind.FREEZE_CARD), ("b", IntentKind.CLOSE_CARD))
    result = gate(state, d)
    assert result.allowed == ("a", "b") and result.blocked == ()
    assert result.questions == () and result.hand_off is None


def test_gate_hands_off_blocked_intents_after_three_failures_and_asks_nothing():
    d = decomposition(("a", IntentKind.REQUEST_HUMAN), ("b", IntentKind.TRANSFER_FUNDS))
    result = gate(locked_state(), d)
    assert result.allowed == ("a",)
    assert result.blocked == ("b",)
    assert result.questions == ()
    assert isinstance(result.hand_off, HandOff)
    assert result.hand_off.intent_ids == ("b",)


def test_gate_with_no_intents_is_empty():
    result = gate(AuthState(), Decomposition(intents=()))
    assert (result.allowed, result.blocked, result.questions) == ((), (), ())
