"""A plain debit answer may not claim an action: only an approved plan closes, orders or issues anything."""

import pytest

from agent.specialists.debit import DEBIT_SPECIALIST, answer_check


@pytest.mark.parametrize(
    "text",
    [
        "Both of your debit cards have already been closed. New cards will be issued for both accounts.",
        "A new card will be ordered for your Blue Account.",
        "I've closed the card and ordered a replacement.",
        "We will issue a new card to you.",
        "Your replacement card has been shipped.",
    ],
)
def test_a_claim_of_an_action_with_no_plan_behind_it_is_refused(text):
    assert answer_check(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "That card is already closed, so I cannot close it again. Would you like me to order a replacement?",
        "Your account is past its replacement limit, so a $25 fee applies. Do you want to continue?",
        "Debit cards cannot be ordered for savings accounts.",
    ],
)
def test_an_honest_answer_passes(text):
    assert answer_check(text) is None


def test_the_debit_specialist_uses_the_check():
    assert DEBIT_SPECIALIST.answer_check is answer_check
