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


# ---------------------------------------------------------------- the temporary limit eligibility (doc 040, task 089)

def _limit_world():
    from datetime import date

    from agent.specialist import ValidationContext
    from banking_tools import BankingSession
    from banking_tools.manifest import load_tasks

    db = BankingSession.from_task({t["id"]: t for t in load_tasks()}["task_089"]).database
    return db, ValidationContext(db, "x", date(2025, 11, 14))


def test_an_overdraft_fee_within_30_days_blocks_a_temporary_limit_increase():
    from agent.specialists.debit import temporary_limit

    _, v = _limit_world()
    args = {"limit_type": "atm", "new_limit": 900}
    assert temporary_limit(v, {**args, "card_id": "dbc_5e1b9d4f2a8c"}) is not None, "the Blue card's overdraft 13 days ago"
    assert temporary_limit(v, {**args, "card_id": "dbc_2f8a7c3d1e9b"}) is None, "the Green card is the one the reference raises"


def test_a_young_account_is_not_eligible_for_a_temporary_limit_increase():
    from agent.specialists.debit import temporary_limit

    db, v = _limit_world()
    db.accounts.data[db.debit_cards.data["dbc_2f8a7c3d1e9b"]["account_id"]]["date_opened"] = "10/20/2025"
    assert "60 days" in str(temporary_limit(v, {"card_id": "dbc_2f8a7c3d1e9b", "limit_type": "atm", "new_limit": 900}))
