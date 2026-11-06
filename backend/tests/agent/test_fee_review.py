"""Slice 2 of the fees specialist: the ATM fee review reproduces tau2's gold credit for every account of
tasks 072, 073 and 074, and a mis-timed fee that totals the right amount is no credit."""

from datetime import date

import pytest

from agent import fee_review as F
from agent.bank_accounts import accounts_of, get
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks

TASKS = {t["id"]: t for t in load_tasks()}
TODAY = date(2025, 11, 14)


def gold_credits(task: str) -> dict[str, float]:
    out = {}
    for a in TASKS[task]["evaluation_criteria"]["actions"]:
        if a["name"] == "call_discoverable_agent_tool" and a["arguments"]["agent_tool_name"].startswith("apply_checking"):
            import json

            args = json.loads(a["arguments"]["arguments"])
            out[args["account_id"]] = float(args["amount"])
    return out


def user_of(task: str) -> str:
    return next(a["arguments"]["user_id"] for a in TASKS[task]["evaluation_criteria"]["actions"] if a["name"] == "log_verification")


@pytest.mark.parametrize("task", ["task_072", "task_073", "task_074"])
def test_the_review_reproduces_every_gold_credit(task: str) -> None:
    db = BankingSession.from_task(TASKS[task]).database
    gold = gold_credits(task)
    reviewed = {}
    for account in accounts_of(db, user_of(task)):
        if account.is_checking:
            review = F.review_atm_fees(db, account)
            assert review.cannot_decide is None, (account.name, review.cannot_decide)
            reviewed[account.account_id] = review.credit
    assert reviewed == gold


def test_a_mistimed_fee_with_the_right_total_is_no_credit() -> None:
    """Task 072's Light Green account: the domestic fees total what the schedule says but sit on the wrong
    withdrawals. Only the foreign over-charge (3.50) is owed."""
    db = BankingSession.from_task(TASKS["task_072"]).database
    account = get(db, user_of("task_072"), "chk_538bfb9cba")
    assert account is not None
    review = F.review_atm_fees(db, account)
    assert review.credit == 3.5 and review.under_charged == 0.0


def test_a_missed_fee_is_set_against_an_over_charge_never_billed() -> None:
    """Task 073's Light Green account: a fee on the 2nd withdrawal and none on the 7th net to zero within the
    month, and the foreign 150 charged 5.00 against 3.50 is the whole 1.50 owed."""
    db = BankingSession.from_task(TASKS["task_073"]).database
    account = get(db, user_of("task_073"), "chk_kj93a7b2e1_3")
    assert account is not None
    review = F.review_atm_fees(db, account)
    assert (review.over_charged, review.under_charged, review.credit) == (1.5, 0.0, 1.5)


def test_an_account_charged_less_than_the_schedule_is_owed_nothing() -> None:
    db = BankingSession.from_task(TASKS["task_072"]).database
    account = get(db, user_of("task_072"), "chk_538bfb9cba")
    assert account is not None
    for row in db.bank_account_transaction_history.data.values():
        if row["account_id"] == account.account_id and row["type"] == "atm_fee":
            row["amount"] = 0.0  # every fee waived: the bank under-charged
    review = F.review_atm_fees(db, account)
    assert review.credit == 0.0 and review.under_charged > 0


def test_a_class_with_no_schedule_is_a_hand_off_not_a_guess() -> None:
    db = BankingSession.from_task(TASKS["task_072"]).database
    account = get(db, user_of("task_072"), "chk_538bfb9cba")
    assert account is not None
    db.accounts.data[account.account_id]["level"] = "Gold Years Account"
    review = F.review_atm_fees(db, get(db, user_of("task_072"), account.account_id))  # type: ignore[arg-type]
    assert review.credit == 0.0 and "no fee schedule" in (review.cannot_decide or "")


def test_a_withdrawal_whose_place_is_unknown_is_a_hand_off() -> None:
    db = BankingSession.from_task(TASKS["task_072"]).database
    account = get(db, user_of("task_072"), "chk_538bfb9cba")
    assert account is not None
    for row in db.bank_account_transaction_history.data.values():
        if row["account_id"] == account.account_id and row["type"] == "atm_withdrawal":
            row["description"] = "ATM WITHDRAWAL - SOME BANK DOVER DE"
            break
    assert F.review_atm_fees(db, account).cannot_decide


def test_a_refund_in_the_last_14_days_is_found_and_an_old_one_is_not() -> None:
    db = BankingSession.from_task(TASKS["task_072"]).database
    account = get(db, user_of("task_072"), "chk_538bfb9cba")
    assert account is not None
    assert F.recent_credit(db, account, TODAY) is None
    db.bank_account_transaction_history.data["t1"] = {
        "transaction_id": "t1", "account_id": account.account_id, "date": "11/10/2025",
        "description": "FEE REFUND - CUSTOMER SERVICE", "amount": 5.0, "type": "fee_refund", "status": "posted",
    }
    assert F.recent_credit(db, account, TODAY) == date(2025, 11, 10)
    assert F.recent_credit(db, account, date(2025, 12, 10)) is None
