"""Slice 3 of the fees specialist: the expected APY and the credit reproduce tau2's gold for every savings
account of tasks 093 to 097 (nine accounts), including the one a balance x gap / 12 formula gets a cent wrong."""

import json

import pytest

from agent.bank_accounts import accounts_of
from agent.interest_review import review_interest
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks

TASKS = {t["id"]: t for t in load_tasks()}


def user_of(task: str) -> str:
    return next(a["arguments"]["user_id"] for a in TASKS[task]["evaluation_criteria"]["actions"] if a["name"] == "log_verification")


def gold(task: str) -> dict[str, tuple[float, float, float, float]]:
    """account id -> (credit, expected_apy, actual_apy, amount_difference) as tau2's gold calls give them."""
    credits, reports = {}, {}
    for a in TASKS[task]["evaluation_criteria"]["actions"]:
        if a["name"] != "call_discoverable_agent_tool":
            continue
        args = json.loads(a["arguments"]["arguments"])
        if a["arguments"]["agent_tool_name"].startswith("apply_savings"):
            credits[args["account_id"]] = float(args["amount"])
        elif a["arguments"]["agent_tool_name"].startswith("submit_interest"):
            reports[args["account_id"]] = (float(args["expected_apy"]), float(args["actual_apy"]), float(args["amount_difference"]))
    return {k: (credits[k], *reports[k]) for k in credits}


CASES = [(t, k, v) for t in ("task_093", "task_094", "task_095", "task_096", "task_097") for k, v in gold(t).items()]


@pytest.mark.parametrize("task, account_id, expected", CASES, ids=[c[1] for c in CASES])
def test_every_gold_credit_and_report_is_reproduced(task: str, account_id: str, expected: tuple[float, float, float, float]) -> None:
    db = BankingSession.from_task(TASKS[task]).database
    account = next(a for a in accounts_of(db, user_of(task)) if a.account_id == account_id)
    review = review_interest(db, account, user_of(task))
    assert review.cannot_decide is None, review.cannot_decide
    credit, expected_apy, actual_apy, difference = expected
    assert (review.credit, round(review.expected_apy, 3), review.actual_apy, review.credit) == (credit, expected_apy, actual_apy, difference)


def test_the_cent_a_gap_formula_gets_wrong_is_right_here() -> None:
    """097 silver: balance x gap / 12 = 220.8333 (220.83), expected interest less interest paid = 220.8367 (220.84)."""
    db = BankingSession.from_task(TASKS["task_097"]).database
    account = next(a for a in accounts_of(db, user_of("task_097")) if a.account_id.endswith("_silver"))
    review = review_interest(db, account, user_of("task_097"))
    assert round(review.balance * (review.expected_apy - review.actual_apy) / 1200, 2) == 220.83
    assert review.credit == 220.84


def test_the_checking_boost_and_the_card_bonus_are_each_the_highest_never_the_sum() -> None:
    db = BankingSession.from_task(TASKS["task_095"]).database  # Green (+0.75) and Purple (+0.1); EcoCard +0.6 beats Gold +0.025
    account = next(a for a in accounts_of(db, user_of("task_095")) if a.product.value == "savings")
    review = review_interest(db, account, user_of("task_095"))
    assert (review.base, review.checking_boost, review.card_bonus, review.relationship_bonus) == (5.5, 0.75, 0.6, 0.0)


def test_the_relationship_bonus_goes_only_to_a_card_less_silver_holder_with_a_checking_account() -> None:
    db = BankingSession.from_task(TASKS["task_093"]).database
    account = next(a for a in accounts_of(db, user_of("task_093")) if a.product.value == "savings")
    assert review_interest(db, account, user_of("task_093")).relationship_bonus == 0.025
    # A card whose bonus applies takes its place.
    db.credit_card_accounts.data["cc_x"] = {
        "account_id": "cc_x", "user_id": user_of("task_093"), "card_type": "Silver Rewards Card", "status": "ACTIVE",
        "date_of_account_open": "01/01/2024",
    }
    assert review_interest(db, account, user_of("task_093")).relationship_bonus == 0.0
    db.credit_card_accounts.data["cc_x"]["card_type"] = "Bronze Rewards Card"  # a +0 card is no card bonus
    assert review_interest(db, account, user_of("task_093")).relationship_bonus == 0.025


def test_no_checking_account_means_no_boost_and_no_relationship_bonus() -> None:
    db = BankingSession.from_task(TASKS["task_093"]).database
    for row in db.accounts.data.values():
        if row["class"] == "checking":
            row["status"] = "CLOSED"
    account = next(a for a in accounts_of(db, user_of("task_093")) if a.product.value == "savings")
    review = review_interest(db, account, user_of("task_093"))
    assert (review.checking_boost, review.relationship_bonus, review.expected_apy) == (0.0, 0.0, 4.0) and review.credit == 0.0


def test_a_balance_that_changed_after_the_interest_was_paid_is_not_decided() -> None:
    db = BankingSession.from_task(TASKS["task_094"]).database
    account = next(a for a in accounts_of(db, user_of("task_094")) if a.product.value == "savings")
    db.bank_account_transaction_history.data["late"] = {
        "transaction_id": "late", "account_id": account.account_id, "date": "11/05/2025", "description": "ACH TRANSFER IN",
        "amount": 1000.0, "type": "ach_transfer_in", "status": "posted",
    }
    assert "balance changed" in (review_interest(db, account, user_of("task_094")).cannot_decide or "")


def test_a_level_without_a_table_and_a_missing_interest_payment_are_not_guessed() -> None:
    db = BankingSession.from_task(TASKS["task_094"]).database
    account = next(a for a in accounts_of(db, user_of("task_094")) if a.product.value == "savings")
    db.accounts.data[account.account_id]["level"] = "Emerald Saver Account"
    changed = next(a for a in accounts_of(db, user_of("task_094")) if a.product.value == "savings")
    assert "no interest table" in (review_interest(db, changed, user_of("task_094")).cannot_decide or "")
    db2 = BankingSession.from_task(TASKS["task_094"]).database
    for row in list(db2.bank_account_transaction_history.data.values()):
        if row["type"] == "interest_credit":
            row["type"] = "other"
    account2 = next(a for a in accounts_of(db2, user_of("task_094")) if a.product.value == "savings")
    assert "no monthly interest payment" in (review_interest(db2, account2, user_of("task_094")).cannot_decide or "")


def test_a_correction_already_posted_is_flagged() -> None:
    db = BankingSession.from_task(TASKS["task_094"]).database
    account = next(a for a in accounts_of(db, user_of("task_094")) if a.product.value == "savings")
    db.bank_account_transaction_history.data["fix"] = {
        "transaction_id": "fix", "account_id": account.account_id, "date": "11/10/2025",
        "description": "INTEREST CORRECTION - CUSTOMER SERVICE", "amount": 140.0, "type": "interest_correction", "status": "posted",
    }
    assert review_interest(db, account, user_of("task_094")).corrected is True
