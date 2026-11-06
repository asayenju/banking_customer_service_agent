"""Slice 4 of the fees specialist: expected points for every transaction of tasks 026 and 028 match tau2's gold,
except one the documented floor rule gets a point under (026 e411, a recorded departure)."""

import json

import pytest

from agent import rewards_review as R
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks

TASKS = {t["id"]: t for t in load_tasks()}


def user_of(task: str) -> str:
    return next(a["arguments"]["user_id"] for a in TASKS[task]["evaluation_criteria"]["actions"] if a["name"] == "log_verification")


def gold(task: str) -> dict[str, int]:
    out = {}
    for a in TASKS[task]["evaluation_criteria"]["actions"]:
        if a["name"] == "call_discoverable_agent_tool" and a["arguments"]["agent_tool_name"].startswith("update_transaction_rewards"):
            args = json.loads(a["arguments"]["arguments"])
            out[args["transaction_id"]] = int(args["new_rewards_earned"].split()[0])
    return out


CASES = [(t, k, v) for t in ("task_026", "task_028") for k, v in gold(t).items()]
DEPARTURE = "txn_a8f1c2d3e411"


@pytest.mark.parametrize("task, txn, points", CASES, ids=[c[1] for c in CASES])
def test_expected_points_match_gold_except_the_one_departure(task: str, txn: str, points: int) -> None:
    db = BankingSession.from_task(TASKS[task]).database
    review = R.review_transaction(db, db.credit_card_transaction_history.data[txn], user_of(task))
    assert review.cannot_decide is None, review.cannot_decide
    if txn == DEPARTURE:
        assert (review.expected, points) == (1499, 1500)  # 149.99 x 10 = 1499.9; the documented rule floors it
    else:
        assert review.expected == points
    assert review.wrong  # every listed transaction was recorded wrongly


def test_the_promotion_depends_on_the_purchase_date_and_the_opening_date() -> None:
    db = BankingSession.from_task(TASKS["task_026"]).database
    user = user_of("task_026")
    early = R.review_transaction(db, db.credit_card_transaction_history.data["txn_a8f1c2d3e403"], user)  # 03/22, opened 02/13
    late = R.review_transaction(db, db.credit_card_transaction_history.data["txn_a8f1c2d3e410"], user)  # 08/25
    assert (early.multiplier, late.multiplier) == (2, 1)
    # Opened outside the promotion window: no doubling at all.
    for row in db.credit_card_accounts.data.values():
        if row["card_type"] == "Business Silver Rewards Card":
            row["date_of_account_open"] = "01/01/2024"
    still = R.review_transaction(db, db.credit_card_transaction_history.data["txn_a8f1c2d3e403"], user)
    assert still.multiplier == 1 and still.expected == 3150


def test_an_excluded_merchant_earns_the_default_rate() -> None:
    db = BankingSession.from_task(TASKS["task_028"]).database
    row = dict(db.credit_card_transaction_history.data["txn_896ac64b98d7"])  # EcoCard, Patagonia, green
    assert R.review_transaction(db, row, user_of("task_028")).expected == 642
    assert R.review_transaction(db, {**row, "merchant_name": "Amazon"}, user_of("task_028")).expected == 128
    assert R.review_transaction(db, {**row, "merchant_name": "Amazon Fresh Market"}, user_of("task_028")).expected == 128
    assert R.review_transaction(db, {**row, "merchant_name": "Amazonia"}, user_of("task_028")).expected == 642


def test_the_documents_own_rounding_examples() -> None:
    db = BankingSession.from_task(TASKS["task_028"]).database
    row = dict(db.credit_card_transaction_history.data["txn_57ecc6da56c2"])  # Crypto-Cash Back, 2%
    assert R.review_transaction(db, {**row, "transaction_amount": "$126.77"}, user_of("task_028")).expected == 253
    assert R.review_transaction(db, {**row, "transaction_amount": "$99.99"}, user_of("task_028")).expected == 199


def test_a_card_without_a_table_and_a_bad_amount_are_not_guessed() -> None:
    db = BankingSession.from_task(TASKS["task_028"]).database
    row = dict(db.credit_card_transaction_history.data["txn_57ecc6da56c2"])
    assert "no rewards table" in (R.review_transaction(db, {**row, "credit_card_type": "Diamond Elite Card"}, "u").cannot_decide or "")
    assert R.review_transaction(db, {**row, "transaction_amount": "free"}, "u").cannot_decide
    assert R.review_transaction(db, {**row, "transaction_date": "someday"}, "u").cannot_decide


def test_stored_values_are_read_only_in_the_one_format() -> None:
    assert R.stored_points({"rewards_earned": "3150 points"}) == 3150
    assert R.stored_points({"rewards_earned": "6,300 pts"}) is None
    assert R.stored_points({"rewards_earned": "6300"}) is None
    assert R.points_text(95) == "95 points"
