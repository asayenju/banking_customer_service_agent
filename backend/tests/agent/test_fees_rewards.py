"""Slice 4 of the fees specialist: the rewards correction workflow and validator.

The dispute is the customer's own step (their tool), so tests put an approved dispute row where that tool,
with the task's auto-resolve setting, would. The plan must be tau2's gold writes (one point lower on the one
documented departure), only for purchases whose dispute was approved, and never from any figure given."""

import json
from datetime import date

import pytest

from agent.specialist import ValidationContext
from agent.specialists import fees_rules as rules
from agent.workflows.base import HandOff, NeedInfo, Plan, Reply, WorkflowContext
from agent.workflows.fee_rewards import RewardsCorrection, RewardsFacts
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks

TASKS = {t["id"]: t for t in load_tasks()}
TODAY = date(2025, 11, 14)
TOOL = rules.REWARDS_TOOL
DEPARTURE = "txn_a8f1c2d3e411"


def user_of(task: str) -> str:
    return next(a["arguments"]["user_id"] for a in TASKS[task]["evaluation_criteria"]["actions"] if a["name"] == "log_verification")


def gold(task: str) -> dict[str, str]:
    return {
        json.loads(a["arguments"]["arguments"])["transaction_id"]: json.loads(a["arguments"]["arguments"])["new_rewards_earned"]
        for a in TASKS[task]["evaluation_criteria"]["actions"]
        if a["name"] == "call_discoverable_agent_tool" and a["arguments"]["agent_tool_name"] == TOOL
    }


def dispute(shared: BankingSession, task: str, txn: str, *, status: str = "RESOLVED", resolution: str = "APPROVED", user: str | None = None) -> None:
    row = {"dispute_id": f"d_{txn}", "user_id": user or user_of(task), "transaction_id": txn, "submitted_at": "11/14/2025", "status": status}
    if resolution:
        row["resolution"] = resolution
    shared.database.cash_back_disputes.data[f"d_{txn}"] = row


def world(task: str, *, approve: bool = True) -> BankingSession:
    shared = BankingSession.from_task(TASKS[task])
    if approve:
        for txn in gold(task):
            dispute(shared, task, txn)
    return shared


def v(shared: BankingSession, task: str) -> ValidationContext:
    return ValidationContext(shared.database, user_of(task), TODAY)


def decide(task: str, named: list[str], shared: BankingSession | None = None):
    shared = shared or world(task)
    return RewardsCorrection().decide(WorkflowContext(shared.database, user_of(task), TODAY), RewardsFacts(purchases=named))


@pytest.mark.parametrize("task", ["task_026", "task_028"])
def test_the_workflow_proposes_gold_writes_with_one_documented_departure(task: str) -> None:
    result = decide(task, [])
    assert isinstance(result, Plan)
    proposed = {s.args["transaction_id"]: s.args["new_rewards_earned"] for s in result.steps}
    expected = dict(gold(task))
    if DEPARTURE in expected:
        expected[DEPARTURE] = "1499 points"  # the documented floor; gold has 1500
    assert proposed == expected
    assert all(s.tool == TOOL for s in result.steps)
    assert all("txn_" not in s.note for s in result.steps), "a plan names purchases, not ids"


def test_the_plan_names_the_purchase_and_the_promotion() -> None:
    result = decide("task_026", [])
    assert isinstance(result, Plan)
    first = next(s for s in result.steps if s.args["transaction_id"] == "txn_a8f1c2d3e403")
    assert "$315.00 JetBlue Airways purchase on 03/22/2025 from 3150 points to 6300 points" in first.note
    assert "double-rewards promotion" in first.note


def test_no_dispute_means_no_write_and_a_way_forward() -> None:
    result = decide("task_026", [], world("task_026", approve=False))
    assert isinstance(result, Reply) and "submit one" in result.text


def test_a_dispute_still_under_review_is_not_acted_on() -> None:
    shared = world("task_026", approve=False)
    for txn in gold("task_026"):
        dispute(shared, "task_026", txn, status="SUBMITTED", resolution="")
    result = decide("task_026", [], shared)
    assert isinstance(result, Reply) and "still under review" in result.text


def test_a_rejected_dispute_is_not_acted_on() -> None:
    shared = world("task_026", approve=False)
    for txn in gold("task_026"):
        dispute(shared, "task_026", txn, resolution="REJECTED")
    assert isinstance(decide("task_026", [], shared), Reply)


def test_naming_a_purchase_changes_only_that_one_and_an_unclear_name_asks() -> None:
    result = decide("task_026", ["JetBlue"])
    assert isinstance(result, Plan) and [s.args["transaction_id"] for s in result.steps] == ["txn_a8f1c2d3e403"]
    asked = decide("task_026", ["Air"])  # JetBlue Airways and Southwest Airlines
    assert isinstance(asked, NeedInfo) and "Southwest" in asked.questions[0].text
    assert isinstance(decide("task_026", ["nothing like this"]), NeedInfo)


def test_a_purchase_that_is_already_right_is_not_written() -> None:
    shared = world("task_026")
    shared.database.credit_card_transaction_history.data["txn_a8f1c2d3e403"]["rewards_earned"] = "6300 points"
    result = decide("task_026", ["JetBlue"], shared)
    assert isinstance(result, Reply) and "Already correct" in result.text


def test_a_dispute_record_figure_is_never_used() -> None:
    shared = world("task_026")
    shared.database.cash_back_disputes.data["d_txn_a8f1c2d3e403"]["expected_rewards"] = "999999 points"
    result = decide("task_026", ["JetBlue"], shared)
    assert isinstance(result, Plan) and result.steps[0].args["new_rewards_earned"] == "6300 points"
    bad = {"transaction_id": "txn_a8f1c2d3e403", "new_rewards_earned": "999999 points"}
    assert "should have earned 6300 points" in (rules.rewards_update(v(shared, "task_026"), bad) or "")


def test_the_validator_accepts_the_gold_write_and_the_readback_confirms_it() -> None:
    shared = world("task_028")
    args = {"transaction_id": "txn_57ecc6da56c2", "new_rewards_earned": "95 points"}
    assert rules.rewards_update(v(shared, "task_028"), args) is None
    assert rules.rewards_readback(v(shared, "task_028"), args) == "the purchase's rewards were not changed"
    assert shared.call(TOOL, args).ok
    assert rules.rewards_readback(v(shared, "task_028"), args) is None
    assert "already correct" in (rules.rewards_update(v(shared, "task_028"), args) or "")


@pytest.mark.parametrize("value", ["95", "95 pts", "6,300 points", "95.5 points", "-95 points", " 95 points", "ninety points", ""])
def test_free_text_rewards_are_refused(value: str) -> None:
    shared = world("task_028")
    reason = rules.rewards_update(v(shared, "task_028"), {"transaction_id": "txn_57ecc6da56c2", "new_rewards_earned": value})
    assert reason is not None and ("whole number" in reason or "should have earned" in reason), reason


def test_another_customers_purchase_reads_like_one_that_does_not_exist() -> None:
    shared = world("task_028")
    other = BankingSession.from_task(TASKS["task_026"]).database.credit_card_transaction_history.data["txn_a8f1c2d3e403"]
    shared.database.credit_card_transaction_history.data["txn_a8f1c2d3e403"] = dict(other)
    dispute(shared, "task_028", "txn_a8f1c2d3e403")
    args = {"transaction_id": "txn_a8f1c2d3e403", "new_rewards_earned": "6300 points"}
    unknown = {"transaction_id": "txn_nope", "new_rewards_earned": "6300 points"}
    assert rules.rewards_update(v(shared, "task_028"), args) == rules.rewards_update(v(shared, "task_028"), unknown) == "that purchase does not exist"


def test_someone_elses_approved_dispute_does_not_unlock_my_purchase() -> None:
    shared = world("task_028", approve=False)
    dispute(shared, "task_028", "txn_57ecc6da56c2", user="someone-else")
    args = {"transaction_id": "txn_57ecc6da56c2", "new_rewards_earned": "95 points"}
    assert "reviewed and approved" in (rules.rewards_update(v(shared, "task_028"), args) or "")


def test_a_card_without_a_table_is_a_hand_off() -> None:
    shared = world("task_028")
    for row in shared.database.credit_card_transaction_history.data.values():
        if row.get("user_id") == user_of("task_028"):
            row["credit_card_type"] = "Diamond Elite Card"
    result = decide("task_028", [], shared)
    assert isinstance(result, HandOff) and "no rewards table" in result.reason


def test_a_pending_purchase_is_not_corrected() -> None:
    shared = world("task_028")
    shared.database.credit_card_transaction_history.data["txn_57ecc6da56c2"]["status"] = "PENDING"
    args = {"transaction_id": "txn_57ecc6da56c2", "new_rewards_earned": "95 points"}
    assert "completed purchase" in (rules.rewards_update(v(shared, "task_028"), args) or "")
