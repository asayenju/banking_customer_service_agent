"""Slice 0 of the fees specialist: what the bank's fee, interest and rewards tools really do, shown by running them.

``docs/fees-policy-decisions.md`` says the tools check far less than the policy documents. Each statement there
that the code relies on is a test here, so the note cannot drift from the tools: if tau2 ever fixes one of these,
a test fails and the note (and the validator that covers it) is revisited.
"""

from pathlib import Path

import pytest

from banking_tools import BankingSession
from banking_tools.manifest import load_tasks

DOC = Path(__file__).resolve().parents[3] / "docs" / "fees-policy-decisions.md"
TASKS = {t["id"]: t for t in load_tasks()}
CHECKING = "apply_checking_account_credit_5829"
SAVINGS = "apply_savings_account_credit_6831"
REPORT = "submit_interest_discrepancy_report_7294"
REWARDS = "update_transaction_rewards_3847"


def world(task: str) -> BankingSession:
    return BankingSession.from_task(TASKS[task])


def test_the_policy_note_has_every_section_the_slices_depend_on():
    text = DOC.read_text()
    for heading in (
        "## 1. The bank's tools",
        "## 2. Savings interest correction",
        "## 3. Checking fee credit",
        "## 4. Rewards",
        "## 5. Document contradictions and rulings",
        "## 6. Status",
    ):
        assert heading in text, heading


# ---------------------------------------------------------------- the discrepancy report

def test_the_report_stores_no_report_only_a_called_marker():
    """No table holds the report, yet the tool says 'Submitted'. The one trace is the tool's own CALLED marker."""
    import json

    shared = world("task_094")
    before = json.loads(shared.database.model_dump_json())
    args = {"account_id": "sav_wl94k7m3p8_gold", "user_id": "wl94k7m3p8", "expected_apy": 6.85, "actual_apy": 5.1, "amount_difference": 140.0}
    result = shared.call(REPORT, args)
    assert result.ok and "Submitted Successfully" in result.text and "PENDING_REVIEW" in result.text
    assert not hasattr(shared.database, "interest_discrepancy_reports"), "there is no table for the report"
    after = json.loads(shared.database.model_dump_json())
    changed = {k for k in after if after[k] != before.get(k)}
    assert changed == {"agent_discoverable_tools"}, "nothing about the report's content is kept"
    assert [r["status"] for r in after["agent_discoverable_tools"]["data"].values()] == ["CALLED"]


def test_the_report_does_not_check_that_the_user_owns_the_account():
    shared = world("task_094")
    other = world("task_093").database.accounts.data["sav_sp93k4m7n2_silver"]
    shared.database.accounts.data["sav_sp93k4m7n2_silver"] = dict(other)
    args = {"account_id": "sav_sp93k4m7n2_silver", "user_id": "wl94k7m3p8", "expected_apy": 1.0, "actual_apy": 0.5, "amount_difference": 1.0}
    assert shared.call(REPORT, args).ok


# ---------------------------------------------------------------- the credits

def test_the_two_credit_tools_write_the_balance_in_different_formats():
    checking, savings = world("task_072"), world("task_094")
    assert checking.call(CHECKING, {"account_id": "chk_lj82d4f1a9", "amount": 5, "credit_type": "fee_refund"}).ok
    assert savings.call(SAVINGS, {"account_id": "sav_wl94k7m3p8_gold", "amount": 5, "credit_type": "interest_correction"}).ok
    assert checking.database.accounts.data["chk_lj82d4f1a9"]["current_holdings"] == "$127455.00"
    assert savings.database.accounts.data["sav_wl94k7m3p8_gold"]["current_holdings"] == "96005.00"


@pytest.mark.parametrize(
    ("task", "tool", "account", "credit_type"),
    [("task_072", CHECKING, "chk_lj82d4f1a9", "fee_refund"), ("task_094", SAVINGS, "sav_wl94k7m3p8_gold", "interest_correction")],
)
def test_the_credit_tools_check_no_ownership_no_cap_and_no_cooldown(task, tool, account, credit_type):
    shared = world(task)
    row = shared.database.accounts.data[account]
    row["user_id"] = "somebody-else"  # not the verified customer's account
    assert shared.call(tool, {"account_id": account, "amount": 99999, "credit_type": credit_type}).ok, "no ownership check, no cap"
    assert shared.call(tool, {"account_id": account, "amount": 1, "credit_type": credit_type}).ok, "no cooldown"


def test_a_repeated_credit_rewrites_the_same_transaction_row_but_adds_the_balance_again():
    shared = world("task_072")
    args = {"account_id": "chk_lj82d4f1a9", "amount": 10, "credit_type": "fee_refund"}
    rows_before = len(shared.database.bank_account_transaction_history.data)
    assert shared.call(CHECKING, args).ok and shared.call(CHECKING, args).ok
    assert len(shared.database.bank_account_transaction_history.data) == rows_before + 1, "one row, because its id is a hash of the arguments"
    assert shared.database.accounts.data["chk_lj82d4f1a9"]["current_holdings"] == "$127470.00", "but the balance went up twice"


def test_the_savings_credit_accepts_the_banks_own_spelling_of_the_class():
    shared = world("task_094")
    assert shared.database.accounts.data["sav_wl94k7m3p8_gold"]["class"] == "saving"
    assert shared.call(SAVINGS, {"account_id": "sav_wl94k7m3p8_gold", "amount": 1, "credit_type": "interest_correction"}).ok


# ---------------------------------------------------------------- rewards

def test_the_rewards_tool_accepts_any_text_for_any_transaction():
    shared = world("task_028")
    other = world("task_026").database.credit_card_transaction_history.data["txn_a8f1c2d3e403"]
    shared.database.credit_card_transaction_history.data["txn_a8f1c2d3e403"] = dict(other)  # another customer's purchase
    assert shared.call(REWARDS, {"transaction_id": "txn_57ecc6da56c2", "new_rewards_earned": "banana"}).ok
    assert shared.call(REWARDS, {"transaction_id": "txn_a8f1c2d3e403", "new_rewards_earned": "99999999 points"}).ok
    assert shared.database.credit_card_transaction_history.data["txn_57ecc6da56c2"]["rewards_earned"] == "banana"


def test_a_resolved_dispute_exists_only_when_the_task_sets_auto_resolve():
    """The customer's own tool writes status RESOLVED / APPROVED only under the task's dispute_settings."""
    for task in ("task_026", "task_028"):
        settings = TASKS[task]["initial_state"]["initialization_data"]["agent_data"]["task_config"]["data"]["dispute_settings"]
        assert settings == {"auto_resolve_disputes": True}
