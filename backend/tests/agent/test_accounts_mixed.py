"""Slice 6 of the accounts specialist: requests that open and close accounts together. The opening goes first
(the planner's rule is tested in tests/orchestrator/test_plan.py); here, what happens when the order is wrong
anyway, and the reference answers that mix the two."""

from datetime import date

import pytest

from agent.specialist import ValidationContext
from agent.specialists import accounts_rules as rules
from agent.specialists.accounts import ACCOUNTS_SPECIALIST
from agent.tier1 import run_tier1
from banking_tools import BankingSession, Fidelity
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex

TODAY = date(2025, 11, 14)
INDEX = KnowledgeIndex({"doc_accounts_x": Document("doc_accounts_x", "Internal: x", "account procedure")})
TASKS = {t["id"]: t for t in load_tasks()}
USER = "jm60a8b9c2"  # task 060: Green checking 58d57780cc15e32d (120 days, $0) and Blue checking 38a2c5f77504be6c (10 days)
GREEN = "58d57780cc15e32d"


def v(shared) -> ValidationContext:
    return ValidationContext(shared.database, USER, TODAY)


def savings() -> dict:
    return {"user_id": USER, "account_type": "savings", "account_class": "Silver Plus Account"}


def world() -> BankingSession:
    return BankingSession.from_task(TASKS["task_060"], fidelity=Fidelity.TAU2)


def test_opening_first_works_and_closing_first_makes_the_opening_impossible():
    """The reason for the planner's rule: the Blue account is 10 days old, so once the Green one is gone no
    checking account is 14 days old."""
    first = world()
    assert rules.open_account(v(first), savings()) is None
    assert first.call("open_bank_account_4821", savings()).ok
    assert rules.close_account(v(first), {"account_id": GREEN}) is None

    wrong = world()
    assert wrong.call("close_bank_account_7392", {"account_id": GREEN, "reason": "x"}).ok
    reason = rules.open_account(v(wrong), savings())
    assert reason == "a savings account needs a checking account that has been open at least 14 days; your checking account has been open 10 days"


def test_a_business_account_cannot_be_opened_after_any_account_was_closed():
    shared = BankingSession.from_task(TASKS["task_062"], fidelity=Fidelity.TAU2)
    user = "jc61f7a8d2"
    args = {"user_id": user, "account_type": "business_checking", "account_class": "Navy Blue"}
    assert rules.open_account(ValidationContext(shared.database, user, TODAY), args) is None
    shared.database.accounts.data["61c9d8e7f6a5b432"]["current_holdings"] = "0"
    assert shared.call("close_bank_account_7392", {"account_id": "61c9d8e7f6a5b432", "reason": "x"}).ok
    assert "any of your accounts has been closed" in rules.open_account(ValidationContext(shared.database, user, TODAY), args)


@pytest.mark.parametrize("task_id", ["task_056", "task_062", "task_065", "task_066", "task_067", "task_068", "task_069"])
def test_the_reference_answer_is_reproduced_when_requests_mix_opening_and_closing(task_id):
    result = run_tier1(TASKS[task_id], ACCOUNTS_SPECIALIST, index=INDEX)
    assert result.error is None and result.excluded is None, result
    assert result.passed, result.mismatches
    assert result.bypasses == 0
    kinds = [s.split(":")[0] for s in result.segments]
    if "CLOSE_ACCOUNT" in kinds:
        assert kinds.index("OPEN_ACCOUNT") < kinds.index("CLOSE_ACCOUNT"), "the opening comes before the closing"


def test_the_money_out_of_the_old_accounts_goes_to_the_new_ones_in_the_four_part_overhaul():
    result = run_tier1(TASKS["task_062"], ACCOUNTS_SPECIALIST, index=INDEX)
    assert result.segments[1].startswith("CLOSE_ACCOUNT: transfer_funds_between_bank_accounts, close_bank_account, transfer")
    assert result.segments[1].count("transfer_funds") == 3 and result.segments[1].count("close_bank_account") == 2
