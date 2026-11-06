"""Slice 2 of the fees specialist: the checking credit validator, its read-back and the workflow.

Real task data (072 to 074). The workflow must propose exactly tau2's gold credits; the validator must refuse
what the bank's tool would happily do (another customer's account, a figure the records do not support, a
second credit within 14 days, a repeat).
"""

import json
from datetime import date

import pytest

from agent import bank_accounts as accounts
from agent.fee_review import review_atm_fees
from agent.specialist import ValidationContext
from agent.specialists import fees_rules as rules
from agent.specialists.fees import FEES_SPECIALIST
from agent.workflows.base import HandOff, NeedInfo, Plan, Reply, WorkflowContext
from agent.workflows.fee_checking import CheckingFeeCredit, CheckingFeeFacts
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks

TASKS = {t["id"]: t for t in load_tasks()}
TODAY = date(2025, 11, 14)
TOOL = rules.CHECKING_CREDIT_TOOL


def user_of(task: str) -> str:
    return next(a["arguments"]["user_id"] for a in TASKS[task]["evaluation_criteria"]["actions"] if a["name"] == "log_verification")


def gold(task: str) -> list[dict]:
    return [
        json.loads(a["arguments"]["arguments"])
        for a in TASKS[task]["evaluation_criteria"]["actions"]
        if a["name"] == "call_discoverable_agent_tool" and a["arguments"]["agent_tool_name"] == TOOL
    ]


def world(task: str) -> BankingSession:
    return BankingSession.from_task(TASKS[task])


def v(shared: BankingSession, task: str) -> ValidationContext:
    return ValidationContext(shared.database, user_of(task), TODAY)


def decide(task: str, named: list[str], shared: BankingSession | None = None):
    shared = shared or world(task)
    return CheckingFeeCredit().decide(WorkflowContext(shared.database, user_of(task), TODAY), CheckingFeeFacts(accounts=named))


@pytest.mark.parametrize("task", ["task_072", "task_073", "task_074"])
def test_the_workflow_proposes_exactly_tau2s_gold_credits(task: str) -> None:
    result = decide(task, [])
    assert isinstance(result, Plan)
    proposed = sorted((s.args["account_id"], s.args["amount"], s.args["credit_type"]) for s in result.steps)
    assert proposed == sorted((g["account_id"], float(g["amount"]), g["credit_type"]) for g in gold(task))
    assert all(s.tool == TOOL for s in result.steps)
    assert all("chk_" not in s.note and "txn_" not in s.note for s in result.steps), "a plan names accounts, not ids"


def test_naming_one_account_reviews_only_that_account() -> None:
    result = decide("task_072", ["Bluest"])
    assert isinstance(result, Plan) and [s.args["account_id"] for s in result.steps] == ["chk_lj82d4f1a9"]


def test_an_unclear_name_is_asked_with_a_list_of_the_customers_own_accounts() -> None:
    result = decide("task_072", ["Platinum"])
    assert isinstance(result, NeedInfo) and "Bluest" in result.questions[0].text and "Light Green" in result.questions[0].text


def test_a_customers_own_figure_never_enters_the_plan() -> None:
    """The facts carry no amount at all: the credit is worked out from the records."""
    assert "amount" not in CheckingFeeFacts.model_fields


def test_nothing_owed_is_said_and_nothing_is_proposed() -> None:
    """Every fee on the account waived: the bank under-charged, so nothing is owed and nothing is billed back."""
    shared = world("task_072")
    for row in shared.database.bank_account_transaction_history.data.values():
        if row["account_id"] == "chk_538bfb9cba" and row["type"] == "atm_fee":
            row["amount"] = 0.0
    result = decide("task_072", ["Light Green"], shared)
    assert isinstance(result, Reply) and "nothing to credit" in result.text


def test_a_credit_in_the_last_14_days_leaves_that_account_out_and_says_why() -> None:
    shared = world("task_072")
    shared.database.bank_account_transaction_history.data["t1"] = {
        "transaction_id": "t1", "account_id": "chk_538bfb9cba", "date": "11/10/2025",
        "description": "FEE REFUND - CUSTOMER SERVICE", "amount": 3.5, "type": "fee_refund", "status": "posted",
    }
    result = decide("task_072", [], shared)
    assert isinstance(result, Plan) and [s.args["account_id"] for s in result.steps] == ["chk_lj82d4f1a9"]
    assert "already applied on 11/10/2025" in result.steps[0].note and "Light Green" in result.steps[0].note
    only = decide("task_072", ["Light Green"], shared)
    assert isinstance(only, HandOff) and "already applied" in only.reason


def test_a_class_without_a_schedule_is_a_hand_off() -> None:
    shared = world("task_072")
    shared.database.accounts.data["chk_538bfb9cba"]["level"] = "Gold Years Account"
    result = decide("task_072", ["Gold Years"], shared)
    assert isinstance(result, HandOff) and "no fee schedule" in result.reason


def good(task: str = "task_072") -> dict:
    return gold(task)[0]


def test_the_validator_accepts_the_gold_credit_and_the_bank_then_posts_it() -> None:
    shared = world("task_072")
    args = good()
    assert rules.checking_credit(v(shared, "task_072"), args) is None
    assert shared.call(TOOL, args).ok
    assert rules.checking_credit_readback(v(shared, "task_072"), args) is None


def test_a_repeat_after_a_timeout_is_refused_before_it_adds_the_balance_twice() -> None:
    shared = world("task_072")
    args = good()
    assert shared.call(TOOL, args).ok
    reason = rules.checking_credit(v(shared, "task_072"), args)
    assert reason is not None and "already applied to this account on 11/14/2025" in reason  # the 14-day rule stops it first


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({"account_id": "chk_does_not_exist"}, "does not exist"),
        ({"amount": 99.0}, "come to"),
        ({"amount": 0.0}, "above zero"),
        ({"amount": -5.0}, "above zero"),
        ({"amount": float("nan")}, "above zero"),
        ({"amount": float("inf")}, "above zero"),
        ({"amount": 14.005}, "dollars and cents"),
        ({"credit_type": "goodwill_credit"}, "not a type"),
        ({"credit_type": "rebate_credit"}, "fee refund"),
    ],
)
def test_the_validator_refuses(change: dict, fragment: str) -> None:
    shared = world("task_072")
    reason = rules.checking_credit(v(shared, "task_072"), {**good(), **change})
    assert reason is not None and fragment in reason, reason


def test_another_customers_account_reads_like_one_that_does_not_exist() -> None:
    shared = world("task_072")
    other = world("task_073")
    mine = rules.checking_credit(v(shared, "task_072"), {**good(), "account_id": "chk_kj93a7b2e1_1"})
    unknown = rules.checking_credit(v(shared, "task_072"), {**good(), "account_id": "chk_nobody"})
    assert mine == unknown == rules.NOT_FOUND
    assert other is not None


def test_a_savings_or_closed_account_cannot_take_an_atm_credit() -> None:
    shared = world("task_072")
    shared.database.accounts.data["chk_538bfb9cba"]["status"] = "CLOSED"
    assert "not open" in (rules.checking_credit(v(shared, "task_072"), gold("task_072")[1]) or "")
    shared.database.accounts.data["chk_538bfb9cba"]["status"] = "OPEN"
    shared.database.accounts.data["chk_538bfb9cba"]["class"] = "saving"
    assert "personal checking" in (rules.checking_credit(v(shared, "task_072"), gold("task_072")[1]) or "")


def test_the_validator_looks_at_the_records_not_the_plan_when_the_fees_change_between_preview_and_yes() -> None:
    shared = world("task_072")
    args = good()
    assert rules.checking_credit(v(shared, "task_072"), args) is None
    for row in shared.database.bank_account_transaction_history.data.values():  # the bank corrected a fee meanwhile
        if row["account_id"] == args["account_id"] and "FOREIGN ATM FEE" in row["description"]:
            row["amount"] = 0.0
    assert "come to" in (rules.checking_credit(v(shared, "task_072"), args) or "")


def test_the_readback_catches_a_credit_that_is_not_there() -> None:
    shared = world("task_072")
    assert rules.checking_credit_readback(v(shared, "task_072"), good()) == "the credit is not on the account's transactions"


def test_the_specialist_is_wired_for_its_kind_and_tool() -> None:
    assert FEES_SPECIALIST.name == "fees"
    assert TOOL in FEES_SPECIALIST.validators and TOOL in FEES_SPECIALIST.readbacks
    assert [w.name for w in FEES_SPECIALIST.workflows] == ["checking_fee_credit"]


def test_the_review_and_the_workflow_agree_on_every_account() -> None:
    for task in ("task_072", "task_073", "task_074"):
        shared = world(task)
        result = decide(task, [], shared)
        assert isinstance(result, Plan)
        by_account = {s.args["account_id"]: s.args["amount"] for s in result.steps}
        for account in accounts.accounts_of(shared.database, user_of(task)):
            if account.is_checking:
                assert by_account[account.account_id] == review_atm_fees(shared.database, account).credit
