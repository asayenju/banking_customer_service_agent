"""Slice 3 of the fees specialist: the savings interest correction workflow, its validators and the report.

The workflow must propose tau2's gold writes, credit first and then report, for tasks 093 to 097. The report
is never filed alone, never claims a credit, and the customer's own figures never enter."""

import json
from datetime import date

import pytest

from agent import bank_accounts as accounts
from agent.specialist import ValidationContext
from agent.specialists import fees_rules as rules
from agent.workflows.base import HandOff, NeedInfo, Plan, Reply, WorkflowContext
from agent.workflows.fee_interest import InterestFacts, SavingsInterestCorrection
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks

TASKS = {t["id"]: t for t in load_tasks()}
TODAY = date(2025, 11, 14)
CREDIT, REPORT = rules.SAVINGS_CREDIT_TOOL, rules.REPORT_TOOL
ALL = ["task_093", "task_094", "task_095", "task_096", "task_097"]


def user_of(task: str) -> str:
    return next(a["arguments"]["user_id"] for a in TASKS[task]["evaluation_criteria"]["actions"] if a["name"] == "log_verification")


def gold_writes(task: str) -> list[tuple[str, dict]]:
    out = []
    for a in TASKS[task]["evaluation_criteria"]["actions"]:
        if a["name"] == "call_discoverable_agent_tool" and a["arguments"]["agent_tool_name"] in (CREDIT, REPORT):
            out.append((a["arguments"]["agent_tool_name"], json.loads(a["arguments"]["arguments"])))
    return out


def world(task: str) -> BankingSession:
    return BankingSession.from_task(TASKS[task])


def v(shared: BankingSession, task: str) -> ValidationContext:
    return ValidationContext(shared.database, user_of(task), TODAY)


def decide(task: str, named: list[str], shared: BankingSession | None = None):
    shared = shared or world(task)
    return SavingsInterestCorrection().decide(WorkflowContext(shared.database, user_of(task), TODAY), InterestFacts(accounts=named))


@pytest.mark.parametrize("task", ALL)
def test_the_workflow_proposes_gold_credits_and_reports_each_credit_before_its_report(task: str) -> None:
    result = decide(task, [])
    assert isinstance(result, Plan)
    gold = gold_writes(task)
    assert len(result.steps) == len(gold)
    for step in result.steps:
        assert step.tool in (CREDIT, REPORT)
    # Same accounts, tools, amounts and rates as gold, in any account order but credit before report for each.
    proposed = sorted((s.tool, s.args["account_id"], float(s.args["amount"] if s.tool == CREDIT else s.args["amount_difference"]),
                       s.args.get("expected_apy", 0.0), s.args.get("actual_apy", 0.0)) for s in result.steps)
    expected = sorted((t, a["account_id"], float(a["amount"] if t == CREDIT else a["amount_difference"]),
                       float(a.get("expected_apy", 0.0)), float(a.get("actual_apy", 0.0))) for t, a in gold)
    assert proposed == expected
    for n, step in enumerate(result.steps):
        if step.tool == REPORT:
            earlier = [s for s in result.steps[:n] if s.tool == CREDIT and s.args["account_id"] == step.args["account_id"]]
            assert earlier, "a report always follows its own credit"
    assert all("sav_" not in s.note and "chk_" not in s.note for s in result.steps), "a plan names accounts, not ids"


@pytest.mark.parametrize("task", ALL)
def test_the_validators_accept_the_whole_plan_in_order_on_a_database_that_has_run_it(task: str) -> None:
    shared = world(task)
    result = decide(task, [], shared)
    assert isinstance(result, Plan)
    for step in result.steps:
        assert rules.savings_credit(v(shared, task), step.args) is None if step.tool == CREDIT else rules.interest_report(v(shared, task), step.args) is None, step
        assert shared.call(step.tool, step.args).ok
        if step.tool == CREDIT:
            assert rules.savings_credit_readback(v(shared, task), step.args) is None


def test_a_report_is_never_filed_before_its_credit() -> None:
    shared = world("task_094")
    plan = decide("task_094", [], shared)
    assert isinstance(plan, Plan)
    report = plan.steps[1].args
    assert "credit has to be applied before the report" in (rules.interest_report(v(shared, "task_094"), report) or "")


def test_the_reply_after_the_plan_says_report_not_credit() -> None:
    plan = decide("task_094", [])
    assert isinstance(plan, Plan)
    text = SavingsInterestCorrection().after(WorkflowContext(world("task_094").database, user_of("task_094"), TODAY), plan.steps)
    assert "not a credit" in text and "report" in text


def test_the_plan_shows_the_working_and_calls_the_report_a_report() -> None:
    plan = decide("task_094", [])
    assert isinstance(plan, Plan)
    assert "$96,000.00 at 6.85% for a month is $548.00; $408.00 was paid" in plan.steps[0].note
    assert "not a credit" in plan.steps[1].note


def test_a_customers_figure_never_enters_the_form() -> None:
    assert set(InterestFacts.model_fields) == {"accounts"}


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({"account_id": "sav_nobody"}, "does not exist"),
        ({"amount": 72.0}, "comes to"),
        ({"amount": 0.0}, "above zero"),
        ({"amount": float("nan")}, "above zero"),
        ({"amount": 140.005}, "dollars and cents"),
        ({"credit_type": "goodwill_credit"}, "needs a person"),
        ({"credit_type": "fee_refund"}, "needs a person"),
    ],
)
def test_the_credit_validator_refuses(change: dict, fragment: str) -> None:
    shared = world("task_094")
    plan = decide("task_094", [], shared)
    assert isinstance(plan, Plan)
    reason = rules.savings_credit(v(shared, "task_094"), {**plan.steps[0].args, **change})
    assert reason is not None and fragment in reason, reason


def test_the_report_validator_refuses_wrong_rates_and_another_customer() -> None:
    shared = world("task_094")
    plan = decide("task_094", [], shared)
    assert isinstance(plan, Plan)
    credit, report = plan.steps
    assert shared.call(CREDIT, credit.args).ok
    assert rules.interest_report(v(shared, "task_094"), report.args) is None
    assert "must be 6.85% expected" in (rules.interest_report(v(shared, "task_094"), {**report.args, "expected_apy": 6.0}) or "")
    assert "must be 6.85% expected" in (rules.interest_report(v(shared, "task_094"), {**report.args, "actual_apy": 5.0}) or "")
    assert "verified" in (rules.interest_report(v(shared, "task_094"), {**report.args, "user_id": "someone-else"}) or "")
    assert "credit has to be applied" in (rules.interest_report(v(shared, "task_094"), {**report.args, "amount_difference": 72.0}) or "")


def test_a_second_correction_for_the_same_payment_is_refused_and_said() -> None:
    shared = world("task_094")
    plan = decide("task_094", [], shared)
    assert isinstance(plan, Plan)
    assert shared.call(CREDIT, plan.steps[0].args).ok
    assert "already applied" in (rules.savings_credit(v(shared, "task_094"), plan.steps[0].args) or "")
    again = decide("task_094", [], shared)
    assert isinstance(again, Reply) and "already applied for the latest payment" in again.text and ". ." not in again.text


def test_nothing_owed_is_answered_in_words_with_no_write() -> None:
    shared = world("task_094")
    for row in shared.database.bank_account_transaction_history.data.values():
        if row["type"] == "interest_credit":
            row["amount"] = 548.0  # paid exactly what it should have earned
    result = decide("task_094", ["Gold"], shared)
    assert isinstance(result, Reply) and "6.85%" in result.text and "linked checking" in result.text and "credit card" in result.text


def test_an_unclear_name_is_asked_with_the_customers_own_accounts() -> None:
    result = decide("task_097", ["Bronze"])
    assert isinstance(result, NeedInfo) and "Silver Plus" in result.questions[0].text


def test_the_customers_own_checking_credit_and_a_savings_account_do_not_mix() -> None:
    shared = world("task_094")
    plan = decide("task_094", [], shared)
    assert isinstance(plan, Plan)
    checking = next(a for a in accounts.accounts_of(shared.database, user_of("task_094")) if a.is_checking)
    assert "personal savings" in (rules.savings_credit(v(shared, "task_094"), {**plan.steps[0].args, "account_id": checking.account_id}) or "")


def test_a_balance_that_moved_after_the_interest_is_left_out_not_guessed() -> None:
    shared = world("task_094")
    shared.database.bank_account_transaction_history.data["late"] = {
        "transaction_id": "late", "account_id": "sav_wl94k7m3p8_gold", "date": "11/05/2025", "description": "ACH TRANSFER IN",
        "amount": 1000.0, "type": "ach_transfer_in", "status": "posted",
    }
    result = decide("task_094", [], shared)
    assert isinstance(result, HandOff) and "balance changed" in result.reason


def test_the_report_readback_sees_only_that_the_tool_ran() -> None:
    shared = world("task_094")
    plan = decide("task_094", [], shared)
    assert isinstance(plan, Plan)
    credit, report = plan.steps
    assert rules.interest_report_readback(v(shared, "task_094"), report.args) == "the report was not submitted"
    assert shared.call(CREDIT, credit.args).ok and shared.call(REPORT, report.args).ok
    assert rules.interest_report_readback(v(shared, "task_094"), report.args) is None
