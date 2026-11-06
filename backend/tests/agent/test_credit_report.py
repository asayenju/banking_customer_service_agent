"""The credit evidence: 18 tasks, each a pass or a documented departure, none writing outside a plan."""

import json
import pathlib

from agent.credit_report import (
    DEPARTURES_PATH,
    CreditReport,
    Status,
    credit_task_ids,
    judge,
    load_departures,
    render,
    run_credit_report,
)
from agent.tier1 import Tier1Result
from banking_tools.manifest import build_manifest, load_tasks
from knowledge.index import Document, KnowledgeIndex

INDEX = KnowledgeIndex({"doc_credit_cards_x": Document("doc_credit_cards_x", "Internal: x", "credit card procedure")})
TASKS = load_tasks()
EXPECTED = [f"task_{n:03d}" for n in (31, 36, 37, 38, 39, 40, 41, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54)]


def test_the_credit_tasks_are_the_14_wholly_credit_and_the_4_that_also_pay_from_checking():
    assert credit_task_ids(TASKS) == EXPECTED
    entries = {e.task_id: e for e in build_manifest(TASKS)}
    assert sum(entries[t].clusters == ("credit",) for t in EXPECTED) == 14
    assert sum(entries[t].clusters == ("accounts", "credit") for t in EXPECTED) == 4


def test_every_credit_task_passes_or_departs_for_a_documented_reason_and_nothing_is_written_outside_a_plan():
    report = run_credit_report(INDEX, TASKS)
    assert [r.task_id for r in report.rows] == EXPECTED
    failing = [(r.task_id, r.note) for r in report.rows if r.status is Status.FAIL]
    assert not failing, failing
    assert report.bypasses == 0 and report.clean
    assert [r.task_id for r in report.rows if r.status is Status.DEPARTS] == ["task_048"]
    assert report.count(Status.PASS) == 17
    assert all(r.segments >= 1 for r in report.rows), "the specialist made the changes, not direct code"


def test_the_departures_file_names_each_task_with_a_reason_and_the_tables_that_may_differ():
    departures = load_departures()
    assert set(departures) == {"task_048", "task_026"}, "026 is the fees specialist's (docs/fees-policy-decisions.md, section 4)"
    assert departures["task_048"]["tables"] == ["credit_card_transaction_history"]
    assert "entry tier" in departures["task_048"]["reason"]
    assert DEPARTURES_PATH.exists()


def result(**kw):
    base = dict(task_id="task_x", strict_equal=False, semantic_equal=False, mismatches=(), segments=("a",))
    return Tier1Result(**{**base, **kw})


def test_a_difference_in_any_other_table_is_a_failure_even_for_a_task_with_a_departure():
    departures = {"task_x": {"reason": "why", "tables": ["credit_card_transaction_history"]}}
    assert judge(result(mismatches=("credit_card_transaction_history/t1: unexpected change",)), departures).status is Status.DEPARTS
    assert judge(result(mismatches=("credit_card_transaction_history/t1: x", "credit_card_accounts/a: y")), departures).status is Status.FAIL
    assert judge(result(mismatches=("credit_card_transaction_history/t1: x",)), {}).status is Status.FAIL
    assert judge(result(semantic_equal=True, strict_equal=True), {}).status is Status.PASS
    assert judge(result(error="boom"), departures).status is Status.FAIL
    assert judge(result(excluded="gold is invalid"), departures).status is Status.FAIL


def test_a_write_outside_the_approved_plan_makes_the_report_not_clean():
    from agent.credit_report import Row

    row = Row("task_x", Status.PASS, True, True, 1, 1, 0.0, 1, "")
    assert not CreditReport((row,)).clean


def test_the_report_renders_every_task_and_says_clean():
    text = render(run_credit_report(INDEX, TASKS))
    assert "18 credit tasks: 17 pass, 1 depart" in text and "0 write(s) outside an approved plan" in text and text.rstrip().endswith("CLEAN")
    assert all(t in text for t in EXPECTED)


def test_this_module_never_reaches_a_live_model():
    source = pathlib.Path(__import__("agent.credit_report", fromlist=["x"]).__file__).read_text()
    assert "litellm" not in source and "agent.live" not in source and "LiteLLM" not in source
    json.loads(DEPARTURES_PATH.read_text())
