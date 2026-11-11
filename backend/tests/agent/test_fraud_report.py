"""Slice 8 of the fraud composer: the five wallet tasks, each a pass or a documented departure, none writing outside a plan."""

import pathlib

from agent.credit_report import Row, Status, judge, load_departures
from agent.fraud_report import FRAUD_TASKS, FraudReport, fraud_task_ids, render, run_fraud_report
from agent.tier1 import Tier1Result
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex

INDEX = KnowledgeIndex({"doc_x": Document("doc_x", "Internal: x", "lost card protocol")})
TASKS = load_tasks()


def test_the_wallet_tasks_are_077_to_081_and_none_is_excluded():
    assert fraud_task_ids(TASKS) == list(FRAUD_TASKS) == [f"task_{n:03d}" for n in range(77, 82)]


def test_every_wallet_task_passes_strictly_and_nothing_is_written_outside_a_plan():
    report = run_fraud_report(INDEX, TASKS)
    assert [r.task_id for r in report.rows] == list(FRAUD_TASKS)
    assert [r.status for r in report.rows] == [Status.PASS] * 5, [(r.task_id, r.note) for r in report.rows]
    assert all(r.strict for r in report.rows) and report.bypasses == 0 and report.clean
    assert all(r.segments == 1 for r in report.rows), "one request, one preview, one yes per task"


def test_no_departure_was_needed_so_the_file_still_names_only_the_other_specialists_tasks():
    assert set(load_departures()) == {"task_048", "task_026"}


def result(**kw):
    base = dict(task_id="task_x", strict_equal=False, semantic_equal=False, mismatches=(), segments=("a",))
    return Tier1Result(**{**base, **kw})


def test_a_difference_in_an_unlisted_table_is_a_failure():
    departures = {"task_x": {"reason": "why", "tables": ["debit_cards"]}}
    assert judge(result(mismatches=("debit_cards/c1: changed",)), departures).status is Status.DEPARTS
    assert judge(result(mismatches=("debit_cards/c1: x", "accounts/a: y")), departures).status is Status.FAIL
    assert judge(result(mismatches=("debit_cards/c1: x",)), {}).status is Status.FAIL


def test_a_write_outside_the_approved_plan_makes_the_report_not_clean():
    assert not FraudReport((Row("task_x", Status.PASS, True, True, 1, 1, 0.0, 1, ""),)).clean


def test_the_report_renders_every_task_and_says_clean():
    text = render(run_fraud_report(INDEX, TASKS))
    assert "5 wallet tasks: 5 pass" in text and "0 write(s) outside an approved plan" in text and text.rstrip().endswith("CLEAN")
    assert all(t in text for t in FRAUD_TASKS)


def test_the_debit_pilots_tasks_do_not_change_because_078_and_079_are_replayed_here_too():
    from agent.pilot import pilot_tasks

    pilot = {str(t["id"]) for t in pilot_tasks(TASKS)}
    assert {"task_078", "task_079"} <= pilot and len(pilot) == 13


def test_this_module_never_reaches_a_live_model():
    source = pathlib.Path(__import__("agent.fraud_report", fromlist=["x"]).__file__).read_text()
    assert "litellm" not in source and "LiteLLM" not in source and "acompletion" not in source
