"""The fees evidence: 10 tasks, nine strict passes and one documented departure, none writing outside a plan."""

import json
import pathlib

from agent.credit_report import DEPARTURES_PATH, Row, Status, judge, load_departures
from agent.fees_report import FeesReport, fees_task_ids, render, run_fees_report
from agent.tier1 import Tier1Result
from banking_tools.manifest import build_manifest, load_manifest, load_tasks
from knowledge.index import Document, KnowledgeIndex

INDEX = KnowledgeIndex({"doc_fees_x": Document("doc_fees_x", "Internal: x", "fee procedure")})
TASKS = load_tasks()
EXPECTED = ["task_026", "task_028", "task_072", "task_073", "task_074", *(f"task_{n:03d}" for n in range(93, 98))]


def test_the_fee_tasks_are_the_10_that_change_only_fees_interest_or_rewards():
    assert fees_task_ids(TASKS) == EXPECTED
    entries = {e.task_id: e for e in build_manifest(TASKS)}
    assert all(entries[t].clusters == ("fees",) and entries[t].category == "wholly_one_cluster" for t in EXPECTED)


def test_every_fee_task_passes_and_nothing_is_written_outside_a_plan():
    report = run_fees_report(INDEX, TASKS)
    assert [r.task_id for r in report.rows] == EXPECTED
    failing = [(r.task_id, r.note) for r in report.rows if r.status is Status.FAIL]
    assert not failing, failing
    assert report.bypasses == 0 and report.clean
    assert report.count(Status.PASS) == 9 and report.count(Status.DEPARTS) == 1
    assert report.strict == 9, "nine databases equal tau2's own hash, not only semantically"
    assert [r.task_id for r in report.rows if r.status is Status.DEPARTS] == ["task_026"]
    assert all(r.segments >= 1 for r in report.rows), "the specialist made the changes, not direct code"


def test_only_task_026_is_a_departure_and_097_is_not():
    departures = load_departures()
    assert set(departures) & set(EXPECTED) == {"task_026"}
    assert departures["task_026"]["tables"] == ["credit_card_transaction_history"]
    assert "task_097" not in departures, "the interest method reproduces 097 exactly"


def test_the_manifest_counts_for_the_fees_cluster():
    mine = [e for e in load_manifest() if "fees" in e.clusters]
    assert sorted(e.task_id for e in mine) == EXPECTED and all(e.excluded_reason is None for e in mine)


def result(**kw):
    base = dict(task_id="task_x", strict_equal=False, semantic_equal=False, mismatches=(), segments=("a",))
    return Tier1Result(**{**base, **kw})


def test_a_difference_in_the_rewards_table_is_a_failure_unless_a_departure_names_it():
    departures = {"task_x": {"reason": "why", "tables": ["credit_card_transaction_history"]}}
    assert judge(result(mismatches=("credit_card_transaction_history/t: x",)), departures).status is Status.DEPARTS
    assert judge(result(mismatches=("credit_card_transaction_history/t: x",)), {}).status is Status.FAIL
    assert judge(result(mismatches=("credit_card_transaction_history/t: x", "accounts/a: y")), departures).status is Status.FAIL


def test_a_write_outside_the_approved_plan_makes_the_report_not_clean():
    assert not FeesReport((Row("task_x", Status.PASS, True, True, 1, 1, 0.0, 1, ""),)).clean
    assert FeesReport((Row("task_x", Status.PASS, True, True, 1, 1, 0.0, 0, ""),)).clean
    assert not FeesReport((Row("task_x", Status.FAIL, False, False, 1, 1, 0.0, 0, "no"),)).clean


def test_the_report_renders_every_task_and_says_clean():
    text = render(run_fees_report(INDEX, TASKS))
    assert "10 fee tasks: 9 pass (9 strictly), 1 depart" in text and "0 write(s) outside an approved plan" in text
    assert text.rstrip().endswith("CLEAN") and all(t in text for t in EXPECTED)


def test_this_module_never_reaches_a_live_model():
    source = pathlib.Path(__import__("agent.fees_report", fromlist=["x"]).__file__).read_text()
    assert "litellm" not in source and "agent.live" not in source and "LiteLLM" not in source
    json.loads(DEPARTURES_PATH.read_text())
