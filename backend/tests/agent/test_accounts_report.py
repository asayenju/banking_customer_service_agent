"""The accounts evidence: 19 tasks, each a pass or a documented departure, none writing outside a plan."""

import json
import pathlib

from agent.accounts_report import (
    AccountsReport,
    accounts_task_ids,
    render,
    run_accounts_report,
)
from agent.credit_report import DEPARTURES_PATH, Row, Status, judge, load_departures
from agent.tier1 import Tier1Result
from banking_tools.manifest import build_manifest, load_manifest, load_tasks
from knowledge.index import Document, KnowledgeIndex

INDEX = KnowledgeIndex({"doc_accounts_x": Document("doc_accounts_x", "Internal: x", "account procedure")})
TASKS = load_tasks()
EXPECTED = [f"task_{n:03d}" for n in (*range(55, 72), 75, 76)]


def test_the_account_tasks_are_the_19_that_change_only_bank_accounts():
    assert accounts_task_ids(TASKS) == EXPECTED
    entries = {e.task_id: e for e in build_manifest(TASKS)}
    assert all(entries[t].clusters == ("accounts",) and entries[t].category == "wholly_one_cluster" for t in EXPECTED)


def test_the_four_credit_flows_and_the_fee_tasks_are_not_counted_here():
    ids = set(accounts_task_ids(TASKS))
    assert not ids & {"task_043", "task_045", "task_048", "task_051"}, "those belong to the credit report"
    assert not ids & {"task_072", "task_073", "task_074", *(f"task_{n:03d}" for n in range(93, 98))}, "those belong to Issue #43"


def test_every_account_task_passes_and_nothing_is_written_outside_a_plan():
    report = run_accounts_report(INDEX, TASKS)
    assert [r.task_id for r in report.rows] == EXPECTED
    failing = [(r.task_id, r.note) for r in report.rows if r.status is Status.FAIL]
    assert not failing, failing
    assert report.bypasses == 0 and report.clean
    assert report.count(Status.PASS) == 19 and report.count(Status.DEPARTS) == 0, "no account departure was needed"
    assert report.strict == 19, "the database equals tau2's own hash, not only semantically"
    assert all(r.segments >= 1 for r in report.rows), "the specialist made the changes, not direct code"


def test_no_account_task_is_listed_as_a_departure_from_policy():
    departures = load_departures()
    assert not set(departures) & set(EXPECTED), "the validators accept every reference write (test_accounts_rules)"
    assert DEPARTURES_PATH.exists()


def test_the_manifest_counts_for_the_accounts_cluster():
    manifest = load_manifest()
    mine = [e for e in manifest if "accounts" in e.clusters]
    assert sorted(e.task_id for e in mine if e.clusters == ("accounts",)) == EXPECTED
    assert sorted(e.task_id for e in mine if e.clusters != ("accounts",)) == ["task_043", "task_045", "task_048", "task_051"]
    assert all(e.excluded_reason is None for e in mine)


def result(**kw):
    base = dict(task_id="task_x", strict_equal=False, semantic_equal=False, mismatches=(), segments=("a",))
    return Tier1Result(**{**base, **kw})


def test_a_difference_in_the_accounts_table_is_a_failure_unless_a_departure_names_it():
    departures = {"task_x": {"reason": "why", "tables": ["accounts"]}}
    assert judge(result(mismatches=("accounts/a: unexpected change",)), departures).status is Status.DEPARTS
    assert judge(result(mismatches=("accounts/a: x",)), {}).status is Status.FAIL
    assert judge(result(mismatches=("accounts/a: x", "credit_card_accounts/c: y")), departures).status is Status.FAIL


def test_a_write_outside_the_approved_plan_makes_the_report_not_clean():
    row = Row("task_x", Status.PASS, True, True, 1, 1, 0.0, 1, "")
    assert not AccountsReport((row,)).clean
    assert AccountsReport((Row("task_x", Status.PASS, True, True, 1, 1, 0.0, 0, ""),)).clean
    assert not AccountsReport((Row("task_x", Status.FAIL, False, False, 1, 1, 0.0, 0, "no"),)).clean


def test_the_report_renders_every_task_and_says_clean():
    text = render(run_accounts_report(INDEX, TASKS))
    assert "19 account tasks: 19 pass (19 strictly), 0 depart" in text and "0 write(s) outside an approved plan" in text
    assert text.rstrip().endswith("CLEAN") and all(t in text for t in EXPECTED)


def test_this_module_never_reaches_a_live_model():
    source = pathlib.Path(__import__("agent.accounts_report", fromlist=["x"]).__file__).read_text()
    assert "litellm" not in source and "agent.live" not in source and "LiteLLM" not in source
    json.loads(DEPARTURES_PATH.read_text())
