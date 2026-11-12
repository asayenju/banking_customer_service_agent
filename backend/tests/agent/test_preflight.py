"""The free check before any paid run: every task classified, a stale recording or a plumbing failure blocked."""

import json
from functools import cache

import agent.live as live
from agent.preflight import Cassette, Check, Offline, cassette_state, conditional_replies, render, validate
from agent.specialists.debit import DEBIT_SPECIALIST
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex

INDEX = KnowledgeIndex({"doc_x": Document("doc_x", "Internal: x", "debit card procedure")})
MODEL = "openai/gpt-4o"
TASKS = {str(t["id"]): t for t in load_tasks()}


@cache
def everything() -> dict[str, Check]:
    return {c.task_id: c for c in validate(INDEX, model="openai/never-recorded")}


def test_every_task_is_classified_and_the_numbers_add_up():
    checks = everything()
    assert len(checks) == 97
    runnable = [c for c in checks.values() if c.runnable]
    assert len(runnable) == 59
    assert all(not c.runnable for c in checks.values() if c.offline is Offline.EXCLUDED)
    assert all(c.offline is not Offline.NOT_BUILT for c in runnable), "a runnable task always has an offline report"


def test_a_task_the_specialist_cannot_reproduce_offline_is_blocked():
    """Task 082's offline replay differs from the reference (the dispute's card action closes a card the reference
    leaves open, and its provisional credit differs). It failed before the fraud work as well, so it cannot pass live."""
    c = everything()["task_082"]
    assert c.runnable and c.offline is Offline.FAIL and c.blocked
    assert not everything()["task_078"].blocked and not everything()["task_079"].blocked


def test_the_tasks_a_paid_run_cannot_grade_are_not_runnable():
    checks = everything()
    for task_id, why in (("task_001", "customer-side"), ("task_004", "hand-off"), ("task_083", "gold")):
        assert not checks[task_id].runnable and checks[task_id].blocked, (task_id, why)


def test_the_customers_conditional_replies_are_counted():
    assert conditional_replies({"user_scenario": {"instructions": "If the agent asks X say Y. ONLY IF asked about Z. if agent offers W"}}) == 3
    assert conditional_replies({"user_scenario": {"instructions": "say hello"}}) == 0
    assert everything()["task_091"].asks > everything()["task_078"].asks > 0


def test_a_recording_for_a_different_prompt_is_stale_not_fresh(tmp_path, monkeypatch):
    """The agent loop reports a cassette mismatch as a quiet model failure, so the check watches the recording."""
    monkeypatch.setattr(live, "DEFAULT_CASSETTES", tmp_path)
    task = TASKS["task_078"]
    assert cassette_state(task, DEBIT_SPECIALIST, MODEL, INDEX) is Cassette.NONE
    turn = {"text": None, "tool_calls": [], "usage": {"input_tokens": 1, "output_tokens": 1}}
    live.cassette_path(tmp_path, "task_078", 0, MODEL).write_text(json.dumps([{"request": "not-the-real-hash", "turn": turn}]))
    assert cassette_state(task, DEBIT_SPECIALIST, MODEL, INDEX) is Cassette.STALE
    assert live.CassetteClient.chat.__name__ == "chat", "the watch is removed afterwards"


def test_a_stale_recording_blocks_unless_the_run_is_a_rerecord():
    stale = Check("task_x", "wholly_one_cluster", ("debit",), True, Offline.PASS, Cassette.STALE, 0, "")
    assert stale.blocked and stale.blocked_without(rerecord=False) and not stale.blocked_without(rerecord=True)
    failing = Check("task_y", "wholly_one_cluster", ("debit",), True, Offline.FAIL, Cassette.NONE, 0, "")
    assert failing.blocked_without(rerecord=True), "re-recording does not fix a plumbing failure"


def test_the_report_names_the_blocked_tasks_and_the_counts():
    text = render(list(everything().values()), only_problems=True)
    assert text.startswith("97 tasks: 59 runnable live") and "task_082  BLOCKED" in text and "task_078  warn" in text
