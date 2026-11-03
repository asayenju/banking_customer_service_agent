import json

import pytest

from banking_tools.gold_replay import (
    load_exclusions,
    parity,
    read_log_allowlist,
    render,
    replay_on_session,
    replay_on_tau2,
)
from banking_tools import gold_replay
from banking_tools.manifest import build_manifest, load_tasks
from banking_tools.session import Fidelity, _base_db
from banking_tools.task_state import apply_initial_state

TASKS = load_tasks()
ENTRIES = {e.task_id: e for e in build_manifest(TASKS)}
GRADEABLE = [
    t
    for t in TASKS
    if ENTRIES[t["id"]].category in ("wholly_one_cluster", "spans_clusters")
    and ENTRIES[t["id"]].excluded_reason is None
]
EXCLUSIONS = load_exclusions()


def test_there_are_62_gradeable_tasks():
    assert len(GRADEABLE) == 62


@pytest.mark.parametrize("task", GRADEABLE, ids=lambda t: t["id"])
def test_replaying_the_gold_on_our_session_ends_in_taus_hash(task):
    result = parity(task)
    if task["id"] in EXCLUSIONS:
        assert not result.strict_equal, (
            f"{task['id']} now matches: remove it from evals/parity_exclusions.json"
        )
        return
    assert result.strict_equal, render([result])
    assert result.semantic_equal and result.mismatches == ()


def test_every_exclusion_is_a_gradeable_task_with_a_stated_reason():
    ids = {t["id"] for t in GRADEABLE}
    for task_id, reason in EXCLUSIONS.items():
        assert task_id in ids and reason.strip(), task_id


def test_the_replay_is_not_vacuous_it_changes_the_database_for_every_task():
    for task in GRADEABLE:
        start = apply_initial_state(_base_db().model_copy(deep=True), task)
        expected, _ = replay_on_tau2(task)
        assert expected.get_hash() != start.get_hash(), task["id"]


def test_the_gate_notices_a_session_that_is_not_faithful(monkeypatch):
    task = next(t for t in GRADEABLE if t["id"] == "task_078")
    original = gold_replay.BankingSession.from_task

    def direct(task, **kwargs):
        kwargs["fidelity"] = Fidelity.DIRECT
        return original(task, **kwargs)

    monkeypatch.setattr(gold_replay.BankingSession, "from_task", staticmethod(direct))
    assert not parity(task).strict_equal


def test_the_gate_notices_a_missing_write():
    task = json.loads(json.dumps(next(t for t in GRADEABLE if t["id"] == "task_078")))
    actions = task["evaluation_criteria"]["actions"]
    last_order = max(
        i
        for i, a in enumerate(actions)
        if a["name"] == "call_discoverable_agent_tool"
        and "order_debit_card" in a["arguments"]["agent_tool_name"]
    )
    actions.pop(last_order)
    session, _ = replay_on_session(task)
    expected, _ = replay_on_tau2(next(t for t in GRADEABLE if t["id"] == "task_078"))
    assert session.db_hash() != expected.get_hash()


def test_the_read_log_allowlist_is_the_discoverable_tools_the_gold_calls():
    task = next(t for t in GRADEABLE if t["id"] == "task_085")
    allow = read_log_allowlist(task)
    assert allow and all(n.endswith(tuple("0123456789")) for n in allow)
    assert "log_verification" not in allow and "unlock_discoverable_agent_tool" not in allow


def test_render_names_the_mismatching_tasks():
    from banking_tools.gold_replay import ParityResult

    bad = ParityResult(
        task_id="task_999", strict_equal=False, semantic_equal=False,
        mismatches=("accounts/01: unexpected change",), output_differences=("999_1",),
    )
    text = render([bad])
    assert "task_999" in text and "accounts/01" in text and "999_1" in text
