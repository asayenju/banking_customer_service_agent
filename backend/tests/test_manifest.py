import collections

import pytest

from banking_tools.manifest import (
    MANIFEST_PATH,
    PILOT,
    TRAPS,
    build_manifest,
    load_manifest,
    load_tasks,
    render,
    uncovered_write_tools,
)

ENTRIES = build_manifest(load_tasks())
BY_ID = {e.task_id: e for e in ENTRIES}
GRADEABLE = [
    e for e in ENTRIES if e.category in ("wholly_one_cluster", "spans_clusters") and e.excluded_reason is None
]


def test_the_committed_manifest_is_what_tasks_json_produces():
    assert MANIFEST_PATH.read_text() == render(ENTRIES), (
        "tasks.json or the cluster tables changed: run `uv run python -m banking_tools.manifest`"
    )
    assert load_manifest() == ENTRIES


def test_the_counts_the_prd_quotes():
    assert len(ENTRIES) == 97
    by_category = collections.Counter(e.category for e in ENTRIES)
    assert by_category == {
        "wholly_one_cluster": 57,
        "spans_clusters": 7,
        "human_transfer_only": 8,
        "customer_side_or_reads_only": 25,
    }
    wholly = collections.Counter(e.clusters for e in ENTRIES if e.category == "wholly_one_cluster")
    assert wholly == {("debit",): 13, ("credit",): 14, ("accounts",): 19, ("fees",): 10, ("profile",): 1}
    spanning = collections.Counter(e.clusters for e in ENTRIES if e.category == "spans_clusters")
    assert spanning == {("accounts", "credit"): 4, ("credit", "debit"): 3}


def test_62_tasks_are_gradeable_and_every_other_task_says_why_not():
    assert len(GRADEABLE) == 62
    for e in ENTRIES:
        assert (e.excluded_reason is None) == (e in GRADEABLE), e.task_id
    assert "bypass" in (BY_ID["task_005"].excluded_reason or "")
    assert BY_ID["task_083"].excluded_reason and BY_ID["task_083"].basis == "ACTION"


def test_action_basis_tasks_are_the_nine_the_evaluator_uses():
    assert sorted(e.task_id for e in ENTRIES if e.basis == "ACTION") == [
        f"task_{n:03d}" for n in (4, 8, 12, 14, 32, 33, 34, 35, 83)
    ]


def test_the_pilot_is_the_13_wholly_debit_tasks():
    assert PILOT == "debit"
    assert [e.task_id for e in ENTRIES if e.pilot] == [
        f"task_{n:03d}" for n in (78, 79, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92)
    ]


def test_the_cross_cluster_debit_tasks_are_not_in_the_pilot():
    for task_id in ("task_077", "task_080", "task_081"):
        assert BY_ID[task_id].clusters == ("credit", "debit") and not BY_ID[task_id].pilot


def test_trap_tasks_are_flagged():
    assert {e.task_id for e in ENTRIES if e.trap} == set(TRAPS)


def test_every_catalog_write_belongs_to_a_cluster_on_purpose():
    assert uncovered_write_tools() == [], "place new tau2 write tools in CLUSTER_TOOLS"


@pytest.mark.parametrize("task_id", ["task_004", "task_032"])
def test_transfer_only_tasks_are_human_transfers_not_customer_side(task_id):
    assert BY_ID[task_id].category == "human_transfer_only" and BY_ID[task_id].human_transfer
