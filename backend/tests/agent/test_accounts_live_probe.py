"""Slice 11 of the accounts specialist: the live probe is prepared, not run. Three tasks can be chosen, the harness
prices them and refuses to spend without --yes, and the new specialist changes nothing for the other two. Nothing in
this file reaches a model: it is blocked in the default run (tests/conftest.py)."""

import pytest

from agent.live import LiveRunRefused, choose_tasks, gradeable_clusters, prepare, task_shape
from agent.specialists.accounts import ACCOUNTS_SPECIALIST
from agent.specialists.credit import CREDIT_SPECIALIST
from agent.specialists.debit import DEBIT_SPECIALIST
from banking_tools.manifest import load_tasks

PROBE = ["task_057", "task_060", "task_075"]  # a false claim and a class choice; open before close; a ranking
ACCOUNTS = [f"task_{n:03d}" for n in (*range(55, 72), 75, 76)]
TASKS = {str(t["id"]): t for t in load_tasks()}
CONFIGS = {"debit": DEBIT_SPECIALIST, "credit": CREDIT_SPECIALIST, "accounts": ACCOUNTS_SPECIALIST}


def test_the_19_account_tasks_can_be_chosen_and_the_other_specialists_keep_theirs():
    both = gradeable_clusters(CONFIGS)
    assert sorted(t for t, c in both.items() if c == "accounts") == ACCOUNTS
    without = gradeable_clusters({"debit": DEBIT_SPECIALIST, "credit": CREDIT_SPECIALIST})
    assert {t: c for t, c in both.items() if c != "accounts"} == without, "adding accounts changes nothing for debit or credit"
    for credit_flow in ("task_043", "task_045", "task_048", "task_051"):
        assert both[credit_flow] == "credit", "a credit flow that pays from checking stays with the credit specialist"


def test_the_probe_tasks_resolve_and_the_whole_specialist_is_over_the_cap():
    both = gradeable_clusters(CONFIGS)
    assert choose_tasks(PROBE, max_tasks=5, gradeable=both) == tuple(PROBE)
    with pytest.raises(LiveRunRefused):
        choose_tasks(["accounts"], max_tasks=5, gradeable=both)


def test_the_harness_prices_the_probe_and_refuses_to_spend_without_yes():
    both = gradeable_clusters(CONFIGS)
    shapes = {t: task_shape(TASKS[t], CONFIGS[c]) for t, c in both.items() if t in PROBE}
    with pytest.raises(LiveRunRefused) as refusal:
        prepare(["--tasks", *PROBE], gradeable=both, shapes=shapes, default_model="openai/gpt-4o-mini")
    text = str(refusal.value)
    assert "3 task(s)" in text and "re-run with --yes to go ahead" in text and "$0." in text


def test_the_probe_cannot_be_widened_past_five_tasks_or_given_a_cost_of_nothing():
    both = gradeable_clusters(CONFIGS)
    shapes = {t: task_shape(TASKS[t], CONFIGS[c]) for t, c in both.items() if c == "accounts"}
    with pytest.raises(LiveRunRefused):
        prepare(["--tasks", *ACCOUNTS[:6]], gradeable=both, shapes=shapes, default_model="openai/gpt-4o-mini")
    with pytest.raises(LiveRunRefused):
        prepare(["--tasks", *PROBE, "--max-cost-usd", "0"], gradeable=both, shapes=shapes, default_model="openai/gpt-4o-mini")
