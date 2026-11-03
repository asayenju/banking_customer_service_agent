import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.data_model import KnowledgeBase
from tau2.domains.banking_knowledge.environment import get_environment

from banking_tools import BankingSession
from banking_tools.session import _base_db, tau2_now
from banking_tools.task_state import apply_initial_state

TASKS_PATH = (
    Path(__file__).resolve().parents[2]
    / "tau2-bench/data/tau2/domains/banking_knowledge/tasks.json"
)
TASKS = json.loads(TASKS_PATH.read_text())
WITH_DATA = [t for t in TASKS if (t.get("initial_state") or {}).get("initialization_data")]
EST = timezone(timedelta(hours=-5))


def tau2_hash(task):
    """What tau2 itself gets: a fresh environment after set_state."""
    env = get_environment(
        db=_base_db().model_copy(deep=True),
        retrieval_variant="no_knowledge",
        knowledge_base=KnowledgeBase(documents={}),
    )
    state = Task.model_validate(task).initial_state
    env.set_state(
        initialization_data=state.initialization_data if state else None,
        initialization_actions=None,
        message_history=[],
    )
    return env.get_db_hash(), env.get_user_db_hash()


def test_the_tasks_file_has_overlays_to_test():
    assert len(WITH_DATA) >= 40


@pytest.mark.parametrize("task", WITH_DATA, ids=lambda t: t["id"])
def test_a_session_from_a_task_has_the_hash_tau2_gets(task):
    agent_hash, user_hash = tau2_hash(task)
    assert agent_hash == user_hash, "tau2 shares one database between agent and user"
    assert BankingSession.from_task(task).db_hash() == agent_hash


def test_a_task_without_an_overlay_is_the_pristine_database():
    plain = next(t for t in TASKS if not (t.get("initial_state") or {}).get("initialization_data"))
    assert BankingSession.from_task(plain).db_hash() == BankingSession().db_hash()


def test_an_overlay_changes_the_database_and_never_the_shared_base():
    task = next(t for t in WITH_DATA if (t["initial_state"]["initialization_data"] or {}).get("agent_data"))
    base_before = BankingSession().db_hash()
    assert BankingSession.from_task(task).db_hash() != base_before
    assert BankingSession().db_hash() == base_before


def test_initialization_actions_are_refused_rather_than_ignored():
    task = {"id": "x", "initial_state": {"initialization_actions": [{"func_name": "f"}]}}
    with pytest.raises(ValueError, match="initialization_actions"):
        apply_initial_state(_base_db().model_copy(deep=True), task)


def test_audit_timestamps_are_tau2s_locked_clock_not_wall_time():
    session = BankingSession()
    result = session.call("change_user_email", {"user_id": "123", "new_email": "a@b.co"})
    assert result.ok
    (entry,) = session.audit_log
    assert entry.timestamp == datetime(2025, 11, 14, 3, 40, 0, tzinfo=EST)
    assert tau2_now() == entry.timestamp


def test_the_clock_can_be_injected():
    moment = datetime(2030, 1, 2, 3, 4, 5, tzinfo=EST)
    session = BankingSession(clock=lambda: moment)
    session.call("change_user_email", {"user_id": "123", "new_email": "a@b.co"})
    assert session.audit_log[0].timestamp == moment
