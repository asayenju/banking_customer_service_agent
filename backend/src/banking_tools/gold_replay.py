"""Replay a task's gold actions on our session and on tau2's own environment.

This is the fidelity gate. tau2 scores a run by hashing the database after replaying
the task's gold actions; if the same gold replayed on our session does not end in the
same hash, a specialist that reaches the right actions could still be graded wrong.

Both sides start from the task's overlay and use the read-log allowlist tau2 derives
from the gold (``runner/build.py``). On the tau2 side every action goes through
``Environment.make_tool_call``. On our side:
- agent tools go through ``BankingSession.call`` in faithful mode;
- tools outside our catalog (``give_discoverable_user_tool``, hand-offs) run on the
  session's toolkit directly, as they are the orchestrator's business, not an agent's;
- customer-side actions run on tau2's user toolkit over the same database.
``unlock_discoverable_agent_tool`` is skipped on our side because the session unlocks
on demand, which leaves the database unchanged.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.data_model import KnowledgeBase, TransactionalDB
from tau2.domains.banking_knowledge.environment import get_environment
from tau2.domains.banking_knowledge.tools import KnowledgeUserTools

from banking_tools.catalog import get_registry
from banking_tools.session import BankingSession, Fidelity, _base_db
from banking_tools.statehash import compare
from banking_tools.task_state import apply_initial_state

EXCLUSIONS_PATH = Path(__file__).resolve().parents[2] / "evals/parity_exclusions.json"

_CALL = "call_discoverable_agent_tool"
_UNLOCK = "unlock_discoverable_agent_tool"


class ParityResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    strict_equal: bool
    semantic_equal: bool
    # "table/pk: problem" for each domain difference.
    mismatches: tuple[str, ...]
    # action_ids whose tool output text differs between the two sides.
    output_differences: tuple[str, ...]


def _actions(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list((task.get("evaluation_criteria") or {}).get("actions") or [])


def read_log_allowlist(task: Mapping[str, Any]) -> frozenset[str]:
    """The discoverable tools tau2 logs reads for: those the gold calls."""
    return frozenset(
        name
        for a in _actions(task)
        if a["name"] == _CALL and (name := (a.get("arguments") or {}).get("agent_tool_name"))
    )


def replay_on_tau2(task: Mapping[str, Any]) -> tuple[TransactionalDB, dict[str, str]]:
    """The database after tau2 replays the gold, and each action's output text."""
    env = get_environment(
        db=_base_db().model_copy(deep=True),
        retrieval_variant="no_knowledge",
        knowledge_base=KnowledgeBase(documents={}),
        read_log_allowlist=set(read_log_allowlist(task)),
    )
    state = Task.model_validate(task).initial_state
    env.set_state(
        initialization_data=state.initialization_data if state else None,
        initialization_actions=None,
        message_history=[],
    )
    outputs: dict[str, str] = {}
    for action in _actions(task):
        try:
            result = env.make_tool_call(
                action["name"], requestor=action["requestor"], **(action.get("arguments") or {})
            )
            outputs[action["action_id"]] = str(result)
        except Exception as exc:  # tau2's evaluator logs and carries on; so do we
            outputs[action["action_id"]] = f"raised {type(exc).__name__}"
    tools_db = env.tools.db
    assert isinstance(tools_db, TransactionalDB)
    return tools_db, outputs


def replay_on_session(task: Mapping[str, Any]) -> tuple[BankingSession, dict[str, str]]:
    session = BankingSession.from_task(
        task, fidelity=Fidelity.TAU2, read_log_allowlist=read_log_allowlist(task)
    )
    user = KnowledgeUserTools(session.database)
    catalog = get_registry()
    outputs: dict[str, str] = {}
    for action in _actions(task):
        name, args, who = action["name"], action.get("arguments") or {}, action["requestor"]
        try:
            if who == "user":
                outputs[action["action_id"]] = str(getattr(user, name)(**args))
            elif name == _UNLOCK:
                continue
            elif name == _CALL:
                inner = json.loads(args.get("arguments") or "{}", parse_int=float)
                outputs[action["action_id"]] = session.call(args["agent_tool_name"], inner).text
            elif name in catalog:
                outputs[action["action_id"]] = session.call(name, args).text
            else:
                outputs[action["action_id"]] = str(getattr(session.toolkit, name)(**args))
        except Exception as exc:
            outputs[action["action_id"]] = f"raised {type(exc).__name__}"
    return session, outputs


def parity(task: Mapping[str, Any]) -> ParityResult:
    start = apply_initial_state(_base_db().model_copy(deep=True), task)
    expected, expected_outputs = replay_on_tau2(task)
    session, got_outputs = replay_on_session(task)
    result = compare(start, expected, session.database, read_log_names=())
    differing = tuple(
        action_id
        for action_id, text in got_outputs.items()
        if expected_outputs.get(action_id) != text
    )
    return ParityResult(
        task_id=str(task["id"]),
        strict_equal=result.strict_equal,
        semantic_equal=result.semantic_equal,
        mismatches=tuple(f"{m.table}/{m.pk}: {m.problem}" for m in result.mismatches),
        output_differences=differing,
    )


def load_exclusions(path: Path = EXCLUSIONS_PATH) -> dict[str, str]:
    """task id -> why our replay may not match tau2's. Empty is the goal."""
    exclusions: dict[str, str] = json.loads(path.read_text())
    return exclusions


def run_gate(tasks: Sequence[Mapping[str, Any]]) -> list[ParityResult]:
    return [parity(t) for t in tasks]


def render(results: Sequence[ParityResult]) -> str:
    bad = [r for r in results if not r.strict_equal]
    lines = [f"{len(results)} tasks, {len(results) - len(bad)} strict matches, {len(bad)} mismatches"]
    for r in bad:
        lines.append(f"- {r.task_id} (semantic {'ok' if r.semantic_equal else 'DIFFERS'})")
        lines += [f"    {m}" for m in r.mismatches[:5]]
        if r.output_differences:
            lines.append(f"    outputs differ at: {', '.join(r.output_differences[:8])}")
    return "\n".join(lines)
