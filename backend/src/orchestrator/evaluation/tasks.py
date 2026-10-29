"""Turn tau2 banking tasks into eval cases: what the customer wants, as intent kinds.

The expected kinds come from the task's expected tool actions. Discoverable
tools are wrapped in ``call_discoverable_agent_tool``, so the real name is read
from its arguments. Only tools the catalog labels write or handoff count: reads
are implicit in almost any conversation, so asking the decomposer to name them
would grade noise. Tools with no customer goal (``log_verification``) and
user-side tools (applying for a card) have no kind and are dropped.
"""

import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from banking_tools.catalog import get_registry
from banking_tools.types import ToolKind
from orchestrator.taxonomy import TOOL_INTENT
from orchestrator.types import IntentKind

_REPO_ROOT = Path(__file__).resolve().parents[4]
TASKS_PATH = (
    _REPO_ROOT
    / "tau2-bench/data/tau2/domains/banking_knowledge/tasks.json"
)
DATASET_PATH = _REPO_ROOT / "backend/evals/banking_eval_set.json"
HOLDOUT_PATH = _REPO_ROOT / "backend/evals/banking_holdout_set.json"

DEFAULT_COUNT = 50
_WRAPPER = "call_discoverable_agent_tool"


class EvalCase(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    task_id: str
    # First-person opening message written from the task's role-play brief.
    message: str
    # Action kinds the message itself asks for (the grading target).
    stated: tuple[IntentKind, ...]
    # Action kinds tau2's expected tool actions imply once the bank's policy has
    # been followed; often more than the customer asks for. Reported, not graded.
    expected: tuple[IntentKind, ...]


def load_tasks(path: Path = TASKS_PATH) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = json.loads(path.read_text())
    return tasks


def expected_kinds(task: Mapping[str, Any]) -> frozenset[IntentKind]:
    registry = get_registry()
    kinds: set[IntentKind] = set()
    for action in task["evaluation_criteria"].get("actions") or []:
        if action.get("requestor") != "assistant":
            continue
        name = action["name"]
        if name == _WRAPPER:
            name = (action.get("arguments") or {}).get("agent_tool_name", "")
        spec = registry.get(name)
        kind = TOOL_INTENT.get(name)
        if spec is None or kind is None or spec.kind is ToolKind.READ:
            continue
        if kind is not IntentKind.OTHER:
            kinds.add(kind)
    return frozenset(kinds)


def select_tasks(
    tasks: Sequence[Mapping[str, Any]],
    count: int = DEFAULT_COUNT,
    exclude: Collection[str] = (),
) -> list[Mapping[str, Any]]:
    """A fixed, varied subset: greedily prefer tasks whose kinds are least picked so far."""
    pool = sorted(
        ((t, expected_kinds(t)) for t in tasks if t["id"] not in exclude and expected_kinds(t)),
        key=lambda pair: str(pair[0]["id"]),
    )
    picked: list[Mapping[str, Any]] = []
    seen: dict[IntentKind, int] = {}
    while pool and len(picked) < count:
        best = max(
            range(len(pool)),
            key=lambda i: (sum(1 / (1 + seen.get(k, 0)) for k in pool[i][1]) / len(pool[i][1]), -i),
        )
        task, kinds = pool.pop(best)
        picked.append(task)
        for k in kinds:
            seen[k] = seen.get(k, 0) + 1
    return picked
