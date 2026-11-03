"""How well each scope covers what the tau2 tasks need. Reported, never asserted.

For every task wholly inside one cluster, the opening message the simulated customer
starts with is used as the search query, and we ask two questions about the task's
``required_documents``: how many of them are inside the specialist's scope at all
(coverage), and how many of them the top 5 results contain (recall@5).

Neither number gates anything. Scopes are written by topic so they do not overfit these
tasks; this report is how we find out how much that costs, and where a scope is too narrow.

Run with ``uv run python -m knowledge.report``.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from banking_tools.manifest import build_manifest, load_tasks
from knowledge.index import KnowledgeIndex, get_index
from knowledge.scopes import ACCOUNTS, CREDIT_CARDS, DEBIT, FEES, DocScope

SCOPE_FOR_CLUSTER: Mapping[str, DocScope] = {
    "debit": DEBIT,
    "credit": CREDIT_CARDS,
    "accounts": ACCOUNTS,
    "fees": FEES,
}
_OPENING = re.compile(r'\*\*Opening:\*\*\s*"(.+?)"', re.DOTALL)
_GOAL = re.compile(r"\*\*Your goal:\*\*\s*(.+?)(?:\n\n|\Z)", re.DOTALL)


@dataclass(frozen=True)
class TaskRecall:
    task_id: str
    cluster: str
    required: int
    in_scope: int
    in_top5: int


def opening_message(task: Mapping[str, Any]) -> str:
    """What the customer starts with: their first line, else their stated goal, else the brief."""
    text = (task.get("user_scenario") or {}).get("instructions") or ""
    if not isinstance(text, str):
        return ""
    if match := _OPENING.search(text):
        return match.group(1)
    if match := _GOAL.search(text):
        return match.group(1).strip()[:800]
    return text[:800]


def measure(index: KnowledgeIndex, tasks: Sequence[Mapping[str, Any]]) -> list[TaskRecall]:
    entries = {e.task_id: e for e in build_manifest(tasks)}
    rows: list[TaskRecall] = []
    for task in tasks:
        entry = entries[str(task["id"])]
        if entry.category != "wholly_one_cluster" or entry.clusters[0] not in SCOPE_FOR_CLUSTER:
            continue
        scope = SCOPE_FOR_CLUSTER[entry.clusters[0]]
        required = set(task.get("required_documents") or [])
        query = opening_message(task)
        top = {h.doc_id for h in index.search(query, scope, 5)} if query else set()
        inside = {d for d in required if (doc := index.get(d)) and scope.allows(doc.id, doc.title)}
        rows.append(
            TaskRecall(
                task_id=entry.task_id,
                cluster=entry.clusters[0],
                required=len(required),
                in_scope=len(inside),
                in_top5=len(required & top),
            )
        )
    return rows


def summarize(rows: Sequence[TaskRecall]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for cluster in sorted({r.cluster for r in rows}):
        mine = [r for r in rows if r.cluster == cluster]
        needed = sum(r.required for r in mine)
        out[cluster] = {
            "tasks": float(len(mine)),
            "coverage": sum(r.in_scope for r in mine) / needed if needed else 0.0,
            "recall_at_5": sum(r.in_top5 for r in mine) / needed if needed else 0.0,
        }
    return out


def render(summary: Mapping[str, Mapping[str, float]]) -> str:
    lines = ["cluster     tasks  required docs in scope  required docs in top 5"]
    for cluster, s in summary.items():
        lines.append(f"{cluster:<10} {int(s['tasks']):>5}  {s['coverage']:>20.1%}  {s['recall_at_5']:>20.1%}")
    return "\n".join(lines)


def main() -> None:
    print(render(summarize(measure(get_index(), load_tasks()))))


if __name__ == "__main__":
    main()
