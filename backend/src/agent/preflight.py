"""Validate every task before any money is spent on it. Free: no model is called and no key is read.

For each of the benchmark's tasks this says whether a paid run of it could mean anything:

  - ``runnable``  the live harness can grade it (a built specialist, a valid gold, a database comparison);
  - ``offline``   the specialist reproduces the reference answer from the reference facts (the five offline
                  reports: debit pilot, credit, accounts, fees, fraud); a task that fails here cannot pass live,
                  because the plumbing or a rule is wrong, not the model;
  - ``cassette``  a recorded live conversation, if there is one, is replayed without a model: a stale one
                  ("recorded for a different prompt") is not a result, it is an error waiting to be paid for;
  - ``asks``      how many "if the agent asks / only if" replies the customer's script holds. The live harness hands
                  the model the whole script once and cannot answer a question, so a task that depends on those
                  replies is not fairly measured by it; this is a warning, not a block.

Run with ``uv run python -m agent.preflight`` (all tasks) or ``--tasks task_078 task_079``. ``agent.live`` runs the same
check on the tasks it is about to run and refuses the ones that are blocked.
"""

import argparse
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Optional

from agent.credit_report import Status, run_credit_report
from banking_tools.manifest import build_manifest, load_tasks
from knowledge.index import KnowledgeIndex

_CONDITIONAL_REPLY = re.compile(r"\bif (?:the )?agent\b|\bonly if\b|\bif asked\b", re.IGNORECASE)


class Offline(StrEnum):
    PASS = "PASS"
    DEPARTS = "DEPARTS"
    FAIL = "FAIL"
    EXCLUDED = "EXCLUDED"
    NOT_BUILT = "NOT BUILT"


class Cassette(StrEnum):
    NONE = "none"
    FRESH = "fresh"
    STALE = "STALE"
    PARTIAL = "partial"


@dataclass(frozen=True)
class Check:
    task_id: str
    category: str
    clusters: tuple[str, ...]
    runnable: bool
    offline: Offline
    cassette: Cassette
    asks: int
    reason: str

    @property
    def blocked(self) -> bool:
        """A paid run of this task would waste money: it cannot be graded, cannot pass, or its recording is stale."""
        return self.blocked_without(rerecord=False)

    def blocked_without(self, *, rerecord: bool) -> bool:
        """The same, when the caller has asked to re-record: a stale recording is then the point, not a problem."""
        stale = self.cassette is Cassette.STALE and not rerecord
        return not self.runnable or self.offline in (Offline.FAIL, Offline.EXCLUDED) or stale


def conditional_replies(task: Mapping[str, Any]) -> int:
    return len(_CONDITIONAL_REPLY.findall(str((task.get("user_scenario") or {}).get("instructions") or "")))


def offline_statuses(index: KnowledgeIndex, tasks: Sequence[Mapping[str, Any]]) -> dict[str, Offline]:
    """What the five offline reports say, per task. A task in two reports (078 and 079) must pass in both."""
    from agent.accounts_report import run_accounts_report
    from agent.fees_report import run_fees_report
    from agent.fraud_report import run_fraud_report
    from agent.pilot import run_pilot_tier1

    found: dict[str, list[Offline]] = {}

    def add(task_id: str, status: Offline) -> None:
        found.setdefault(task_id, []).append(status)

    for report in (run_credit_report(index, tasks), run_accounts_report(index, tasks), run_fees_report(index, tasks), run_fraud_report(index, tasks)):
        for row in report.rows:
            add(row.task_id, Offline(row.status.value))
    for result in run_pilot_tier1(index, tasks).results:
        add(result.task_id, Offline.EXCLUDED if result.excluded else (Offline.PASS if result.passed else Offline.FAIL))
    order = [Offline.FAIL, Offline.EXCLUDED, Offline.DEPARTS, Offline.PASS]
    return {t: min(statuses, key=order.index) for t, statuses in found.items()}


class _NoSpend(Exception):
    pass


def cassette_state(task: Mapping[str, Any], config: Any, model: str, index: KnowledgeIndex) -> Cassette:
    """Replay the recorded conversation without a model. Nothing recorded for this model is NONE."""
    from agent.live import DEFAULT_CASSETTES, Plan, RunContext, cassette_path, run_live_task

    task_id = str(task["id"])
    if not any(cassette_path(DEFAULT_CASSETTES, task_id, n, model).exists() for n in range(6)):
        return Cassette.NONE

    def refuse(_: str) -> Any:
        raise _NoSpend()

    # The agent loop turns a failed model call into a quiet "model_failed", which hides a cassette mismatch, so
    # watch the recording itself.
    from agent import live

    mismatches: list[str] = []
    original = live.CassetteClient.chat

    async def watching(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await original(self, *args, **kwargs)
        except live.CassetteMismatch as exc:
            mismatches.append(str(exc))
            raise

    context = RunContext(plan=Plan((task_id,), 1, 0.0, model, 0.0, 0.0), make_client=refuse, rerecord=False)
    live.CassetteClient.chat = watching  # type: ignore[method-assign]
    try:
        result = run_live_task(task, config, DEFAULT_CASSETTES, context, index=index)
    except _NoSpend:
        return Cassette.PARTIAL  # some segment was never recorded: running it would pay for that part
    finally:
        live.CassetteClient.chat = original  # type: ignore[method-assign]
    return Cassette.STALE if mismatches or result.note.startswith("CassetteMismatch") else Cassette.FRESH


def validate(
    index: KnowledgeIndex,
    *,
    task_ids: Optional[Sequence[str]] = None,
    model: str,
    configs: Optional[Mapping[str, Any]] = None,
) -> list[Check]:
    from agent.live import gradeable_clusters

    if configs is None:
        from agent.specialists.accounts import ACCOUNTS_SPECIALIST
        from agent.specialists.credit import CREDIT_SPECIALIST
        from agent.specialists.debit import DEBIT_SPECIALIST
        from agent.specialists.fees import FEES_SPECIALIST

        configs = {"debit": DEBIT_SPECIALIST, "credit": CREDIT_SPECIALIST, "accounts": ACCOUNTS_SPECIALIST, "fees": FEES_SPECIALIST}
    tasks = load_tasks()
    wanted = set(task_ids) if task_ids is not None else {str(t["id"]) for t in tasks}
    by_id = {str(t["id"]): t for t in tasks}
    unknown = wanted - set(by_id)
    if unknown:
        raise KeyError(f"unknown task ids: {', '.join(sorted(unknown))}")
    entries = {e.task_id: e for e in build_manifest(tasks)}
    gradeable = gradeable_clusters(configs)
    offline = offline_statuses(index, tasks)
    checks: list[Check] = []
    for task_id in sorted(wanted):
        entry, task = entries[task_id], by_id[task_id]
        runnable = task_id in gradeable
        status = offline.get(task_id)
        if entry.excluded_reason is not None:
            status, reason = Offline.EXCLUDED, entry.excluded_reason
        elif status is None:
            status, reason = Offline.NOT_BUILT, "no offline report covers this task"
        else:
            reason = ""
        if not runnable and not reason:
            reason = "the live harness cannot grade it (no single built specialist, or not a database comparison)"
        cassette = cassette_state(task, configs[gradeable[task_id]], model, index) if runnable else Cassette.NONE
        if cassette is Cassette.STALE:
            reason = (reason + "; " if reason else "") + "its recording is for a different prompt: re-record it or drop it"
        checks.append(Check(task_id, entry.category, tuple(entry.clusters), runnable, status, cassette, conditional_replies(task), reason))
    return checks


def render(checks: Sequence[Check], *, only_problems: bool = False) -> str:
    runnable = [c for c in checks if c.runnable]
    lines = [
        f"{len(checks)} tasks: {len(runnable)} runnable live, {sum(1 for c in checks if c.blocked and c.runnable)} of them blocked, "
        f"{sum(1 for c in checks if not c.runnable)} not runnable by the live harness"
    ]
    by_category: dict[str, int] = {}
    for c in checks:
        by_category[c.category] = by_category.get(c.category, 0) + 1
    lines.append("categories: " + ", ".join(f"{k} {v}" for k, v in sorted(by_category.items())))
    lines.append("offline: " + ", ".join(f"{s.value} {sum(1 for c in checks if c.offline is s)}" for s in Offline if any(c.offline is s for c in checks)))
    lines.append("cassettes (runnable): " + ", ".join(f"{s.value} {sum(1 for c in runnable if c.cassette is s)}" for s in Cassette))
    heavy = [c for c in runnable if c.asks >= 3]
    lines.append(f"{len(heavy)} runnable tasks lean on 3 or more 'if the agent asks' replies the one-shot harness cannot give")
    lines.append("")
    for c in checks:
        if only_problems and not (c.runnable and (c.blocked or c.asks >= 3 or c.cassette is not Cassette.NONE)):
            continue
        flag = "BLOCKED" if c.blocked and c.runnable else ("n/a" if not c.runnable else ("warn" if c.asks >= 3 else "ok"))
        lines.append(
            f"{c.task_id}  {flag:<7} {c.category:<28} {'+'.join(c.clusters) or '-':<18} offline {c.offline.value:<9} cassette {c.cassette.value:<7} asks {c.asks}"
            + (f"  | {c.reason}" if c.reason else "")
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - thin wrapper; the parts are tested
    from knowledge.index import get_index
    from orchestrator.config import agent_model

    parser = argparse.ArgumentParser(prog="agent.preflight")
    parser.add_argument("--tasks", nargs="*", default=None, help="task ids; omit for every task")
    parser.add_argument("--model", default=None)
    parser.add_argument("--problems", action="store_true", help="show only blocked or interactive tasks")
    args = parser.parse_args(argv)
    checks = validate(get_index(), task_ids=args.tasks, model=args.model or agent_model())
    print(render(checks, only_problems=args.problems))
    return 1 if any(c.blocked and c.runnable for c in checks) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
