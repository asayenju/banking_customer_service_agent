"""Offline evidence for the fraud and loss composer: the lost-wallet reference answers replayed through it.

For each of the five wallet tasks (077 to 081) the gold is cut as the orchestrator would cut it: the freezes are
the containment made before the plan, a card found again is unfrozen on its own, and activating the new card is
the debit specialist's request. Everything else is one request: a scripted model hands the composer what the
reference says the customer wanted, the composer plans it (one preview), the plan is approved, and it runs. The
database is compared with the one tau2 gets from the reference. Each task ends PASS, DEPARTS (only in the tables
of a documented departure, ``evals/policy_departures.json``) or FAIL.

Tasks 078 and 079 are also the debit pilot's tasks (``agent.pilot``); they are replayed here too, through the
composer, and the pilot's count does not change. There is no live part: this module never makes a model call.

Run with ``uv run python -m agent.fraud_report``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from agent.credit_report import Row, Status, judge, load_departures
from agent.specialists.fraud import FRAUD_SPECIALIST
from agent.tier1 import run_tier1
from banking_tools.manifest import build_manifest, load_tasks
from knowledge.index import KnowledgeIndex

FRAUD_TASKS = ("task_077", "task_078", "task_079", "task_080", "task_081")


@dataclass(frozen=True)
class FraudReport:
    rows: tuple[Row, ...]

    def count(self, status: Status) -> int:
        return sum(1 for r in self.rows if r.status is status)

    @property
    def bypasses(self) -> int:
        return sum(r.bypasses for r in self.rows)

    @property
    def clean(self) -> bool:
        """Every task passes or departs for a documented reason, and nothing was written outside a plan."""
        return self.count(Status.FAIL) == 0 and self.bypasses == 0


def fraud_task_ids(tasks: Optional[Sequence[Mapping[str, Any]]] = None) -> list[str]:
    """The wallet tasks: the ones that lose cards of both products or only debit, and are not excluded."""
    entries = {e.task_id: e for e in build_manifest(tasks if tasks is not None else load_tasks())}
    return [t for t in FRAUD_TASKS if entries[t].excluded_reason is None]


def run_fraud_report(
    index: KnowledgeIndex,
    tasks: Optional[Sequence[Mapping[str, Any]]] = None,
    departures: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> FraudReport:
    chosen = list(tasks if tasks is not None else load_tasks())
    wanted = set(fraud_task_ids(chosen))
    known = departures if departures is not None else load_departures()
    return FraudReport(tuple(judge(run_tier1(t, FRAUD_SPECIALIST, index=index), known) for t in chosen if str(t["id"]) in wanted))


def render(report: FraudReport) -> str:
    lines = [
        f"{len(report.rows)} wallet tasks: {report.count(Status.PASS)} pass, {report.count(Status.DEPARTS)} depart from the "
        f"reference for a documented reason, {report.count(Status.FAIL)} fail; {report.bypasses} write(s) outside an approved plan "
        "(offline, scripted model, no live call)"
    ]
    for r in report.rows:
        level = "strict" if r.strict else ("semantic" if r.semantic else "differs")
        lines.append(f"- {r.task_id}: {r.status.value} ({level}), {r.segments} requests, {r.model_turns} model turns, {r.seconds:.2f}s")
        if r.note:
            lines.append(f"    {r.note}")
    lines.append("")
    lines.append("CLEAN" if report.clean else "NOT CLEAN")
    return "\n".join(lines)


def main() -> int:  # pragma: no cover - thin wrapper; the parts are tested
    from knowledge.index import get_index

    report = run_fraud_report(get_index())
    print(render(report))
    return 0 if report.clean else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
