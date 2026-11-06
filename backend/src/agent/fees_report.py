"""Offline evidence for the fees specialist: the reference answers replayed through it.

For each of the 10 fee, interest and rewards tasks (026, 028, 072 to 074, 093 to 097) a scripted model hands the
specialist what the reference says the customer wanted, the specialist plans and runs it, and the database is
compared with the one tau2 gets from the reference. Each task ends in exactly one of three states, judged the
way the credit and accounts reports judge them:

  PASS     the database matches the reference;
  DEPARTS  it differs, but only in the tables of a documented departure from policy
           (``evals/policy_departures.json``: 026, one reward row the documented floor rounds a point lower);
  FAIL     anything else.

The report is a measurement, not a gate, and there is no live part: this module never makes a model call.

Run with ``uv run python -m agent.fees_report``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from agent.credit_report import Row, Status, judge, load_departures
from agent.specialists.fees import FEES_SPECIALIST
from agent.tier1 import run_tier1
from banking_tools.manifest import build_manifest, load_tasks
from knowledge.index import KnowledgeIndex


@dataclass(frozen=True)
class FeesReport:
    rows: tuple[Row, ...]

    def count(self, status: Status) -> int:
        return sum(1 for r in self.rows if r.status is status)

    @property
    def bypasses(self) -> int:
        return sum(r.bypasses for r in self.rows)

    @property
    def strict(self) -> int:
        return sum(1 for r in self.rows if r.strict)

    @property
    def clean(self) -> bool:
        """Every task passes or departs for a documented reason, and nothing was written outside a plan."""
        return self.count(Status.FAIL) == 0 and self.bypasses == 0


def fees_task_ids(tasks: Optional[Sequence[Mapping[str, Any]]] = None) -> list[str]:
    """The tasks whose changes are all fee, interest or rewards ones, in the manifest's order."""
    entries = build_manifest(tasks if tasks is not None else load_tasks())
    return [
        e.task_id
        for e in entries
        if e.category == "wholly_one_cluster" and e.clusters == ("fees",) and e.excluded_reason is None
    ]


def run_fees_report(
    index: KnowledgeIndex,
    tasks: Optional[Sequence[Mapping[str, Any]]] = None,
    departures: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> FeesReport:
    chosen = list(tasks if tasks is not None else load_tasks())
    wanted = set(fees_task_ids(chosen))
    known = departures if departures is not None else load_departures()
    return FeesReport(tuple(judge(run_tier1(t, FEES_SPECIALIST, index=index), known) for t in chosen if str(t["id"]) in wanted))


def render(report: FeesReport) -> str:
    lines = [
        f"{len(report.rows)} fee tasks: {report.count(Status.PASS)} pass ({report.strict} strictly), "
        f"{report.count(Status.DEPARTS)} depart from the reference for a documented reason, {report.count(Status.FAIL)} fail; "
        f"{report.bypasses} write(s) outside an approved plan (offline, scripted model, no live call)"
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

    report = run_fees_report(get_index())
    print(render(report))
    return 0 if report.clean else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
