"""Offline evidence for the credit-card specialist: the reference answers replayed through it.

For each of the 18 credit tasks (the 14 wholly-credit tasks and the 4 that pay a card from checking
first) a scripted model hands the specialist what the reference says the customer wanted, the
specialist plans and runs it, and the database is compared with the one tau2 gets from the
reference. Every task ends in exactly one of three states:

  PASS     the database matches the reference;
  DEPARTS  it differs, but only in the tables of a documented departure from policy
           (``evals/policy_departures.json``), which is where the policy and the reference disagree;
  FAIL     anything else.

The report is a measurement, not a gate. There is no live part: this module never makes a model call.

Run with ``uv run python -m agent.credit_report``.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional

from agent.specialists.credit import CREDIT_SPECIALIST
from agent.tier1 import Tier1Result, run_tier1
from banking_tools.manifest import MANIFEST_PATH, build_manifest, load_tasks
from knowledge.index import KnowledgeIndex

DEPARTURES_PATH = MANIFEST_PATH.parent / "policy_departures.json"


class Status(StrEnum):
    PASS = "PASS"
    DEPARTS = "DEPARTS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Row:
    task_id: str
    status: Status
    strict: bool
    semantic: bool
    segments: int
    model_turns: int
    seconds: float
    bypasses: int
    note: str


@dataclass(frozen=True)
class CreditReport:
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


def credit_task_ids(tasks: Optional[Sequence[Mapping[str, Any]]] = None) -> list[str]:
    """Tasks whose changes are credit-card ones, with or without a payment from checking, and no debit."""
    entries = build_manifest(tasks if tasks is not None else load_tasks())
    return [
        e.task_id
        for e in entries
        if e.category in ("wholly_one_cluster", "spans_clusters")
        and "credit" in e.clusters
        and "debit" not in e.clusters
        and e.excluded_reason is None
    ]


def load_departures(path: Path = DEPARTURES_PATH) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = json.loads(path.read_text())
    return data


def judge(result: Tier1Result, departures: Mapping[str, Mapping[str, Any]]) -> Row:
    tables = {m.split("/", 1)[0] for m in result.mismatches}

    def row(status: Status, note: str) -> Row:
        return Row(
            result.task_id, status, result.strict_equal, result.semantic_equal, len(result.segments),
            result.model_turns, result.seconds, result.bypasses, note,
        )

    if result.error or result.excluded:
        return row(Status.FAIL, result.error or result.excluded or "")
    if result.passed:
        return row(Status.PASS, "")
    allowed = departures.get(result.task_id)
    if allowed is not None and result.mismatches and tables <= set(allowed["tables"]):
        return row(Status.DEPARTS, str(allowed["reason"]))
    return row(Status.FAIL, "; ".join(result.mismatches[:3]))


def run_credit_report(
    index: KnowledgeIndex,
    tasks: Optional[Sequence[Mapping[str, Any]]] = None,
    departures: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> CreditReport:
    chosen = list(tasks if tasks is not None else load_tasks())
    wanted = set(credit_task_ids(chosen))
    known = departures if departures is not None else load_departures()
    return CreditReport(tuple(judge(run_tier1(t, CREDIT_SPECIALIST, index=index), known) for t in chosen if str(t["id"]) in wanted))


def render(report: CreditReport) -> str:
    lines = [
        f"{len(report.rows)} credit tasks: {report.count(Status.PASS)} pass, {report.count(Status.DEPARTS)} depart from the "
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

    report = run_credit_report(get_index())
    print(render(report))
    return 0 if report.clean else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
