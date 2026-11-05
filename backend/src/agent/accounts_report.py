"""Offline evidence for the accounts specialist: the reference answers replayed through it.

For each of the 19 account tasks (055 to 071, 075 and 076: openings, closings and the transfers that go with
them) a scripted model hands the specialist what the reference says the customer wanted, the specialist plans
and runs it, and the database is compared with the one tau2 gets from the reference. Every task ends in
exactly one of three states, judged the way the credit report judges them:

  PASS     the database matches the reference;
  DEPARTS  it differs, but only in the tables of a documented departure from policy
           (``evals/policy_departures.json``);
  FAIL     anything else.

The 4 tasks that pay a card from checking inside a credit flow (043, 045, 048, 051) belong to the credit
report; the fees and interest tasks (072 to 074, 093 to 097) to Issue #43. The report is a measurement, not a
gate, and there is no live part: this module never makes a model call.

Run with ``uv run python -m agent.accounts_report``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from agent.credit_report import Row, Status, judge, load_departures
from agent.specialists.accounts import ACCOUNTS_SPECIALIST
from agent.tier1 import run_tier1
from banking_tools.manifest import build_manifest, load_tasks
from knowledge.index import KnowledgeIndex


@dataclass(frozen=True)
class AccountsReport:
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


def accounts_task_ids(tasks: Optional[Sequence[Mapping[str, Any]]] = None) -> list[str]:
    """The tasks whose changes are all bank-account ones, in the manifest's order."""
    entries = build_manifest(tasks if tasks is not None else load_tasks())
    return [
        e.task_id
        for e in entries
        if e.category == "wholly_one_cluster" and e.clusters == ("accounts",) and e.excluded_reason is None
    ]


def run_accounts_report(
    index: KnowledgeIndex,
    tasks: Optional[Sequence[Mapping[str, Any]]] = None,
    departures: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> AccountsReport:
    chosen = list(tasks if tasks is not None else load_tasks())
    wanted = set(accounts_task_ids(chosen))
    known = departures if departures is not None else load_departures()
    return AccountsReport(
        tuple(judge(run_tier1(t, ACCOUNTS_SPECIALIST, index=index), known) for t in chosen if str(t["id"]) in wanted)
    )


def render(report: AccountsReport) -> str:
    lines = [
        f"{len(report.rows)} account tasks: {report.count(Status.PASS)} pass ({report.strict} strictly), "
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

    report = run_accounts_report(get_index())
    print(render(report))
    return 0 if report.clean else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
