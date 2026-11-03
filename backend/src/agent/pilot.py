"""The debit-card pilot gate: is the framework good enough to build the other specialists on?

The rule (decided in the PRD, not here) is three things at once:
  1. every usable pilot task passes the offline tier-1 replay;
  2. a live run, with a real model deciding each plan, reaches at least 80% semantic
     pass@1 on those tasks;
  3. no write ever took effect outside the plan the customer approved.

Tier 1 is free and always runs. The live part is paid, opt-in and capped (``agent.live``),
so this module never starts one: it reads the saved result of an earlier run
(``agent.live --out``). With no live result the verdict is "not yet", never "go".

Run with ``uv run python -m agent.pilot [--live results.json]``.
"""

import argparse
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from agent.specialists.debit import DEBIT_SPECIALIST
from agent.tier1 import Tier1Report, Tier1Result, render, run_tier1
from banking_tools.manifest import build_manifest, load_tasks
from knowledge.index import KnowledgeIndex

LIVE_BAR = 0.80


class Verdict(StrEnum):
    GO = "GO"
    NO_GO = "NO-GO"
    NOT_YET = "NOT YET"


@dataclass(frozen=True)
class Gate:
    verdict: Verdict
    reasons: tuple[str, ...]
    tier1_passed: int
    tier1_usable: int
    live_pass_rate: Optional[float]
    live_tasks: int
    bypasses: int


def pilot_tasks(tasks: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    entries = {e.task_id: e for e in build_manifest(tasks)}
    return [t for t in tasks if entries[str(t["id"])].pilot]


def run_pilot_tier1(index: KnowledgeIndex, tasks: Optional[Sequence[Mapping[str, Any]]] = None) -> Tier1Report:
    chosen = pilot_tasks(tasks if tasks is not None else load_tasks())
    return Tier1Report(tuple(run_tier1(t, DEBIT_SPECIALIST, index=index) for t in chosen))


def decide(report: Tier1Report, live: Optional[Sequence[Mapping[str, Any]]]) -> Gate:
    """The verdict from the offline report and, if there is one, the saved live results."""
    usable = [r for r in report.results if not r.excluded]
    passed = sum(1 for r in usable if r.passed)
    bypasses = sum(r.bypasses for r in report.results)
    reasons: list[str] = []
    ok = True
    if not usable:
        reasons.append("no usable pilot tasks")
        ok = False
    if passed != len(usable):
        failing = ", ".join(r.task_id for r in usable if not r.passed)
        reasons.append(f"tier 1: {passed}/{len(usable)} passed; failing: {failing}")
        ok = False
    if bypasses:
        reasons.append(f"{bypasses} write(s) took effect outside the approved plan")
        ok = False
    if not ok:
        return Gate(Verdict.NO_GO, tuple(reasons), passed, len(usable), None, 0, bypasses)
    reasons.append(f"tier 1: all {len(usable)} usable tasks pass, zero writes outside the approved plan")
    if not live:
        reasons.append("the live run has not been done: spend is opt-in (python -m agent.live --tasks debit)")
        return Gate(Verdict.NOT_YET, tuple(reasons), passed, len(usable), None, 0, bypasses)
    ran = [r for r in live if not str(r.get("note", "")).startswith("skipped")]
    rate = sum(1 for r in ran if r.get("passed")) / len(ran) if ran else 0.0
    reasons.append(f"live: {sum(1 for r in ran if r.get('passed'))}/{len(ran)} semantic pass@1 = {rate:.0%} (bar {LIVE_BAR:.0%})")
    if len(ran) < len(usable):
        reasons.append(f"live covered {len(ran)} of {len(usable)} usable tasks: the bar is measured on all of them")
        return Gate(Verdict.NOT_YET, tuple(reasons), passed, len(usable), rate, len(ran), bypasses)
    verdict = Verdict.GO if rate >= LIVE_BAR else Verdict.NO_GO
    return Gate(verdict, tuple(reasons), passed, len(usable), rate, len(ran), bypasses)


def render_report(report: Tier1Report, gate: Gate) -> str:
    lines = [render(report), ""]
    lines.append("seconds per task (wall clock, no model): " + ", ".join(f"{r.task_id[-3:]}={r.seconds:.2f}" for r in report.results if not r.excluded))
    lines.append("model tokens: none in tier 1 (scripted); live runs report tokens and latency per task")
    lines += ["", f"VERDICT: {gate.verdict.value}"] + [f"  - {r}" for r in gate.reasons]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - thin wrapper; the parts are tested
    from knowledge.index import get_index

    parser = argparse.ArgumentParser(prog="agent.pilot")
    parser.add_argument("--live", type=Path, default=None, help="results saved by `agent.live --out`")
    args = parser.parse_args(argv)
    live = json.loads(args.live.read_text()) if args.live else None
    report = run_pilot_tier1(get_index())
    print(render_report(report, decide(report, live)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
