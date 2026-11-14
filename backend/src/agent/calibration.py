"""Offline report on the decision log: how each risk class's thresholds behave, and where the new gate and the old one differ.

Run: ``uv run python -m agent.calibration [path ...]`` (default: the passive log from the demo). Makes no network or model
call and says how many decisions it rests on. Outcomes: ``completed`` (the request was done), ``escalated`` (handed over).
A threshold sweep per risk class shows, for each proceed line, how many requests would run, how many would be asked about,
and how many of the ones that ran went on to be completed rather than handed over; with too few decisions it says so
instead of recommending anything.
"""

import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

MIN_DECISIONS = 30
SWEEP = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)


def load(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return records


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """The chance a completed request scored higher than an escalated one (None when either side is empty)."""
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def report(records: Sequence[dict[str, Any]]) -> str:
    decisions = [r for r in records if r.get("type") == "decision"]
    outcomes = {(r["turn"], r["intent"]): r["outcome"] for r in records if r.get("type") == "outcome"}
    lines = [f"decisions: {len(decisions)}   with an outcome: {sum((d['turn'], d['intent']) in outcomes for d in decisions)}"]
    if len(decisions) < MIN_DECISIONS:
        lines.append(f"Too few decisions to recommend thresholds (need {MIN_DECISIONS}); the numbers below are only a first look.")
    disagree = sum(bool(d.get("disagrees")) for d in decisions)
    lines.append(f"new gate differs from the old single gate on {disagree} of {len(decisions)} decisions")
    actions = Counter(d["action"] for d in decisions)
    lines.append("actions: " + ", ".join(f"{a} {n}" for a, n in sorted(actions.items())))
    by_risk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in decisions:
        by_risk[d["risk"]].append(d)
    for risk, rows in sorted(by_risk.items()):
        scored = [d for d in rows if d.get("kind_confidence") is not None]
        lines.append(f"\n[{risk}] {len(rows)} decisions, {len(scored)} scored")
        done = [min(d["kind_confidence"], d["scope_confidence"]) for d in scored if outcomes.get((d["turn"], d["intent"])) == "completed"]
        handed = [min(d["kind_confidence"], d["scope_confidence"]) for d in scored if outcomes.get((d["turn"], d["intent"])) == "escalated"]
        value = auroc(done, handed)
        lines.append("  separation (completed vs handed over): " + ("n/a, need both" if value is None else f"{value:.2f}"))
        for line in SWEEP:
            runs = [d for d in scored if min(d["kind_confidence"], d["scope_confidence"]) >= line]
            finished = sum(outcomes.get((d["turn"], d["intent"])) == "completed" for d in runs)
            lines.append(f"  proceed at {line:.2f}: {len(runs)} would run, {finished} of them completed, {len(scored) - len(runs)} asked about")
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    from agent.assembly import DECISION_LOG_PATH

    paths = [Path(a) for a in argv] or [DECISION_LOG_PATH]
    print(report(load(paths)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
