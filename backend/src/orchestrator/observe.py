"""See where a turn went wrong, for testing and the demo.

Three parts, none of which carries what the customer or the bank said:

* ``note`` is a one-line hook the code calls where something decisive happens (a workflow decided, an agent run ended,
  a model call failed). It does nothing unless a turn is being recorded, and it redacts and truncates what it is given.
* ``TurnRecord`` is one turn as a flat record: what was decided and why, what ran, how it ended, how long it took, and a
  ``failure_class`` worked out in code (never by a model) so failures can be counted and found.
* ``Recorder`` keeps the records per conversation (for the demo's debug panel) and appends them to a git-ignored file
  (for ``python -m orchestrator.observe report`` and ``replay``).

Slot names, kinds, ids of requests, decision reasons, exception types and counts are recorded. Message text, slot values,
account ids and amounts are not.
"""

import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Optional

from orchestrator.decision import Action
from orchestrator.executor import OutcomeStatus
from orchestrator.sanitize import redact

TURNS_PATH = Path(__file__).resolve().parents[2] / "evals" / "turns.jsonl"
_MAX_TEXT = 160

_collector: ContextVar[Optional[list[dict[str, Any]]]] = ContextVar("turn_notes", default=None)


def begin() -> list[dict[str, Any]]:
    """Start collecting notes for one turn; the returned list fills as the turn runs."""
    notes: list[dict[str, Any]] = []
    _collector.set(notes)
    return notes


def note(kind: str, **fields: Any) -> None:
    """Record something decisive. Strings are redacted and cut short; anything else must be a number, bool or list of them."""
    notes = _collector.get()
    if notes is None:
        return
    clean: dict[str, Any] = {"kind": kind}
    for key, value in fields.items():
        if isinstance(value, str):
            clean[key] = redact(value)[:_MAX_TEXT]
        elif isinstance(value, (bool, int, float)) or value is None:
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            clean[key] = [redact(v)[:_MAX_TEXT] if isinstance(v, str) else v for v in value][:20]
    notes.append(clean)


def failure_class(
    decisions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    notes: Sequence[Mapping[str, Any]],
    intents: Sequence[str],
    previous_asks: Mapping[str, str],
    asked: Mapping[str, str],
) -> Optional[str]:
    """The one label that says what went wrong this turn, or None when nothing did. Worked out in code, most severe first."""
    if "unprocessed" in intents or any(n["kind"] == "decomposer_failed" for n in notes):
        return "decomposer_failed"
    reasons = [str(d.get("reason", "")) for d in decisions if d.get("action") == Action.ESCALATE.value]
    if any("still not enough information" in r for r in reasons):
        return "ask_limit_escalation"
    if any(n["kind"] == "agent" and n.get("stop") in ("model_failed", "deadline", "steps_exhausted", "tokens_exhausted", "too_many_errors") for n in notes):
        return "agent_stopped"
    if any(o.get("status") in (OutcomeStatus.FAILED.value,) for o in outcomes if "status" in o):
        return "route_failed"
    if any("went stale" in str(d.get("reason", "")) for d in decisions):
        return "plan_went_stale"
    if any(n["kind"] == "workflow" and n.get("outcome") == "Refuse" for n in notes):
        return "workflow_refused"
    if any(intent in previous_asks and previous_asks[intent] == slot for intent, slot in asked.items()):
        return "repeated_ask"
    if reasons:
        return "escalated"
    return None


class Recorder:
    """The records of every conversation, in memory and (optionally) on disk."""

    def __init__(self, path: Optional[Path] = TURNS_PATH) -> None:
        self._path = path
        self._records: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        self._asked: dict[str, dict[str, str]] = {}

    def record(self, conversation_id: str, message_id: str, result: Any, notes: Sequence[Mapping[str, Any]], started: float, intents: Sequence[str]) -> dict[str, Any]:
        decisions = [
            {
                "intent": d.intent_id,
                "action": d.action.value,
                "reason": redact(d.reason)[:_MAX_TEXT],
                "asked": [q.slot for q in d.questions],
            }
            for d in result.decisions
        ]
        events = [
            {"kind": e.kind.value, "intent": e.intent_id, "detail": redact(e.detail)[:_MAX_TEXT], "reason": redact(e.reason or "")[:_MAX_TEXT], "wave": e.wave}
            for e in result.trace.events
        ]
        outcomes = [{"intent": o.intent_id, "status": o.status.value, "error": (o.error or "")[:_MAX_TEXT]} for o in result.outcomes]
        asked = {d["intent"]: d["asked"][0] for d in decisions if d["action"] == Action.ASK.value and d["asked"]}
        label = failure_class(decisions, outcomes, notes, intents, self._asked.get(conversation_id, {}), asked)
        self._asked[conversation_id] = asked
        rows = self._records[conversation_id]
        record: dict[str, Any] = {
            "conversation": conversation_id,
            "message": message_id,
            "turn": len(rows) + 1,
            "at": round(time.time(), 3),
            "ms": round((time.monotonic() - started) * 1000),
            "reply_kind": _reply_kind(result),
            # What the chat was actually sent for each question: the slot and its kind ("yes_no" gets Yes and No buttons).
            "reply_asks": [{"slot": q.slot, "kind": q.kind} for q in result.reply.questions],
            "failure_class": label,
            "intents": list(intents),
            "decisions": decisions,
            "outcomes": outcomes,
            "events": events,
            "notes": list(notes),
        }
        rows.append(record)
        if self._path is not None:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            except OSError:
                pass  # observing must never break a turn
        return record

    def conversation(self, conversation_id: str) -> list[dict[str, Any]]:
        return list(self._records.get(conversation_id, ()))

    def summary(self) -> dict[str, Any]:
        return summarise([r for rows in self._records.values() for r in rows])


def _reply_kind(result: Any) -> str:
    text = result.reply.text
    if result.reply.escalated_intent_ids:
        return "escalated"
    if result.reply.failed_intent_ids:
        return "failed"
    if "Here is exactly what I will do" in text:
        return "plan"
    if text.startswith("Done:"):
        return "done"
    if result.reply.questions:
        return "ask"
    return "message"


def summarise(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classes = Counter(r["failure_class"] for r in records if r.get("failure_class"))
    slow = sorted((r for r in records if r.get("ms")), key=lambda r: -r["ms"])[:3]
    return {
        "turns": len(records),
        "failures": dict(classes.most_common()),
        "slowest_ms": [{"conversation": r["conversation"][:8], "turn": r["turn"], "ms": r["ms"]} for r in slow],
    }


def render_turn(record: Mapping[str, Any]) -> str:
    lines = [f"turn {record['turn']}  {record['reply_kind']}  {record['ms']} ms" + (f"  FAILURE: {record['failure_class']}" if record.get("failure_class") else "")]
    for d in record["decisions"]:
        lines.append(f"  decided {d['intent']}: {d['action']} - {d['reason']}" + (f" (asked {', '.join(d['asked'])})" if d["asked"] else ""))
    if record.get("reply_asks"):
        lines.append("  sent to chat: " + ", ".join(f"{a['slot']}[{a['kind'] or 'plain'}]" for a in record["reply_asks"]))
    for n in record["notes"]:
        lines.append("  " + ", ".join(f"{k}={v}" for k, v in n.items()))
    for o in record["outcomes"]:
        lines.append(f"  outcome {o['intent']}: {o['status']}" + (f" ({o['error']})" if o["error"] else ""))
    return "\n".join(lines)


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: Sequence[str]) -> int:
    """``report`` counts failures over every recorded turn; ``replay <conversation id prefix>`` prints one conversation."""
    command = argv[0] if argv else "report"
    records = _load(TURNS_PATH)
    if command == "replay" and len(argv) > 1:
        chosen = [r for r in records if r["conversation"].startswith(argv[1])]
        print("\n".join(render_turn(r) for r in chosen) or "no such conversation")
        return 0 if chosen else 1
    summary = summarise(records)
    print(f"{summary['turns']} turns recorded in {TURNS_PATH.name}")
    for label, count in summary["failures"].items():
        print(f"  {count:3d}  {label}")
    for row in summary["slowest_ms"]:
        print(f"  slowest: conversation {row['conversation']} turn {row['turn']} took {row['ms']} ms")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin wrapper; the parts are tested
    sys.exit(main(sys.argv[1:]))
