"""A record of every decision, so the thresholds can be tuned from what happened instead of guessed.

One JSON line per decision: the kind, its risk class, what the scorer said (numbers and the type of any failure, never
the message), what the decision was, and what the old one-size gate would have done ("shadow": computed beside the live
decision and never acted on). A later line records the outcome of the request. No message text, slot value, id, PIN or
card number is written; reasons are code-built strings and still pass through ``redact``.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Protocol

from orchestrator.bands import ASK_AT, PROCEED_AT, Risk
from orchestrator.decision import Action, IntentDecision, risk_of
from orchestrator.sanitize import redact
from orchestrator.scoring import IntentScore
from orchestrator.types import Intent


class DecisionSink(Protocol):
    def write(self, record: Mapping[str, Any]) -> None: ...


class MemoryLog:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: Mapping[str, Any]) -> None:
        self.records.append(dict(record))


class JsonlLog:
    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, record: Mapping[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def legacy_action(intent: Intent, score: IntentScore) -> str:
    """What the single 0.85/0.65 gate did before risk classes: the shadow the new gate is compared with."""
    if score.error is not None:
        return Action.ESCALATE.value
    understood = min(score.kind_confidence, score.scope_confidence)
    if understood < ASK_AT:
        return Action.ESCALATE.value
    if intent.missing_slots:
        return Action.ASK.value
    if score.confidence < ASK_AT:
        return Action.ESCALATE.value
    if score.confidence < PROCEED_AT:
        return Action.ASK.value
    return Action.RUN.value


def _error_kind(error: Optional[str]) -> Optional[str]:
    # "scorer failed: ValueError" -> keep only the part after the colon (a type name), or the fixed timeout wording.
    if error is None:
        return None
    return error.split(":")[-1].strip()[:60]


class DecisionLog:
    def __init__(self, sink: DecisionSink) -> None:
        self._sink = sink
        self.turn = 0
        self.disagreements = 0
        self.decisions = 0

    def decision(
        self,
        intent: Intent,
        score: Optional[IntentScore],
        decision: IntentDecision,
        *,
        asks_so_far: int,
        signals: Optional[Mapping[str, float]] = None,
    ) -> None:
        shadow = legacy_action(intent, score) if score is not None else None
        self.decisions += 1
        disagree = shadow is not None and shadow != decision.action.value
        self.disagreements += disagree
        risk: Risk = risk_of(intent.kind)
        self._sink.write(
            {
                "type": "decision",
                "turn": self.turn,
                "intent": intent.id,
                "kind": intent.kind.value,
                "risk": risk.value,
                "kind_confidence": score.kind_confidence if score else None,
                "slots_confidence": score.slots_confidence if score else None,
                "scope_confidence": score.scope_confidence if score else None,
                "scorer_error": _error_kind(score.error) if score else None,
                "action": decision.action.value,
                "reason": redact(decision.reason),
                "asks_so_far": asks_so_far,
                "shadow_action": shadow,
                "disagrees": disagree,
                "signals": dict(signals or {}),
            }
        )

    def outcome(self, intent_id: str, kind: str, outcome: str) -> None:
        self._sink.write({"type": "outcome", "turn": self.turn, "intent": intent_id, "kind": kind, "outcome": outcome})
