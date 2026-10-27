"""An ordered record of every choice the orchestrator makes in one turn.

Scenario tests assert on the trace rather than only the final reply, so which
intents ran, in what order, and why one was skipped or escalated is checkable.
"""

from enum import StrEnum
from typing import Optional, Self

from pydantic import BaseModel, ConfigDict, model_validator


class TraceKind(StrEnum):
    SCORED = "scored"
    DECIDED = "decided"
    ROUTED = "routed"
    STARTED = "started"
    FINISHED = "finished"
    FAILED = "failed"
    SKIPPED = "skipped"
    ASKED = "asked"
    ESCALATED = "escalated"


_NEEDS_A_REASON = frozenset({TraceKind.FAILED, TraceKind.SKIPPED, TraceKind.ESCALATED})


class TraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    kind: TraceKind
    # None when the event is about the whole turn rather than one intent.
    intent_id: Optional[str] = None
    detail: str = ""
    reason: Optional[str] = None

    @model_validator(mode="after")
    def _failures_say_why(self) -> Self:
        if self.kind in _NEEDS_A_REASON and not self.reason:
            raise ValueError(f"a {self.kind.value} event must carry a reason")
        return self


class Trace(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    events: tuple[TraceEvent, ...] = ()
