"""Runs a plan wave by wave against whatever is behind each route.

Everything in a wave runs at the same time; the next wave starts when the whole
wave has finished. A route that raises, stalls past the timeout, or returns
something that is not a RouteOutcome becomes a FAILED outcome. Nothing escapes
as an exception, and only the exception's type is recorded (never its message),
because an error can echo the request and the request holds customer data.

Failure policy: an intent whose dependency did not complete (failed, partly
done, or itself skipped) is skipped. Independent intents carry on. Work that
already finished is never undone.
"""

import asyncio
from collections.abc import Mapping
from enum import StrEnum
from typing import Optional, Protocol, Self

from pydantic import BaseModel, ConfigDict, model_validator

from orchestrator.plan import Plan
from orchestrator.routes import Route, route_for
from orchestrator.trace import Trace, TraceEvent, TraceKind
from orchestrator.types import Intent

ROUTE_TIMEOUT_SECONDS = 30.0


class OutcomeStatus(StrEnum):
    DONE = "done"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class RouteOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    intent_id: str
    status: OutcomeStatus
    # What the customer can be told. For PARTIAL, what was done.
    text: str = ""
    # Why it did not complete. Required unless DONE.
    error: Optional[str] = None

    @model_validator(mode="after")
    def _error_matches_status(self) -> Self:
        if self.status is OutcomeStatus.DONE and self.error is not None:
            raise ValueError("a done outcome must not carry an error")
        if self.status is not OutcomeStatus.DONE and not self.error:
            raise ValueError(f"a {self.status.value} outcome must say why")
        return self


class RouteRunner(Protocol):
    async def run(self, intent: Intent) -> RouteOutcome: ...


class ExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    outcomes: tuple[RouteOutcome, ...]
    trace: Trace


def _failed(intent_id: str, error: str) -> RouteOutcome:
    return RouteOutcome(intent_id=intent_id, status=OutcomeStatus.FAILED, error=error)


async def _run_one(
    intent: Intent,
    wave: int,
    runners: Mapping[Route, RouteRunner],
    timeout: float,
    events: list[TraceEvent],
) -> RouteOutcome:
    route = route_for(intent.kind)
    events.append(
        TraceEvent(kind=TraceKind.ROUTED, intent_id=intent.id, detail=route.value, wave=wave)
    )
    runner = runners.get(route)
    if runner is None:
        outcome = _failed(intent.id, f"no runner for route {route.value}")
    else:
        events.append(TraceEvent(kind=TraceKind.STARTED, intent_id=intent.id, wave=wave))
        try:
            returned = await asyncio.wait_for(runner.run(intent), timeout)
            if not isinstance(returned, RouteOutcome) or returned.intent_id != intent.id:
                outcome = _failed(intent.id, "route returned a malformed result")
            else:
                outcome = returned
        except TimeoutError:
            outcome = _failed(intent.id, f"timed out after {timeout}s")
        except Exception as exc:
            outcome = _failed(intent.id, f"route failed: {type(exc).__name__}")
    if outcome.status is OutcomeStatus.DONE:
        events.append(TraceEvent(kind=TraceKind.FINISHED, intent_id=intent.id, wave=wave))
    else:
        events.append(
            TraceEvent(
                kind=TraceKind.FAILED, intent_id=intent.id, reason=outcome.error, wave=wave
            )
        )
    return outcome


async def execute_plan(
    plan: Plan,
    intents: Mapping[str, Intent],
    runners: Mapping[Route, RouteRunner],
    *,
    timeout: float = ROUTE_TIMEOUT_SECONDS,
) -> ExecutionResult:
    events: list[TraceEvent] = []
    outcomes: dict[str, RouteOutcome] = {}
    for number, wave in enumerate(plan.waves, start=1):
        runnable: list[Intent] = []
        for intent_id in wave:
            intent = intents.get(intent_id)
            if intent is None:
                outcomes[intent_id] = _failed(intent_id, "unknown intent")
                events.append(
                    TraceEvent(
                        kind=TraceKind.FAILED, intent_id=intent_id, reason="unknown intent", wave=number
                    )
                )
                continue
            incomplete = next(
                (
                    d
                    for d in intent.depends_on
                    if d in outcomes and outcomes[d].status is not OutcomeStatus.DONE
                ),
                None,
            )
            if incomplete is not None:
                reason = f"dependency '{incomplete}' did not complete"
                outcomes[intent_id] = RouteOutcome(
                    intent_id=intent_id, status=OutcomeStatus.SKIPPED, error=reason
                )
                events.append(
                    TraceEvent(kind=TraceKind.SKIPPED, intent_id=intent_id, reason=reason, wave=number)
                )
                continue
            runnable.append(intent)
        finished = await asyncio.gather(
            *(_run_one(i, number, runners, timeout, events) for i in runnable)
        )
        for outcome in finished:
            outcomes[outcome.intent_id] = outcome
    return ExecutionResult(outcomes=tuple(outcomes.values()), trace=Trace(events=tuple(events)))
