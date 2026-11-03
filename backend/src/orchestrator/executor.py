"""Runs a plan wave by wave against whatever is behind each route.

Everything in a wave runs at the same time; the next wave starts when the whole
wave has finished. A route that raises, stalls past the timeout, or returns
something that is not a RouteOutcome becomes a FAILED outcome. Nothing escapes
as an exception, and only the exception's type is recorded (never its message),
because an error can echo the request and the request holds customer data.

Failure policy: an intent whose dependency did not complete (failed, partly
done, or itself skipped) is skipped. Independent intents carry on. Work that
already finished is never undone.

Reconciliation: when a route does not finish (timeout, exception, malformed result,
or a FAILED it returned) the executor asks the route's session what it had already
committed. If any write took effect, the outcome becomes PARTIAL, with a text built
from the audit instead of the route's own words, so the customer is told what really
happened and a retry knows what not to repeat. A route that wrote nothing stays FAILED.
"""

import asyncio
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Optional, Protocol, Self

from pydantic import BaseModel, ConfigDict, model_validator

from orchestrator.context import RouteContext
from orchestrator.plan import Plan
from orchestrator.plan_preview import PlanPreview
from orchestrator.routes import Route, route_for
from orchestrator.trace import Trace, TraceEvent, TraceKind
from orchestrator.types import Intent, Question

# A direct recipe is a few tool calls; a sub-agent loops over a model and tools.
ROUTE_TIMEOUTS: Mapping[Route, float] = {
    Route.DIRECT: 10.0,
    Route.SUB_AGENT: 90.0,
    Route.HAND_OFF: 10.0,
}
ROUTE_TIMEOUT_SECONDS = 30.0  # for a route missing from a custom mapping


class OutcomeStatus(StrEnum):
    DONE = "done"
    # Not finished, and not a failure: the route cannot go on without an answer from the
    # customer. The intent stays open and the questions are asked.
    NEEDS_INFO = "needs_info"
    # Planning only: here is what I would do. Carries the proposal for the customer's yes.
    NEEDS_CONFIRMATION = "needs_confirmation"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class RouteOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    intent_id: str
    status: OutcomeStatus
    # What the customer can be told. For PARTIAL, what was done.
    text: str = ""
    # Why it did not complete. Required unless DONE or NEEDS_INFO.
    error: Optional[str] = None
    # What to ask the customer. Slot names (and wording built by code), never values the
    # customer gave: a PIN must not travel through a question. Only for NEEDS_INFO.
    questions: tuple[Question, ...] = ()
    # The plan to approve. Only for NEEDS_CONFIRMATION.
    proposal: Optional[PlanPreview] = None

    @model_validator(mode="after")
    def _fields_match_status(self) -> Self:
        asking = self.status is OutcomeStatus.NEEDS_INFO
        no_error = (OutcomeStatus.DONE, OutcomeStatus.NEEDS_INFO, OutcomeStatus.NEEDS_CONFIRMATION)
        if self.status in no_error and self.error is not None:
            raise ValueError(f"a {self.status.value} outcome must not carry an error")
        if self.status not in no_error and not self.error:
            raise ValueError(f"a {self.status.value} outcome must say why")
        proposing = self.status is OutcomeStatus.NEEDS_CONFIRMATION
        if proposing and (self.proposal is None or self.proposal.intent_id != self.intent_id):
            raise ValueError("a needs_confirmation outcome must carry a plan for its own intent")
        if not proposing and self.proposal is not None:
            raise ValueError(f"a {self.status.value} outcome must not carry a plan")
        if asking and not self.questions:
            raise ValueError("a needs_info outcome must carry at least one question")
        if not asking and self.questions:
            raise ValueError(f"a {self.status.value} outcome must not carry questions")
        if any(q.intent_id != self.intent_id for q in self.questions):
            raise ValueError("a route may only ask about its own intent")
        return self


class RouteRunner(Protocol):
    async def run(self, intent: Intent, ctx: RouteContext) -> RouteOutcome: ...


ContextFor = Callable[[Intent, float], RouteContext]


class ExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    outcomes: tuple[RouteOutcome, ...]
    trace: Trace


def _failed(intent_id: str, error: str) -> RouteOutcome:
    return RouteOutcome(intent_id=intent_id, status=OutcomeStatus.FAILED, error=error)


def _timeout_for(timeout: float | Mapping[Route, float], route: Route) -> float:
    if isinstance(timeout, Mapping):
        return timeout.get(route, ROUTE_TIMEOUT_SECONDS)
    return timeout


def _reconciled(outcome: RouteOutcome, ctx: RouteContext) -> RouteOutcome:
    """PARTIAL, told from the audit, if a route that did not finish had committed writes."""
    settled = (OutcomeStatus.DONE, OutcomeStatus.NEEDS_INFO, OutcomeStatus.NEEDS_CONFIRMATION)
    if outcome.status in settled or ctx.session is None:
        return outcome
    try:
        steps = tuple(ctx.session.committed_steps())
    except Exception:  # an audit that cannot be read must not hide the original failure
        return outcome
    if not steps:
        return outcome
    return RouteOutcome(
        intent_id=outcome.intent_id,
        status=OutcomeStatus.PARTIAL,
        text="Already done before the problem: " + "; ".join(steps) + ".",
        error=outcome.error or "stopped partway",
    )


async def _run_one(
    intent: Intent,
    wave: int,
    runners: Mapping[Route, RouteRunner],
    timeouts: float | Mapping[Route, float],
    context_for: ContextFor,
    events: list[TraceEvent],
) -> RouteOutcome:
    route = route_for(intent.kind)
    events.append(
        TraceEvent(kind=TraceKind.ROUTED, intent_id=intent.id, detail=route.value, wave=wave)
    )
    runner = runners.get(route)
    timeout = _timeout_for(timeouts, route)
    ctx = context_for(intent, timeout)
    if runner is None:
        outcome = _failed(intent.id, f"no runner for route {route.value}")
    else:
        events.append(TraceEvent(kind=TraceKind.STARTED, intent_id=intent.id, wave=wave))
        try:
            returned = await asyncio.wait_for(runner.run(intent, ctx), timeout)
            if not isinstance(returned, RouteOutcome) or returned.intent_id != intent.id:
                outcome = _failed(intent.id, "route returned a malformed result")
            else:
                outcome = returned
        except TimeoutError:
            outcome = _failed(intent.id, f"timed out after {timeout}s")
        except Exception as exc:
            outcome = _failed(intent.id, f"route failed: {type(exc).__name__}")
        if outcome.status is OutcomeStatus.NEEDS_CONFIRMATION and ctx.mode == "execute":
            outcome = _failed(intent.id, "a route asked for approval after it was given")
        outcome = _reconciled(outcome, ctx)
    if outcome.status is OutcomeStatus.DONE:
        events.append(TraceEvent(kind=TraceKind.FINISHED, intent_id=intent.id, wave=wave))
    elif outcome.status is OutcomeStatus.NEEDS_CONFIRMATION:
        events.append(
            TraceEvent(kind=TraceKind.ASKED, intent_id=intent.id, detail="confirmation", wave=wave)
        )
    elif outcome.status is OutcomeStatus.NEEDS_INFO:
        # Slot names only: the trace never carries what the customer said.
        events.append(
            TraceEvent(
                kind=TraceKind.ASKED,
                intent_id=intent.id,
                detail=",".join(q.slot for q in outcome.questions),
                wave=wave,
            )
        )
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
    timeout: float | Mapping[Route, float] = ROUTE_TIMEOUTS,
    context_for: Optional[ContextFor] = None,
) -> ExecutionResult:
    make_context = context_for or (lambda intent, seconds: RouteContext(intent_id=intent.id))
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
                reason = (
                    f"waiting for an answer about '{incomplete}'"
                    if outcomes[incomplete].status
                    in (OutcomeStatus.NEEDS_INFO, OutcomeStatus.NEEDS_CONFIRMATION)
                    else f"dependency '{incomplete}' did not complete"
                )
                outcomes[intent_id] = RouteOutcome(
                    intent_id=intent_id, status=OutcomeStatus.SKIPPED, error=reason
                )
                events.append(
                    TraceEvent(kind=TraceKind.SKIPPED, intent_id=intent_id, reason=reason, wave=number)
                )
                continue
            runnable.append(intent)
        finished = await asyncio.gather(
            *(_run_one(i, number, runners, timeout, make_context, events) for i in runnable)
        )
        for outcome in finished:
            outcomes[outcome.intent_id] = outcome
    return ExecutionResult(outcomes=tuple(outcomes.values()), trace=Trace(events=tuple(events)))
