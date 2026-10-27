"""One conversation with one customer, turn by turn.

Each turn: read the message, check who is asking, score and decide per intent,
plan what can run together, run it, and answer once. Every choice lands in a
trace, so a test can assert what was decided and why, not just what was said.

What carries from turn to turn lives here: the intents still open, how many
times each has been asked about, which yes was given to which exact details,
whether the caller is verified, how many help menus were shown, and the result
of every message id already handled.

Rules this module is responsible for:
  - A message id handled once is never handled again. The stored result comes
    back, so a double submit cannot run anything twice.
  - Each intent has its own ask counter. After MAX_ASKS questions it escalates
    and the others carry on. Changing an intent's details starts its count over.
    Answering one intent does not reset another's.
  - A yes counts only for what it was asked about. It is bound to the details
    shown when we asked, so a change since makes it stale, and a yes with no
    question pending is ignored.
  - Only intents the customer touched this turn are scored again.
  - A greeting is caught before any model, but never when the message is the
    answer to a question we just asked.
  - Text the customer sends can raise a flag in the trace. It never changes a
    route or a band.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Optional

from pydantic import BaseModel, ConfigDict

from orchestrator.auth import (
    AuthState,
    Directory,
    gate_intents,
    reject_unknown,
    verify,
)
from orchestrator.decision import Action, IntentDecision, decide, fingerprint
from orchestrator.decomposer import Decomposer, Interpretation
from orchestrator.executor import (
    ROUTE_TIMEOUT_SECONDS,
    ExecutionResult,
    OutcomeStatus,
    RouteOutcome,
    RouteRunner,
    execute_plan,
)
from orchestrator.plan import Plan, PlanRejected, plan
from orchestrator.reply import Reply, compose_reply
from orchestrator.routes import Route, route_for
from orchestrator.scoring import SCORER_TIMEOUT_SECONDS, IntentScore, Scorer, score_intents
from orchestrator.trace import Trace, TraceEvent, TraceKind
from orchestrator.types import Decomposition, Intent, IntentKind, Question
from orchestrator.unclear import HelpState, handle_unclear, offer_help, precheck

MAX_ASKS = 3


class TurnResult(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    reply: Reply
    trace: Trace
    decisions: tuple[IntentDecision, ...]
    outcomes: tuple[RouteOutcome, ...]


class _HelpAwareRunner:
    """The help menu is a real tool, not a mock: unclear intents get it directly."""

    def __init__(self, inner: Optional[RouteRunner]) -> None:
        self._inner = inner

    async def run(self, intent: Intent) -> RouteOutcome:
        if intent.kind is IntentKind.UNCLEAR_OR_GREETING:
            return RouteOutcome(
                intent_id=intent.id, status=OutcomeStatus.DONE, text=offer_help()
            )
        if self._inner is None:
            return RouteOutcome(
                intent_id=intent.id, status=OutcomeStatus.FAILED, error="no direct runner"
            )
        return await self._inner.run(intent)


def _view(intents: Sequence[Intent]) -> Decomposition:
    """The given intents as a Decomposition, with dependencies on anything outside
    the list dropped (those intents are finished or gone and are not the scorer's concern)."""
    ids = {i.id for i in intents}
    return Decomposition(
        intents=tuple(
            i.model_copy(update={"depends_on": tuple(d for d in i.depends_on if d in ids)})
            for i in intents
        )
    )


def _decision_events(decision: IntentDecision) -> list[TraceEvent]:
    events = [
        TraceEvent(
            kind=TraceKind.DECIDED,
            intent_id=decision.intent_id,
            detail=decision.action.value,
            reason=decision.reason,
        )
    ]
    if decision.action is Action.ESCALATE:
        events.append(
            TraceEvent(kind=TraceKind.ESCALATED, intent_id=decision.intent_id, reason=decision.reason)
        )
    if decision.action is Action.ASK:
        events.append(
            TraceEvent(
                kind=TraceKind.ASKED,
                intent_id=decision.intent_id,
                detail=",".join(q.slot for q in decision.questions),
            )
        )
    return events


class Orchestrator:
    def __init__(
        self,
        *,
        decomposer: Decomposer,
        scorer: Scorer,
        directory: Directory,
        runners: Mapping[Route, RouteRunner],
        route_timeout: float = ROUTE_TIMEOUT_SECONDS,
        scorer_timeout: float = SCORER_TIMEOUT_SECONDS,
    ) -> None:
        self._decomposer = decomposer
        self._scorer = scorer
        self._directory = directory
        self._runners: Mapping[Route, RouteRunner] = {
            **runners,
            Route.DIRECT: _HelpAwareRunner(runners.get(Route.DIRECT)),
        }
        self._route_timeout = route_timeout
        self._scorer_timeout = scorer_timeout

        self._active: dict[str, Intent] = {}
        self._done: list[str] = []
        self._asks: dict[str, int] = {}
        self._prints: dict[str, str] = {}
        self._scores: dict[str, IntentScore] = {}
        self._pending_confirm: dict[str, str] = {}
        self._confirmed: dict[str, str] = {}
        self._auth = AuthState()
        self._help = HelpState()
        self._awaiting = False
        self._processed: dict[str, TurnResult] = {}
        self._lock = asyncio.Lock()

    @property
    def active_ids(self) -> tuple[str, ...]:
        return tuple(self._active)

    @property
    def done_ids(self) -> tuple[str, ...]:
        return tuple(self._done)

    @property
    def ask_counts(self) -> dict[str, int]:
        return dict(self._asks)

    @property
    def verified(self) -> bool:
        return self._auth.verified

    async def handle_turn(self, message_id: str, message: str) -> TurnResult:
        async with self._lock:
            cached = self._processed.get(message_id)
            if cached is not None:
                return cached
            result = await self._turn(message)
            self._processed[message_id] = result
            return result

    def _forget(self, intent_id: str) -> None:
        for table in (
            self._active,
            self._asks,
            self._prints,
            self._scores,
            self._pending_confirm,
            self._confirmed,
        ):
            table.pop(intent_id, None)

    async def _turn(self, message: str) -> TurnResult:
        events: list[TraceEvent] = []
        if not self._awaiting:
            unclear = precheck(message)
            if unclear is not None:
                return await self._unclear_turn(unclear.intents[0], events)
        try:
            interpretation = await self._decomposer.interpret(
                message, tuple(self._active.values())
            )
        except Exception as exc:
            return self._failed_turn(f"decomposer failed: {type(exc).__name__}", events)
        return await self._real_turn(message, interpretation, events)

    async def _unclear_turn(self, intent: Intent, events: list[TraceEvent]) -> TurnResult:
        decision, self._help = handle_unclear(intent, self._help)
        events.extend(_decision_events(decision))
        outcomes: tuple[RouteOutcome, ...] = ()
        if decision.action is Action.RUN:
            executed = await self._execute(Plan(waves=((intent.id,),), waiting=()), {intent.id: intent})
            outcomes = executed.outcomes
            events.extend(executed.trace.events)
        self._awaiting = False
        reply = compose_reply({intent.id: intent}, outcomes=outcomes, decisions=[decision])
        return TurnResult(
            reply=reply, trace=Trace(events=tuple(events)), decisions=(decision,), outcomes=outcomes
        )

    def _failed_turn(self, reason: str, events: list[TraceEvent]) -> TurnResult:
        synthetic = Intent(id="turn", kind=IntentKind.OTHER, summary="your request")
        decision = IntentDecision(intent_id="turn", action=Action.ESCALATE, reason=reason)
        events.extend(_decision_events(decision))
        self._awaiting = False
        reply = compose_reply({"turn": synthetic}, decisions=[decision])
        return TurnResult(
            reply=reply, trace=Trace(events=tuple(events)), decisions=(decision,), outcomes=()
        )

    async def _execute(self, the_plan: Plan, intents: Mapping[str, Intent]) -> ExecutionResult:
        return await execute_plan(the_plan, intents, self._runners, timeout=self._route_timeout)

    async def _verify(self, claims: Mapping[str, str], events: list[TraceEvent]) -> None:
        try:
            record = await self._directory.find(claims)
        except Exception as exc:
            # Not the customer's fault, so no failed attempt is counted.
            events.append(
                TraceEvent(
                    kind=TraceKind.FAILED, reason=f"directory lookup failed: {type(exc).__name__}"
                )
            )
            return
        if record is None:
            self._auth = reject_unknown(self._auth, claims)
        else:
            self._auth = verify(self._auth, record, claims)

    async def _real_turn(
        self, message: str, interpretation: Interpretation, events: list[TraceEvent]
    ) -> TurnResult:
        decomposition = interpretation.decomposition
        if any(i.kind is not IntentKind.UNCLEAR_OR_GREETING for i in decomposition.intents):
            self._help = HelpState()
        for flag in decomposition.flags:
            events.append(TraceEvent(kind=TraceKind.FLAGGED, detail=flag.value))

        for intent_id in interpretation.declined:
            if intent_id in self._active:
                self._forget(intent_id)
                events.append(
                    TraceEvent(
                        kind=TraceKind.SKIPPED, intent_id=intent_id, reason="withdrawn by the customer"
                    )
                )

        touched: set[str] = set()
        for intent in decomposition.intents:
            if intent.id in self._done:
                continue
            current = fingerprint(intent)
            earlier = self._prints.get(intent.id)
            if earlier is not None and earlier != current:
                self._asks[intent.id] = 0
                events.append(
                    TraceEvent(
                        kind=TraceKind.DECIDED,
                        intent_id=intent.id,
                        detail="superseded",
                        reason="details changed",
                    )
                )
            self._prints[intent.id] = current
            self._active[intent.id] = intent
            touched.add(intent.id)

        for intent_id in interpretation.confirmed:
            shown = self._pending_confirm.pop(intent_id, None)
            if shown is not None and intent_id in self._active:
                self._confirmed[intent_id] = shown
                touched.add(intent_id)
                events.append(
                    TraceEvent(kind=TraceKind.DECIDED, intent_id=intent_id, detail="confirmed")
                )

        if interpretation.claims:
            await self._verify(interpretation.claims, events)

        decisions = await self._decide_all(message, touched, events)
        run_intents = [self._active[i] for i, d in decisions.items() if d.action is Action.RUN]
        the_plan = plan(
            run_intents, done=frozenset(self._done), conflicts=decomposition.conflicts
        )
        if isinstance(the_plan, PlanRejected):
            for intent in run_intents:
                decisions[intent.id] = IntentDecision(
                    intent_id=intent.id, action=Action.ESCALATE, reason=the_plan.reason
                )
            run_intents = []
            the_plan = Plan(waves=(), waiting=())

        for decision in decisions.values():
            events.extend(_decision_events(decision))

        view = dict(self._active)
        executed = await self._execute(the_plan, {i.id: i for i in run_intents})
        events.extend(executed.trace.events)

        for outcome in executed.outcomes:
            if outcome.status is OutcomeStatus.DONE:
                self._done.append(outcome.intent_id)
            self._forget(outcome.intent_id)
        for decision in decisions.values():
            if decision.action is Action.ESCALATE:
                self._forget(decision.intent_id)

        ordered = tuple(decisions.values())
        self._awaiting = any(d.action is Action.ASK for d in ordered)
        reply = compose_reply(
            view, outcomes=executed.outcomes, decisions=ordered, waiting=the_plan.waiting
        )
        return TurnResult(
            reply=reply,
            trace=Trace(events=tuple(events)),
            decisions=ordered,
            outcomes=executed.outcomes,
        )

    async def _decide_all(
        self, message: str, touched: set[str], events: list[TraceEvent]
    ) -> dict[str, IntentDecision]:
        active = list(self._active.values())
        gate = gate_intents(self._auth, active)
        blocked: dict[str, IntentDecision] = {}
        for intent_id in gate.blocked:
            if gate.hand_off is not None:
                blocked[intent_id] = IntentDecision(
                    intent_id=intent_id, action=Action.ESCALATE, reason=gate.hand_off.reason
                )
            else:
                questions: tuple[Question, ...] = tuple(
                    q for q in gate.questions if q.intent_id == intent_id
                )
                blocked[intent_id] = IntentDecision(
                    intent_id=intent_id,
                    action=Action.ASK,
                    reason="needs identity verification",
                    questions=questions,
                )

        allowed = [self._active[i] for i in gate.allowed]
        scoreable = [
            i
            for i in allowed
            if i.kind is not IntentKind.UNCLEAR_OR_GREETING
            and route_for(i.kind) is not Route.HAND_OFF
        ]
        to_score = [i for i in scoreable if i.id in touched or i.id not in self._scores]
        if to_score:
            for score in await score_intents(
                self._scorer, message, _view(to_score), timeout=self._scorer_timeout
            ):
                self._scores[score.intent_id] = score
                events.append(
                    TraceEvent(
                        kind=TraceKind.SCORED,
                        intent_id=score.intent_id,
                        detail=score.error or f"{score.confidence:.2f}",
                    )
                )

        decided: dict[str, IntentDecision] = dict(blocked)
        for intent in allowed:
            if intent.kind is IntentKind.UNCLEAR_OR_GREETING:
                decision, self._help = handle_unclear(intent, self._help)
            elif route_for(intent.kind) is Route.HAND_OFF:
                decision = decide(intent, IntentScore.unscored(intent.id, "not scored"), {})
            else:
                decision = decide(intent, self._scores[intent.id], self._confirmed)
            decided[intent.id] = decision

        result: dict[str, IntentDecision] = {}
        for intent in active:
            decision = decided[intent.id]
            if decision.action is Action.ASK:
                if self._asks.get(intent.id, 0) >= MAX_ASKS:
                    decision = IntentDecision(
                        intent_id=intent.id,
                        action=Action.ESCALATE,
                        reason=f"still not enough information after {MAX_ASKS} questions",
                    )
                else:
                    self._asks[intent.id] = self._asks.get(intent.id, 0) + 1
                    if decision.fingerprint is not None:
                        self._pending_confirm[intent.id] = decision.fingerprint
            result[intent.id] = decision
        return result
