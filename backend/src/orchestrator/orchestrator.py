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
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Optional

from pydantic import BaseModel, ConfigDict

from banking_tools.types import CompletedStep
from orchestrator.auth import (
    IDENTITY_SLOT,
    AuthState,
    Directory,
    gate_intents,
    reject_unknown,
    verify,
)
from orchestrator.containment import Containment
from orchestrator.context import AgentSession, Mode, RouteContext, SessionRequest
from orchestrator.llm_decomposer import is_affirmation
from orchestrator.decision import CLARIFY_SLOT, CONFIRM_SLOT, HUMAN_AFTER_ASKS, Action, IntentDecision, decide, fingerprint
from orchestrator.decomposer import Decomposer, Interpretation
from orchestrator.executor import (
    ROUTE_TIMEOUTS,
    ExecutionResult,
    OutcomeStatus,
    RouteOutcome,
    RouteRunner,
    execute_plan,
)
from orchestrator.plan import READ_ONLY_KINDS, Plan, PlanRejected, Waiting, closes_wait_for_openings, plan
from orchestrator.plan_preview import PlanPreview
from orchestrator.reply import Reply, compose_reply
from orchestrator.routes import Route, route_for
from orchestrator.scoring import SCORER_TIMEOUT_SECONDS, IntentScore, Scorer, score_intents
from orchestrator.trace import Trace, TraceEvent, TraceKind
from orchestrator.types import Decomposition, Intent, IntentKind, Question
from orchestrator.unclear import HelpState, handle_unclear, offer_help, precheck

MAX_ASKS = 3
# The recent conversation kept for the models: this many messages (customer and agent alternate), each cut short.
HISTORY_MESSAGES = 8
HISTORY_CHARS = 600
_NO = re.compile(r"^\W*(?:no|nope|nah)\b")
_WITHDRAWS = re.compile(r"never ?mind|forget it|cancel|stop|no longer|don'?t need|drop it|leave it")
# What the decomposer makes of an answer it did not recognise as one; never a request to change anything.
_ANSWER_NOISE = READ_ONLY_KINDS | {IntentKind.OTHER, IntentKind.UNCLEAR_OR_GREETING, IntentKind.GENERAL_QUESTION}
# Details that are secrets: an answer to a question about one is captured even without a cue word.
SECRET_SLOTS = frozenset({"pin", "new_pin", "current_pin", "cvv"})


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

    async def run(self, intent: Intent, ctx: Optional[RouteContext] = None) -> RouteOutcome:
        if intent.kind is IntentKind.UNCLEAR_OR_GREETING:
            return RouteOutcome(
                intent_id=intent.id, status=OutcomeStatus.DONE, text=offer_help()
            )
        if self._inner is None:
            return RouteOutcome(
                intent_id=intent.id, status=OutcomeStatus.FAILED, error="no direct runner"
            )
        return await self._inner.run(intent, ctx or RouteContext(intent_id=intent.id))


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
        route_timeout: Optional[float] = None,
        route_timeouts: Optional[Mapping[Route, float]] = None,
        session_for: Optional[Callable[[SessionRequest], AgentSession]] = None,
        capture_secrets: Optional[Callable[..., str]] = None,
        scorer_timeout: float = SCORER_TIMEOUT_SECONDS,
        containment: Optional[Containment] = None,
    ) -> None:
        self._decomposer = decomposer
        self._scorer = scorer
        self._directory = directory
        self._runners: Mapping[Route, RouteRunner] = {
            **runners,
            Route.DIRECT: _HelpAwareRunner(runners.get(Route.DIRECT)),
        }
        # One number applies to every route (handy in tests); otherwise per route.
        self._route_timeouts: float | Mapping[Route, float] = (
            route_timeout if route_timeout is not None else (route_timeouts or ROUTE_TIMEOUTS)
        )
        self._session_for = session_for
        # Swaps a PIN or CVV in the customer's words for a reference, before anything else
        # (a greeting check, the decomposer, the scorer) sees the message.
        self._capture_secrets = capture_secrets
        # The details we asked the customer for in the last reply.
        self._last_asked: frozenset[str] = frozenset()
        # Intents that stopped partway: what they had committed, and the intent itself,
        # kept so "try again" finds it. They are not run again until the customer says so.
        self._progress: dict[str, tuple[CompletedStep, ...]] = {}
        self._resumable: dict[str, Intent] = {}
        self._sessions_this_run: dict[str, AgentSession] = {}
        # Routes that propose a plan (a runner with ``proposes_plans = True``) first work out
        # what they would do. The customer's yes then binds to that plan, not to the intent.
        self._proposals: dict[str, PlanPreview] = {}
        self._approved: dict[str, PlanPreview] = {}
        self._mode: Mode = "execute"
        self._scorer_timeout = scorer_timeout
        # Freezes made before a lost-wallet plan is worked out: (the freeze intent, what happened), this turn only.
        self._containment = containment
        self._contained: list[tuple[Intent, RouteOutcome]] = []
        # What the customer wrote while a lost-wallet report was open: containment reads the cards named in it.
        self._report_text: dict[str, str] = {}
        # Requests for a person made while card protection was open: how many, and the last one, still owed.
        self._human_asks = 0
        self._deferred_human: Optional[Intent] = None

        self._active: dict[str, Intent] = {}
        self._done: list[str] = []
        self._asks: dict[str, int] = {}
        # Questions a route asked mid-run. Counted apart from _asks: a changed detail
        # (the answer to the question) restarts _asks, and must not restart this, or a
        # route that keeps asking would never be stopped.
        self._runner_asks: dict[str, int] = {}
        self._last_route_slot: dict[str, str] = {}
        # What a route asked and the customer has not answered yet. Slot names and
        # code-built wording only. Re-asked, not re-run, until the customer touches the intent.
        self._pending_route_questions: dict[str, tuple[Question, ...]] = {}
        self._prints: dict[str, str] = {}
        self._scores: dict[str, IntentScore] = {}
        self._transcript: list[tuple[str, str]] = []
        self._history_now: tuple[tuple[str, str], ...] = ()
        self._latest = ""
        self._said: dict[str, str] = {}
        self._scored_kind: dict[str, IntentKind] = {}
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
    def runner_ask_counts(self) -> dict[str, int]:
        return dict(self._runner_asks)

    @property
    def resumable_ids(self) -> tuple[str, ...]:
        return tuple(self._resumable)

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
            self._history_now = tuple(self._transcript)
            result = await self._turn(message)
            self._remember(self._latest or message, result.reply.text)
            self._processed[message_id] = result
            return result

    def _remember(self, said: str, replied: str) -> None:
        """Keep the last few turns, with the customer's words as they were after secrets were hidden."""
        self._transcript.append(("customer", said[:HISTORY_CHARS]))
        self._transcript.append(("agent", replied[:HISTORY_CHARS]))
        del self._transcript[:-HISTORY_MESSAGES]

    def _context_for(self, intent: Intent, timeout: float) -> RouteContext:
        """What a runner is given. The caller's identity comes from AuthState alone."""
        progress = self._progress.get(intent.id, ())
        approved = self._approved.get(intent.id) if self._mode == "execute" else None
        session = (
            self._session_for(
                SessionRequest(intent.id, progress, mode=self._mode, approved=approved)
            )
            if self._session_for
            else None
        )
        if session is not None:
            self._sessions_this_run[intent.id] = session
        return RouteContext(
            intent_id=intent.id,
            verified_user_id=self._auth.verified_user_id,
            session=session,
            deadline=asyncio.get_running_loop().time() + timeout,
            progress=progress,
            mode=self._mode,
            approved=approved,
            recent=self._history_now,
        )

    def _forget(self, intent_id: str) -> None:
        for table in (
            self._active,
            self._asks,
            self._runner_asks,
            self._pending_route_questions,
            self._proposals,
            self._approved,
            self._prints,
            self._scores,
            self._pending_confirm,
            self._confirmed,
            self._report_text,
            self._scored_kind,
        ):
            table.pop(intent_id, None)

    async def _turn(self, message: str) -> TurnResult:
        if self._capture_secrets is not None:
            # Just asked for a PIN or a security code? Then bare digits in the answer are one.
            message = self._capture_secrets(message, expecting_secret=bool(self._last_asked & SECRET_SLOTS))
        self._latest = message
        events: list[TraceEvent] = []
        if not self._awaiting:
            unclear = precheck(message)
            if unclear is not None:
                return await self._unclear_turn(unclear.intents[0], events)
        try:
            interpretation = await self._decomposer.interpret(
                message, tuple(self._active.values()) + tuple(self._resumable.values()), self._history_now
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
        self._sessions_this_run = {}
        return await execute_plan(
            the_plan,
            intents,
            self._runners,
            timeout=self._route_timeouts,
            context_for=self._context_for,
        )

    @staticmethod
    def _is_answer_no(message: str, asked: tuple[Question, ...]) -> bool:
        """A no that answers the one open question a route asked, rather than withdrawing the request."""
        if len(asked) != 1 or asked[0].slot in (CONFIRM_SLOT, CLARIFY_SLOT, IDENTITY_SLOT):
            return False
        text = message.strip().lower()
        return bool(_NO.match(text)) and not _WITHDRAWS.search(text)

    def _free_answer(
        self, message: str, asked_before: Mapping[str, tuple[Question, ...]], decomposition: Decomposition
    ) -> Optional[tuple[str, Question, tuple[Intent, ...]]]:
        """(intent id, the question, intents to fold in) when this message is the customer's answer to the one open question a route asked.

        Only when exactly one intent waits on one question of its own (not a yes, not identity), the message is short,
        it is not a bare yes or no, the decomposer gave that intent no new details, and nothing it found is a new
        request that changes anything."""
        waiting = [
            (iid, qs[0])
            for iid, qs in asked_before.items()
            if iid in self._active and len(qs) == 1 and qs[0].slot not in (CONFIRM_SLOT, CLARIFY_SLOT, IDENTITY_SLOT)
        ]
        if len(waiting) != 1 or len(message) > 300 or is_affirmation(message):
            return None
        intent_id, question = waiting[0]
        fold: list[Intent] = []
        for found in decomposition.intents:
            if found.id == intent_id and found.slots != self._active[intent_id].slots:
                return None  # the decomposer already put the details where they belong
            if found.id not in self._active and found.kind is self._active[intent_id].kind:
                fold.append(found)  # the same request said again with more detail: the answer, not a second request
            elif found.id not in self._active and found.kind not in _ANSWER_NOISE:
                return None  # a real new request
        return intent_id, question, tuple(fold)

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
        self._contained = []
        if any(i.kind is not IntentKind.UNCLEAR_OR_GREETING for i in decomposition.intents):
            self._help = HelpState()
        for flag in decomposition.flags:
            events.append(TraceEvent(kind=TraceKind.FLAGGED, detail=flag.value))

        asked_pending = dict(self._pending_route_questions)
        for intent_id in interpretation.declined:
            if self._is_answer_no(message, asked_pending.get(intent_id, ())):
                # "No" to the one question a route asked ("have you contacted the merchant?") is the answer, not the
                # customer withdrawing the whole request.
                (asked_no,) = asked_pending[intent_id]
                intent_now = self._active[intent_id]
                no_said = message.strip()[:500]
                talk_so_far = intent_now.slots.get("conversation_so_far", "")
                exchange = f"{talk_so_far}\nAsked: {asked_no.text or asked_no.slot.replace('_', ' ')}\nCustomer: {no_said}".strip()
                self._active[intent_id] = intent_now.model_copy(
                    update={"slots": {**intent_now.slots, asked_no.slot: "no", "conversation_so_far": exchange[-1500:]}}
                )
                self._prints[intent_id] = fingerprint(self._active[intent_id])
                self._asks[intent_id] = 0
                self._pending_route_questions.pop(intent_id, None)
                events.append(TraceEvent(kind=TraceKind.DECIDED, intent_id=intent_id, detail="answered no", reason=asked_no.slot))
                continue
            if self._resumable.pop(intent_id, None) is not None:
                self._progress.pop(intent_id, None)
                events.append(
                    TraceEvent(
                        kind=TraceKind.SKIPPED, intent_id=intent_id, reason="withdrawn by the customer"
                    )
                )
            if intent_id in self._active:
                self._forget(intent_id)
                events.append(
                    TraceEvent(
                        kind=TraceKind.SKIPPED, intent_id=intent_id, reason="withdrawn by the customer"
                    )
                )

        asked_before = dict(self._pending_route_questions)  # what a route was waiting on, before this message
        answering = self._free_answer(message, asked_before, decomposition)
        if answering is not None:
            # The message answers the one question on screen. The decomposer sometimes reads such an answer as a new
            # request or as a yes; the answer is kept as the detail that was asked for, and nothing new is made of it.
            decomposition = decomposition.model_copy(
                update={"intents": tuple(i for i in decomposition.intents if i.id in self._active)}
            )
        touched: set[str] = set()
        for intent in decomposition.intents:
            if intent.id in self._done:
                continue
            if intent.kind is IntentKind.REPORT_FRAUD_OR_LOSS:
                # The customer's words when they reported it, kept once as a detail of the intent: the cards they
                # name there are what containment freezes and what the specialist is told was in the wallet. It
                # never changes afterwards, so a plan's approval stays bound to the same details.
                said = self._report_text.setdefault(intent.id, message[:2000])
                intent = intent.model_copy(update={"slots": {**intent.slots, "said": said}})
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
            self._resumable.pop(intent.id, None)  # asked to try again: it runs with its progress
            touched.add(intent.id)

        if answering is not None:
            answered_id, asked, folded = answering
            asked_slot = asked.slot
            waiting_intent = self._active[answered_id]
            for extra in folded:  # details the decomposer put on a second copy of the same request
                waiting_intent = waiting_intent.model_copy(update={"slots": {**extra.slots, **waiting_intent.slots}})
            said = message.strip()[:500]
            # The exchange is kept as one detail of the request, in words: the specialist reads it like a conversation
            # (the summary stays the customer's own short statement of what they want).
            earlier = waiting_intent.slots.get("conversation_so_far", "")
            exchange = f"{earlier}\nAsked: {asked.text or asked_slot.replace('_', ' ')}\nCustomer: {said}".strip()
            self._active[answered_id] = waiting_intent.model_copy(
                update={"slots": {**waiting_intent.slots, asked_slot: said, "conversation_so_far": exchange[-1500:]}}
            )
            self._prints[answered_id] = fingerprint(self._active[answered_id])
            self._asks[answered_id] = 0
            # Not "touched", like a yes: the customer only answered, so the request keeps the score it earned when it
            # was stated (scoring the answer alone would call a clear request unclear), and runs again at once.
            self._pending_route_questions.pop(answered_id, None)
            events.append(
                TraceEvent(kind=TraceKind.DECIDED, intent_id=answered_id, detail="answered", reason=asked_slot)
            )

        for intent_id in touched:
            # The customer said something about it: whatever the route asked is answered
            # or superseded, so it is run again (and scored again) with the new details.
            self._pending_route_questions.pop(intent_id, None)

        for intent_id in interpretation.confirmed:
            shown = self._pending_confirm.pop(intent_id, None)
            if shown is not None and intent_id in self._active:
                self._confirmed[intent_id] = shown
                touched.add(intent_id)
                events.append(
                    TraceEvent(kind=TraceKind.DECIDED, intent_id=intent_id, detail="confirmed")
                )
            elif intent_id in self._active and len(asked_before.get(intent_id, ())) == 1:
                # A yes to the one question a route asked ("are you sure?") is its answer: the asked detail is "yes".
                (asked,) = asked_before[intent_id]
                self._pending_route_questions.pop(intent_id, None)
                answered = self._active[intent_id]
                self._active[intent_id] = answered.model_copy(update={"slots": {**answered.slots, asked.slot: "yes"}})
                self._prints[intent_id] = fingerprint(self._active[intent_id])
                # Not "touched": the customer only answered, so the request keeps the score it earned when it was
                # stated. Scoring a bare "yes" on its own would call a clear request unclear.
                events.append(
                    TraceEvent(kind=TraceKind.DECIDED, intent_id=intent_id, detail="answered yes", reason=asked.slot)
                )

        if interpretation.claims:
            await self._verify(interpretation.claims, events)

        decisions = await self._decide_all(message, touched, events)
        preview_ids = [i for i, d in decisions.items() if d.action is Action.PREVIEW]
        previewed: tuple[RouteOutcome, ...] = ()
        settled: frozenset[str] = frozenset()
        view = dict(self._active)
        if preview_ids:
            previewed, settled = await self._preview(preview_ids, decisions, events)
        run_intents = [
            self._active[i]
            for i, d in decisions.items()
            if d.action is Action.RUN and i not in settled
        ]
        run_intents = closes_wait_for_openings(run_intents)
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

        executed = await self._execute(the_plan, {i.id: i for i in run_intents})
        events.extend(executed.trace.events)
        for outcome in previewed:
            if outcome.status is OutcomeStatus.DONE:
                self._done.append(outcome.intent_id)
            self._forget(outcome.intent_id)

        for outcome in executed.outcomes:
            if outcome.status is OutcomeStatus.NEEDS_INFO:
                self._route_asked(outcome, decisions, events)
        # Only intents still open (not escalated for asking too often) hold up others.
        still_asking = {
            o.intent_id
            for o in executed.outcomes
            if o.status is OutcomeStatus.NEEDS_INFO and o.intent_id in self._active
        }
        waiting_on_answer = self._waiting_on_answer(executed.outcomes, still_asking, view)
        for outcome in executed.outcomes:
            if outcome.status is OutcomeStatus.DONE:
                self._done.append(outcome.intent_id)
                self._progress.pop(outcome.intent_id, None)
            if outcome.status is OutcomeStatus.PARTIAL:
                self._keep_for_retry(outcome.intent_id, view)
            if outcome.intent_id in still_asking or outcome.intent_id in waiting_on_answer:
                continue  # stays open until the question it depends on is answered
            self._forget(outcome.intent_id)
        for decision in decisions.values():
            if decision.action in (Action.ESCALATE, Action.DEFER):
                self._forget(decision.intent_id)
        if self._deferred_human is not None and not self._protection_open():
            # The cards are protected (or the report was withdrawn): the person the customer asked for.
            owed, self._deferred_human = self._deferred_human, None
            owed_decision = IntentDecision(
                intent_id=owed.id,
                action=Action.ESCALATE,
                reason="the customer asked for a person while their cards were being protected",
            )
            decisions[owed.id] = owed_decision
            view[owed.id] = owed
            events.extend(_decision_events(owed_decision))

        ordered = tuple(decisions.values())
        self._awaiting = any(d.action is Action.ASK for d in ordered)
        self._last_asked = frozenset(
            q.slot for d in ordered if d.action is Action.ASK for q in d.questions
        )
        contained = tuple(outcome for _, outcome in self._contained)
        view.update({freeze.id: freeze for freeze, _ in self._contained})
        reported = contained + previewed + tuple(
            o
            for o in executed.outcomes
            if o.status is not OutcomeStatus.NEEDS_INFO and o.intent_id not in waiting_on_answer
        )
        waiting = the_plan.waiting + tuple(
            Waiting(intent_id=i, blocked_by=blockers) for i, blockers in waiting_on_answer.items()
        )
        reply = compose_reply(view, outcomes=reported, decisions=ordered, waiting=waiting)
        for on_screen in reply.questions:
            # The question on screen is the one the customer's next message answers.
            if on_screen.intent_id in self._pending_route_questions:
                self._pending_route_questions[on_screen.intent_id] = (on_screen,)
        return TurnResult(
            reply=reply,
            trace=Trace(events=tuple(events)),
            decisions=ordered,
            # Settled while previewing (nothing needed changing, or the plan was refused)
            # first, then what ran.
            outcomes=contained + previewed + executed.outcomes,
        )

    def _continued(self, score: IntentScore) -> IntentScore:
        """A request continued with the same kind keeps the kind and scope confidence it earned when it was
        stated. A short reply ("yes, I am sure") says nothing about what the request is, so scoring it alone would
        call a clear request unclear. Only the details are scored again, and nothing is ever raised above what an
        earlier score gave."""
        intent = self._active.get(score.intent_id)
        earlier = self._scores.get(score.intent_id)
        kept = self._scored_kind.get(score.intent_id)
        if intent is not None:
            self._scored_kind[score.intent_id] = intent.kind
        if (
            intent is None
            or earlier is None
            or kept is not intent.kind
            or earlier.error is not None
            or score.error is not None
        ):
            return score
        return score.model_copy(
            update={
                "kind_confidence": max(score.kind_confidence, earlier.kind_confidence),
                "scope_confidence": max(score.scope_confidence, earlier.scope_confidence),
            }
        )

    def _protection_open(self) -> bool:
        """A lost-wallet report is still being worked on: asked about, waiting for a yes, or half done."""
        return any(
            i.kind is IntentKind.REPORT_FRAUD_OR_LOSS
            for i in (*self._active.values(), *self._resumable.values())
        )

    def _hand_off_decision(self, intent: Intent) -> IntentDecision:
        """A request for a person escalates at once, except while card protection is open: then it is counted and
        kept for when the cards are safe, and escalates at once on the HUMAN_AFTER_ASKS-th ask."""
        if intent.kind is not IntentKind.REQUEST_HUMAN or not self._protection_open():
            return decide(intent, IntentScore.unscored(intent.id, "not scored"), {})
        self._human_asks += 1
        if self._human_asks >= HUMAN_AFTER_ASKS:
            self._deferred_human = None
            return IntentDecision(
                intent_id=intent.id,
                action=Action.ESCALATE,
                reason=f"the customer asked for a person {self._human_asks} times",
            )
        self._deferred_human = intent
        return IntentDecision(
            intent_id=intent.id,
            action=Action.DEFER,
            reason="card protection is open: the customer's cards come first",
        )

    def _keep_for_retry(self, intent_id: str, view: Mapping[str, Intent]) -> None:
        """A partly finished intent: remember what it committed so a retry does not repeat it."""
        session = self._sessions_this_run.get(intent_id)
        intent = view.get(intent_id)
        if session is None or intent is None:
            return
        steps = tuple(session.completed())
        if steps:
            self._progress[intent_id] = steps
            self._resumable[intent_id] = intent

    def _previews(self, intent: Intent) -> bool:
        runner = self._runners.get(route_for(intent.kind))
        return bool(getattr(runner, "proposes_plans", False))

    def _plan_decision(self, intent: Intent) -> IntentDecision:
        """Where an intent stands once it passed the confidence and detail checks."""
        current = fingerprint(intent)
        proposal = self._proposals.get(intent.id)
        if proposal is not None and proposal.intent_fp != current:
            # The details changed since the plan was shown: it, and any yes to it, is stale.
            self._proposals.pop(intent.id, None)
            self._pending_confirm.pop(intent.id, None)
            self._confirmed.pop(intent.id, None)
            proposal = None
        if proposal is None:
            return IntentDecision(
                intent_id=intent.id, action=Action.PREVIEW, reason="working out exactly what to do"
            )
        if proposal.reversible_only or self._confirmed.get(intent.id) == proposal.plan_fp:
            self._approved[intent.id] = proposal
            reason = "every step can be undone" if proposal.reversible_only else "plan approved"
            return IntentDecision(intent_id=intent.id, action=Action.RUN, reason=reason)
        return self._ask_for_plan(intent.id, proposal, "awaiting the customer's yes to the plan")

    @staticmethod
    def _ask_for_plan(intent_id: str, proposal: PlanPreview, reason: str) -> IntentDecision:
        return IntentDecision(
            intent_id=intent_id,
            action=Action.ASK,
            reason=reason,
            questions=(Question(intent_id=intent_id, slot=CONFIRM_SLOT, text=proposal.rendered),),
            fingerprint=proposal.plan_fp,
        )

    async def _preview(
        self,
        intent_ids: Sequence[str],
        decisions: dict[str, IntentDecision],
        events: list[TraceEvent],
    ) -> tuple[tuple[RouteOutcome, ...], frozenset[str]]:
        """Run the routes in plan mode: reads happen, writes are refused, a plan comes back.

        Returns the outcomes that are already final (nothing needed changing, or the route
        failed) and the ids they belong to, so those are not run again.
        """
        for intent_id in intent_ids:
            events.extend(_decision_events(decisions[intent_id]))
        await self._contain(intent_ids, events)
        self._mode = "plan"
        try:
            executed = await self._execute(
                Plan(waves=(tuple(intent_ids),), waiting=()),
                {i: self._active[i] for i in intent_ids},
            )
        finally:
            self._mode = "execute"
        events.extend(executed.trace.events)
        final: list[RouteOutcome] = []
        for outcome in executed.outcomes:
            intent_id = outcome.intent_id
            if outcome.status is OutcomeStatus.NEEDS_CONFIRMATION and outcome.proposal is not None:
                proposal = outcome.proposal
                self._proposals[intent_id] = proposal
                if proposal.reversible_only:
                    self._approved[intent_id] = proposal
                    decisions[intent_id] = IntentDecision(
                        intent_id=intent_id, action=Action.RUN, reason="every step can be undone"
                    )
                else:
                    self._pending_confirm[intent_id] = proposal.plan_fp
                    self._asks[intent_id] = self._asks.get(intent_id, 0) + 1
                    decisions[intent_id] = self._ask_for_plan(
                        intent_id, proposal, "awaiting the customer's yes to the plan"
                    )
            elif outcome.status is OutcomeStatus.NEEDS_INFO:
                self._route_asked(outcome, decisions, events)
            else:
                final.append(outcome)
                decisions[intent_id] = IntentDecision(
                    intent_id=intent_id, action=Action.RUN, reason="nothing needed to change"
                )
        return tuple(final), frozenset(o.intent_id for o in final)

    async def _contain(self, intent_ids: Sequence[str], events: list[TraceEvent]) -> None:
        """Freeze the cards a lost-wallet report names, before its plan is worked out (the freeze is reversible)."""
        user_id = self._auth.verified_user_id
        if self._containment is None or user_id is None:
            return
        for intent_id in intent_ids:
            intent = self._active[intent_id]
            if intent.kind is not IntentKind.REPORT_FRAUD_OR_LOSS:
                continue
            freezes = self._containment.freezes(intent, user_id, self._report_text.get(intent_id, ""))
            if not freezes:
                continue
            executed = await self._execute(
                Plan(waves=(tuple(f.id for f in freezes),), waiting=()), {f.id: f for f in freezes}
            )
            events.extend(executed.trace.events)
            by_id = {f.id: f for f in freezes}
            self._contained.extend((by_id[o.intent_id], o) for o in executed.outcomes if o.intent_id in by_id)

    def _route_asked(
        self,
        outcome: RouteOutcome,
        decisions: dict[str, IntentDecision],
        events: list[TraceEvent],
    ) -> None:
        """A route needs an answer. Keep the intent open and ask, up to MAX_ASKS times."""
        intent_id = outcome.intent_id
        # Asked one at a time, a request may need several questions in turn. The count is of the same question
        # asked again, so a new question starts it over.
        first = outcome.questions[0].slot if outcome.questions else ""
        if self._last_route_slot.get(intent_id) != first:
            self._runner_asks[intent_id] = 0
        self._last_route_slot[intent_id] = first
        asked = self._runner_asks.get(intent_id, 0) + 1
        if asked > MAX_ASKS:
            decision = IntentDecision(
                intent_id=intent_id,
                action=Action.ESCALATE,
                reason=f"still not enough information after {MAX_ASKS} questions from the route",
            )
            events.extend(_decision_events(decision)[1:])  # the ESCALATED event
            self._forget(intent_id)
        else:
            self._runner_asks[intent_id] = asked
            self._pending_route_questions[intent_id] = outcome.questions
            decision = IntentDecision(
                intent_id=intent_id,
                action=Action.ASK,
                reason="the route needs more information",
                questions=outcome.questions,
            )
        decisions[intent_id] = decision

    @staticmethod
    def _waiting_on_answer(
        outcomes: Sequence[RouteOutcome], asking: set[str], view: Mapping[str, Intent]
    ) -> dict[str, tuple[str, ...]]:
        """Skipped intents that depend, directly or not, on an intent that is asking."""
        blocked: dict[str, tuple[str, ...]] = {}
        stuck = set(asking)
        changed = True
        while changed:
            changed = False
            for outcome in outcomes:
                intent = view.get(outcome.intent_id)
                if (
                    intent is None
                    or outcome.status is not OutcomeStatus.SKIPPED
                    or outcome.intent_id in blocked
                ):
                    continue
                direct = tuple(d for d in intent.depends_on if d in stuck)
                if direct:
                    blocked[outcome.intent_id] = direct
                    stuck.add(outcome.intent_id)
                    changed = True
        return blocked

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
        for intent_id in touched:
            self._said.setdefault(intent_id, message)
        to_score = [i for i in scoreable if i.id in touched or i.id not in self._scores]
        # An intent held back (by the identity check, say) is scored on the message that stated it, not on whatever
        # the customer says later ("my phone is ..." would score a clear request as unclear).
        by_words: dict[str, list[Intent]] = {}
        for i in to_score:
            by_words.setdefault(message if i.id in touched else self._said.get(i.id, message), []).append(i)
        for words, group in by_words.items():
            for score in await score_intents(self._scorer, words, _view(group), timeout=self._scorer_timeout):
                score = self._continued(score)
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
                decision = self._hand_off_decision(intent)
            elif intent.id in self._pending_route_questions and intent.id not in touched:
                decision = IntentDecision(
                    intent_id=intent.id,
                    action=Action.ASK,
                    reason="still waiting for an answer",
                    questions=self._pending_route_questions[intent.id],
                )
            else:
                previewing = self._previews(intent)
                # A route that proposes plans gets its yes from the plan, not the intent.
                already = {intent.id: fingerprint(intent)} if previewing else self._confirmed
                decision = decide(intent, self._scores[intent.id], already)
                if previewing and decision.action is Action.RUN:
                    decision = self._plan_decision(intent)
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
