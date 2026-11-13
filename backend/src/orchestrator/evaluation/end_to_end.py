"""Run each message through the whole orchestrator and count what would change.

The decomposer and the scorer are the real ones; the caller is already verified and
the routes are mocks that always succeed, so nothing touches a bank. For one turn we
record, for every write the system settled on, whether the customer asked for it and
what the decision layer did: ran it, asked the customer, or handed it off. The number
that matters most is writes that ran although the customer never asked for them.
The invariant that must stay at zero: an irreversible write running with no yes.
"""

import asyncio
import inspect
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from orchestrator.auth import CustomerRecord, ScriptedDirectory
from orchestrator.decision import Action, needs_confirmation
from orchestrator.decomposer import Decomposer, Interpretation
from orchestrator.evaluation.tasks import EvalCase
from orchestrator.mocks import ScriptedRoute
from orchestrator.orchestrator import Orchestrator
from orchestrator.plan import READ_ONLY_KINDS
from orchestrator.routes import Route
from orchestrator.scoring import Scorer
from orchestrator.types import Intent, IntentKind

# Matches the one record in ``ScriptedDirectory`` below; two of four fields verify.
_CLAIMS = {"email": "kenji.tanaka@example.com", "phone_number": "206-555-0293"}


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class WriteOutcome(_Model):
    task_id: str
    kind: IntentKind
    requested: bool
    action: Action


class Summary(_Model):
    messages: int
    no_action_messages: int
    writes: int
    # A write ran although the customer never asked for it.
    unrequested_ran: int
    unrequested_asked: int
    unrequested_handed_off: int
    messages_with_unrequested_ran: int
    # Unrequested writes of a kind that needs no yes. They ran only if no detail was
    # missing; once the customer supplies the details, nothing else stops them.
    unrequested_skipping_the_yes: int
    # Must be zero: an irreversible kind ran on turn one with no yes.
    irreversible_ran_without_yes: int
    requested_ran: int
    requested_asked: int
    requested_handed_off: int
    requested_not_found: int
    ran_by_kind: tuple[tuple[IntentKind, int], ...]


class _VerifiedCaller:
    """Wraps a decomposer: the caller has already given two matching details."""

    def __init__(self, inner: Decomposer) -> None:
        self._inner = inner
        self.intents: tuple[Intent, ...] = ()

    async def interpret(
        self,
        message: str,
        open_intents: Sequence[Intent],
        history: Sequence[tuple[str, str]] = (),
        pending: Sequence[tuple[str, str]] = (),
    ) -> Interpretation:
        if "pending" in inspect.signature(self._inner.interpret).parameters:
            result = await self._inner.interpret(message, open_intents, history, pending)
        else:
            result = await self._inner.interpret(message, open_intents, history)
        self.intents = result.decomposition.intents
        return result.model_copy(update={"claims": dict(_CLAIMS)})


def _directory() -> ScriptedDirectory:
    return ScriptedDirectory(
        CustomerRecord(
            user_id="u_kenji01",
            date_of_birth="03/14/1990",
            email=_CLAIMS["email"],
            phone_number=_CLAIMS["phone_number"],
            address="12 Pine St, Seattle, WA 98101",
        )
    )


async def run_message(
    case: EvalCase, decomposer: Decomposer, scorer: Scorer
) -> tuple[list[WriteOutcome], int]:
    """Writes the system settled on for one message, and how many stated writes it never found."""
    caller = _VerifiedCaller(decomposer)
    orchestrator = Orchestrator(
        decomposer=caller,
        scorer=scorer,
        directory=_directory(),
        runners={Route.DIRECT: ScriptedRoute(), Route.SUB_AGENT: ScriptedRoute()},
    )
    result = await orchestrator.handle_turn("m1", case.message)
    kinds = {i.id: i.kind for i in caller.intents}
    stated = set(case.stated)
    found = {kinds[d.intent_id] for d in result.decisions if d.intent_id in kinds}
    outcomes = [
        WriteOutcome(
            task_id=case.task_id,
            kind=kinds[d.intent_id],
            requested=kinds[d.intent_id] in stated,
            action=d.action,
        )
        for d in result.decisions
        if d.intent_id in kinds and kinds[d.intent_id] not in READ_ONLY_KINDS
    ]
    return outcomes, len(stated - found)


async def run_end_to_end(
    cases: Sequence[EvalCase],
    make_decomposer: Callable[[], Decomposer],
    make_scorer: Callable[[], Scorer],
    *,
    concurrency: int = 6,
) -> tuple[list[WriteOutcome], dict[str, int]]:
    gate = asyncio.Semaphore(concurrency)

    async def one(case: EvalCase) -> tuple[list[WriteOutcome], int]:
        async with gate:
            return await run_message(case, make_decomposer(), make_scorer())

    rows = await asyncio.gather(*(one(c) for c in cases))
    not_found = {c.task_id: n for c, (_, n) in zip(cases, rows, strict=True)}
    return [o for outcomes, _ in rows for o in outcomes], not_found


def summarize(
    cases: Sequence[EvalCase], outcomes: Sequence[WriteOutcome], not_found: dict[str, int]
) -> Summary:
    unrequested = [o for o in outcomes if not o.requested]
    requested = [o for o in outcomes if o.requested]
    ran_by_kind: dict[IntentKind, int] = {}
    for o in unrequested:
        if o.action is Action.RUN:
            ran_by_kind[o.kind] = ran_by_kind.get(o.kind, 0) + 1
    return Summary(
        messages=len(cases),
        no_action_messages=sum(not c.stated for c in cases),
        writes=len(outcomes),
        unrequested_ran=sum(o.action is Action.RUN for o in unrequested),
        unrequested_asked=sum(o.action is Action.ASK for o in unrequested),
        unrequested_handed_off=sum(o.action is Action.ESCALATE for o in unrequested),
        messages_with_unrequested_ran=len(
            {o.task_id for o in unrequested if o.action is Action.RUN}
        ),
        unrequested_skipping_the_yes=sum(not needs_confirmation(o.kind) for o in unrequested),
        irreversible_ran_without_yes=sum(
            o.action is Action.RUN and needs_confirmation(o.kind) for o in outcomes
        ),
        requested_ran=sum(o.action is Action.RUN for o in requested),
        requested_asked=sum(o.action is Action.ASK for o in requested),
        requested_handed_off=sum(o.action is Action.ESCALATE for o in requested),
        requested_not_found=sum(not_found.values()),
        ran_by_kind=tuple(sorted(ran_by_kind.items())),
    )


def _share(n: int, total: int) -> str:
    return f"{n}/{total} ({n / total:.1%})" if total else f"{n}/0"


def render(summary: Summary) -> str:
    s = summary
    unrequested = s.unrequested_ran + s.unrequested_asked + s.unrequested_handed_off
    lines = [
        f"{s.messages} messages ({s.no_action_messages} ask the bank for nothing), {s.writes} writes settled on",
        "",
        f"writes nobody asked for: {unrequested}",
        f"  ran with no yes:        {s.unrequested_ran}  (in {s.messages_with_unrequested_ran} messages)",
        f"  asked the customer:     {s.unrequested_asked}",
        f"  handed to a person:     {s.unrequested_handed_off}",
        f"  of a kind that skips the yes (would run once details are given): {s.unrequested_skipping_the_yes}",
        "",
        f"writes the customer asked for: {s.requested_ran + s.requested_asked + s.requested_handed_off}",
        f"  ran with no yes:        {s.requested_ran}",
        f"  asked the customer:     {s.requested_asked}",
        f"  handed to a person:     {s.requested_handed_off}",
        f"  never found at all:     {s.requested_not_found}",
        "",
        f"IRREVERSIBLE RAN WITHOUT A YES (must be 0): {s.irreversible_ran_without_yes}",
    ]
    if s.ran_by_kind:
        lines.append("unrequested writes that ran: " + ", ".join(f"{k} x{n}" for k, n in s.ran_by_kind))
    return "\n".join(lines)
