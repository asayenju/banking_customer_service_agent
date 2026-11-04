"""Tier-1 evidence that a specialist works: no model, no network, a real database.

For one tau2 task, replay what the task's gold trajectory does, but send every change the
specialist is responsible for through the specialist itself:

  1. the gold's writes are cut into segments, each one a single request the specialist
     could be handed (a PIN reset, a close-and-replace, a set of disputes);
  2. for each segment a scripted "model" reads what the gold reads and then proposes the
     gold's writes as a plan;
  3. the specialist checks the plan (the code guard, the validators, a dry run on a copy
     of the database), the plan is "approved", and the specialist runs it in code;
  4. what the gold does outside the specialists' job is done directly, as the orchestrator
     would: the identity log, standalone freezes, customer-side tools.

At the end the database is compared with the one tau2 gets by replaying the same gold. This
proves the plumbing (the guard accepts what it should, the validators agree with the bank's
tools, the plan runs to the same state). It does not prove a model would choose these
steps: that is tier 2, run live, in small capped batches.

Tasks scored on tool calls, not on the database (tau2's ACTION basis), are compared by
the set of changes the session made against the gold's.
"""

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from pydantic import ValidationError
from tau2.domains.banking_knowledge.tools import KnowledgeUserTools

from agent.guard import KIND_WRITE_TOOLS
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.loop import Budget
from agent.specialist import Specialist, SpecialistConfig
from agent.toolbox import SecretVault
from banking_tools.catalog import get_registry
from banking_tools.gold_replay import read_log_allowlist, replay_on_tau2
from banking_tools.session import REDACTED_ARGUMENTS, BankingSession, Fidelity, _base_db
from banking_tools.statehash import compare
from banking_tools.task_state import apply_initial_state
from banking_tools.types import ToolKind
from knowledge.index import KnowledgeIndex
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.executor import OutcomeStatus
from orchestrator.plan_preview import unapproved_steps
from orchestrator.types import Intent, IntentKind

_CALL = "call_discoverable_agent_tool"
_ID_SLOTS = ("card_id", "account_id", "transaction_id", "credit_card_account_id")
# Writes the orchestrator or a deterministic recipe makes; never a specialist.
DIRECT_WRITES = frozenset({"freeze_debit_card_3892", "change_user_email"})
DISPUTE_TOOLS = frozenset({"file_debit_card_transaction_dispute_6281", "file_credit_card_transaction_dispute_4829"})
_HANDOFFS = frozenset(
    {
        "transfer_to_human_agents",
        "initial_transfer_to_human_agent_0218",
        "initial_transfer_to_human_agent_1822",
        "emergency_credit_bureau_incident_transfer_1114",
        "give_discoverable_user_tool",
    }
)
# More room than production: a gold trajectory can read a dozen things before one change.
_ROOM = Budget(max_steps=80, max_tool_calls=120, max_writes=40, max_errors=10, max_input_tokens=10**9)


@dataclass(frozen=True)
class Step:
    kind: str  # verify | user | other | read | write
    name: str
    args: dict[str, Any]
    action_id: str = ""


@dataclass(frozen=True)
class Tier1Result:
    task_id: str
    strict_equal: bool
    semantic_equal: bool
    mismatches: tuple[str, ...]
    segments: tuple[str, ...]
    # For tasks scored on tool calls: did the session make the same changes as the gold?
    actions_equal: Optional[bool] = None
    error: Optional[str] = None
    # Set when the task cannot be used at all (its own gold breaks the bank's tool contract).
    # Such a task is neither a pass nor a failure.
    excluded: Optional[str] = None
    # How many model turns the scripted run needed (a proxy for work, since no tokens are spent).
    model_turns: int = 0
    seconds: float = 0.0
    # Writes that took effect although the approved plan did not contain them. Must be zero.
    bypasses: int = 0

    @property
    def passed(self) -> bool:
        if self.error or self.excluded:
            return False
        return self.actions_equal if self.actions_equal is not None else self.semantic_equal


def gold_steps(task: Mapping[str, Any]) -> list[Step]:
    """The task's gold agent-side actions, in order, with discoverable tools unwrapped."""
    registry = get_registry()
    steps: list[Step] = []
    for action in (task.get("evaluation_criteria") or {}).get("actions") or []:
        name, args, who = action["name"], dict(action.get("arguments") or {}), action["requestor"]
        action_id = str(action.get("action_id", ""))
        if who == "user":
            steps.append(Step("user", name, args, action_id))
        elif name == "unlock_discoverable_agent_tool":
            continue
        elif name == _CALL:
            inner = json.loads(args.get("arguments") or "{}", parse_int=float)
            steps.append(_classify(args["agent_tool_name"], inner, action_id, registry))
        else:
            steps.append(_classify(name, args, action_id, registry))
    return steps


def _classify(name: str, args: dict[str, Any], action_id: str, registry: Mapping[str, Any]) -> Step:
    if name == "log_verification":
        return Step("verify", name, args, action_id)
    spec = registry.get(name)
    if name in _HANDOFFS or spec is None or spec.kind is ToolKind.HANDOFF:
        return Step("other", name, args, action_id)
    return Step("read" if spec.kind is ToolKind.READ else "write", name, args, action_id)


def invalid_gold(steps: Sequence[Step]) -> Optional[str]:
    """Why the gold itself cannot be replayed through typed tools, if it cannot."""
    registry = get_registry()
    for step in steps:
        spec = registry.get(step.name)
        if step.kind != "write" or spec is None or spec.args_model is None:
            continue
        try:
            spec.args_model.model_validate(step.args)
        except ValidationError as exc:
            missing = sorted({str(e["loc"][0]) for e in exc.errors() if e["type"] == "missing"})
            detail = f"is missing {', '.join(missing)}" if missing else "has invalid arguments"
            return f"the gold's call to {step.name} {detail}, which the bank's tool requires"
    return None


def choose_kind(writes: Sequence[Step], kinds: Sequence[IntentKind]) -> tuple[Optional[IntentKind], int]:
    """The kind that can make the longest run of the next writes, and how long that run is."""
    best: Optional[IntentKind] = None
    best_run = 0
    for kind in kinds:
        allowed = KIND_WRITE_TOOLS.get(kind, frozenset())
        run = 0
        while run < len(writes) and writes[run].name in allowed:
            run += 1
        if run > best_run:
            best, best_run = kind, run
    return best, best_run


def slots_of(first: Step) -> dict[str, str]:
    return {k: str(first.args[k]) for k in _ID_SLOTS if first.args.get(k)}


def _preference(config: SpecialistConfig) -> list[IntentKind]:
    order = [
        IntentKind.CHANGE_OR_RESET_PIN,
        IntentKind.ACTIVATE_CARD,
        IntentKind.CARD_CONTROLS,
        IntentKind.CLOSE_CARD,
        IntentKind.ORDER_REPLACEMENT_CARD,
        IntentKind.DISPUTE_TRANSACTION,
    ]
    return [k for k in order if k in config.kinds]


@dataclass(frozen=True)
class Direct:
    """Something the gold does outside a specialist's job: done in code, as the orchestrator would."""

    step: Step


@dataclass(frozen=True)
class Segment:
    """One request a specialist is handed: the reads that lead up to it and the writes it makes."""

    kind: IntentKind
    reads: tuple[Step, ...]
    writes: tuple[Step, ...]


Item = Direct | Segment


def plan_items(steps: Sequence[Step], config: SpecialistConfig) -> list[Item]:
    """Cut the gold into direct steps and specialist segments, keeping every step in order."""
    items: list[Item] = []
    pending: list[Step] = []
    i = 0

    def flush() -> None:
        items.extend(Direct(r) for r in pending)
        pending.clear()

    while i < len(steps):
        step = steps[i]
        if step.kind == "read":
            pending.append(step)
            i += 1
            continue
        if step.kind != "write" or step.name in DIRECT_WRITES:
            flush()
            items.append(Direct(step))
            i += 1
            continue
        ahead: list[Step] = []
        for later in steps[i:]:  # as far as reads and writes run unbroken: a user or hand-off step ends a request
            if later.kind not in ("write", "read"):
                break
            ahead.append(later)
        kind, run = choose_kind([s for s in ahead if s.kind == "write"], _preference(config))
        taken: list[Step] = []
        j, seen = i, 0
        while kind is not None and j < len(steps) and seen < run and steps[j].kind in ("read", "write"):
            taken.append(steps[j])
            seen += steps[j].kind == "write"
            j += 1
        writes = tuple(s for s in taken if s.kind == "write")
        if kind is None or len(writes) != run or run == 0:
            flush()
            items.append(Direct(step))
            i += 1
            continue
        items.append(Segment(kind, tuple(pending) + tuple(s for s in taken if s.kind == "read"), writes))
        pending.clear()
        i = j
    flush()
    return _split_requests(items, config)


def _split_requests(items: list[Item], config: SpecialistConfig) -> list[Item]:
    """A gold segment can hold several customer requests. When a workflow covers the dispute in
    one, cut the rest out as its own requests, so a model is handed one request at a time:
    unfreezing is a direct request, clearing an alert is card controls, and closing and
    ordering cards is a replacement."""
    out: list[Item] = []
    for item in items:
        disputes = [w for w in item.writes if w.name in DISPUTE_TOOLS] if isinstance(item, Segment) else []
        if not isinstance(item, Segment) or not disputes or not any(item.kind in w.kinds for w in config.workflows):
            out.append(item)
            continue
        closed_later = {w.args["card_id"] for w in item.writes if w.name == "close_debit_card_4721"}
        controls = [
            w
            for w in item.writes
            if w.name in ("unfreeze_debit_card_3893", "clear_debit_card_fraud_alert_4892")
            and w.args["card_id"] not in closed_later
        ]
        # The card action a fraud dispute implies (close) is the dispute workflow's own step.
        fraud_cards = {d.args["card_id"] for d in disputes if d.args["card_action"] == "close_and_reissue"}
        implied = [
            w
            for w in item.writes
            if w.name == "close_debit_card_4721" and w.args["card_id"] in fraud_cards and w.args["reason"] == "fraud_suspected"
        ]
        rest = [w for w in item.writes if w not in disputes and w not in controls and w not in implied]
        out.extend(Direct(w) for w in controls if w.name == "unfreeze_debit_card_3893")
        alerts = tuple(w for w in controls if w.name != "unfreeze_debit_card_3893")
        if alerts:
            out.append(Segment(IntentKind.CARD_CONTROLS, (), alerts))
        out.append(Segment(item.kind, item.reads, tuple(disputes)))
        if rest:
            out.append(Segment(IntentKind.ORDER_REPLACEMENT_CARD, (), tuple(rest)))
    return out


class Calls:
    """Fresh ids for the scripted model's tool calls."""

    def __init__(self) -> None:
        self._n = 0

    def make(self, name: str, arguments: Mapping[str, Any]) -> ToolCall:
        self._n += 1
        return ToolCall(id=f"call-{self._n}", name=name, arguments_json=json.dumps(arguments))


def run_tier1(
    task: Mapping[str, Any],
    config: SpecialistConfig,
    *,
    index: KnowledgeIndex,
) -> Tier1Result:
    task_id = str(task["id"])
    config = replace(config, budget=_ROOM)
    allow = read_log_allowlist(task)
    session = BankingSession.from_task(task, fidelity=Fidelity.TAU2, read_log_allowlist=allow)
    start = apply_initial_state(_base_db().model_copy(deep=True), task)
    expected, _ = replay_on_tau2(task)
    steps = gold_steps(task)
    if (broken := invalid_gold(steps)) is not None:
        return Tier1Result(task_id, False, False, (), (), excluded=broken)
    user_id = next((str(s.args["user_id"]) for s in steps if s.kind == "verify"), None)
    if user_id is None:
        return Tier1Result(task_id, False, False, (), (), error="the gold never verifies a customer")
    customer_tools = KnowledgeUserTools(session.database)
    vault = SecretVault()
    ids = Calls()
    segments: list[str] = []
    turns_used = 0
    bypasses = 0
    started = time.perf_counter()

    def direct(step: Step) -> None:
        if step.kind == "user":
            getattr(customer_tools, step.name)(**step.args)
        elif step.kind == "other":
            getattr(session.toolkit, step.name)(**step.args)
        else:
            session.call(step.name, step.args)

    try:
        for item in plan_items(steps, config):
            if isinstance(item, Direct):
                direct(item.step)
                continue
            turns, extra = _run_segment(
                task, config, session, vault, ids, index, user_id,
                item.kind, item.reads, item.writes, len(segments),
            )
            turns_used += turns
            bypasses += extra
            segments.append(f"{item.kind.value}: {', '.join(w.name.rsplit('_', 1)[0] for w in item.writes)}")
    except SegmentError as exc:
        return Tier1Result(task_id, False, False, (), tuple(segments), error=str(exc), model_turns=turns_used, seconds=time.perf_counter() - started, bypasses=bypasses)

    if (task.get("evaluation_criteria") or {}).get("reward_basis") == ["ACTION"]:
        return Tier1Result(
            task_id, False, False, (), tuple(segments),
            actions_equal=_same_changes(steps, session), model_turns=turns_used,
            seconds=time.perf_counter() - started, bypasses=bypasses,
        )
    result = compare(start, expected, session.database, read_log_names=())
    return Tier1Result(
        task_id,
        result.strict_equal,
        result.semantic_equal,
        tuple(f"{m.table}/{m.pk}: {m.problem}" for m in result.mismatches),
        tuple(segments),
        model_turns=turns_used,
        seconds=time.perf_counter() - started,
        bypasses=bypasses,
    )


class SegmentError(Exception):
    pass


def _run_segment(
    task: Mapping[str, Any],
    config: SpecialistConfig,
    session: BankingSession,
    vault: SecretVault,
    ids: Calls,
    index: KnowledgeIndex,
    user_id: str,
    kind: IntentKind,
    reads: Sequence[Step],
    writes: Sequence[Step],
    number: int,
) -> tuple[int, int]:
    if kind is IntentKind.ORDER_REPLACEMENT_CARD:
        # An earlier request may already have done some of this (a fraud dispute closes the card).
        writes = [
            w
            for w in writes
            if not (
                w.name == "close_debit_card_4721"
                and session.database.debit_cards.data[w.args["card_id"]]["status"] == "CLOSED"
            )
        ]
        if not writes:
            return 0, 0
    script = workflow_turns(config, kind, reads, writes, session.database, user_id, ids) or scripted_turns(
        reads, writes, vault, ids
    )
    llm = FakeToolLLM(script)
    specialist = Specialist(config, llm=llm, index=index, database=session.database, vault=vault)
    intent = Intent(id=f"segment-{number}", kind=kind, summary="what the gold trajectory does", slots=slots_of(writes[0]))
    bypasses = plan_and_execute(specialist, intent, session, user_id, label=f"segment {number} ({kind.value})")
    return len(script), bypasses


def scripted_turns(reads: Sequence[Step], writes: Sequence[Step], vault: SecretVault, ids: Calls) -> list[AssistantTurn]:
    """What a perfect model would say for a segment: do the reads, then propose the writes.

    PINs are handed to the vault and replaced by references, exactly as a customer's would be.
    """
    plan_steps = []
    for write in writes:
        args = dict(write.args)
        for key in REDACTED_ARGUMENTS & set(args):
            args[key] = vault.store(key, str(args[key]))
        plan_steps.append({"tool": write.name, "arguments_json": json.dumps(args), "description": "gold"})
    turns = [AssistantTurn(tool_calls=(ids.make(r.name, r.args),), usage=Usage(input_tokens=1)) for r in reads]
    turns.append(AssistantTurn(tool_calls=(ids.make("propose_plan", {"steps": plan_steps}),), usage=Usage(input_tokens=1)))
    return turns



def workflow_turns(
    config: SpecialistConfig,
    kind: IntentKind,
    reads: Sequence[Step],
    writes: Sequence[Step],
    database: Any,
    user_id: str,
    ids: Calls,
) -> Optional[list[AssistantTurn]]:
    """For a kind that has a workflow: what a perfect model would say, which is the same reads
    and then the workflow's form filled in from what the gold trajectory says the customer wanted.
    None when the kind has no workflow (the segment is then scripted as a free-form plan)."""
    from agent.workflows.base import terminal_name
    from agent.workflows.gold_facts import facts_from_gold

    for workflow in config.workflows:
        if kind not in workflow.kinds:
            continue
        facts = facts_from_gold(workflow.name, writes, database)
        if facts is None:
            return None
        turns = [AssistantTurn(tool_calls=(ids.make(r.name, r.args),), usage=Usage(input_tokens=1)) for r in reads]
        turns.append(
            AssistantTurn(tool_calls=(ids.make(terminal_name(workflow), facts),), usage=Usage(input_tokens=1))
        )
        return turns
    return None


_LOOP: Optional[asyncio.AbstractEventLoop] = None


def use_loop(loop: Optional[asyncio.AbstractEventLoop]) -> None:
    """Run every coroutine of this module on one event loop (or, with None, a fresh one each time).

    A real model client caches its HTTP connection to the loop it first ran on, so a live run
    that started a new loop per call failed on the second call with "Event loop is closed".
    """
    global _LOOP
    _LOOP = loop


def run_async(coro: Any) -> Any:
    return _LOOP.run_until_complete(coro) if _LOOP is not None else asyncio.run(coro)


def plan_and_execute(
    specialist: Specialist, intent: Intent, session: BankingSession, user_id: str, *, label: str
) -> int:
    """Plan, treat the plan as approved, run it. Raises SegmentError with the reason if either fails.

    Returns how many writes took effect that the approved plan did not contain (zero, always,
    unless a guard has a hole).
    """

    def context(mode: str, approved: Any = None) -> RouteContext:
        request = SessionRequest(intent.id, mode=mode, approved=approved)  # type: ignore[arg-type]
        return RouteContext(
            intent_id=intent.id,
            verified_user_id=user_id,
            session=scoped(session)(request),
            mode=mode,  # type: ignore[arg-type]
            approved=approved,
        )

    planned = run_async(specialist.run(intent, context("plan")))
    if planned.status is not OutcomeStatus.NEEDS_CONFIRMATION or planned.proposal is None:
        detail = planned.error or " / ".join(q.slot for q in planned.questions) or planned.text[:160]
        raise SegmentError(f"{label} was not planned: {planned.status.value} {detail}".strip())
    done = run_async(specialist.run(intent, context("execute", planned.proposal)))
    if done.status is not OutcomeStatus.DONE:
        raise SegmentError(f"{label} did not run: {done.status.value} {done.error or ''}".strip())
    return len(unapproved_steps(session.completed_for(intent.id), planned.proposal))


def _canonical(tool: str, args: Mapping[str, Any]) -> str:
    cleaned = {k: v for k, v in args.items() if v is not None and k not in REDACTED_ARGUMENTS}
    return json.dumps([tool, cleaned], sort_keys=True, default=str)


def _same_changes(steps: Sequence[Step], session: BankingSession) -> bool:
    """For tasks scored on tool calls: the same set of changes, in any order."""
    wanted = sorted(_canonical(s.name, s.args) for s in steps if s.kind == "write")
    made = sorted(_canonical(e.tool, e.arguments) for e in session.audit_log if e.ok)
    return wanted == made


@dataclass(frozen=True)
class Tier1Report:
    results: tuple[Tier1Result, ...] = field(default_factory=tuple)

    def passed(self) -> int:
        return sum(r.passed for r in self.results)


def render(report: Tier1Report) -> str:
    excluded = sum(1 for r in report.results if r.excluded)
    used = len(report.results) - excluded
    lines = [f"{used} tasks, {report.passed()} passed, {excluded} excluded (offline, scripted model)"]
    for r in report.results:
        if r.excluded:
            lines.append(f"- {r.task_id}: EXCLUDED, {r.excluded}")
            continue
        verdict = "PASS" if r.passed else "FAIL"
        level = "ACTION" if r.actions_equal is not None else ("strict" if r.strict_equal else "semantic only" if r.semantic_equal else "differs")
        lines.append(f"- {r.task_id}: {verdict} ({level}), {len(r.segments)} segments, {r.model_turns} model turns")
        if r.error:
            lines.append(f"    {r.error}")
        for m in r.mismatches[:4]:
            lines.append(f"    {m}")
    return "\n".join(lines)
