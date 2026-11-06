"""Run the specialists against a real model, in small, capped, confirmed batches.

Live runs cost money, so this module is built around not spending it by accident:

  - ``--tasks`` is required and has no "all": name task ids, or one specialist (a cluster),
    and never more than ``--max-tasks`` (default 5, hard ceiling 20) or 3 ``--trials``.
  - Only tasks the offline gate can grade are accepted; an excluded task cannot be run.
  - Before anything is sent, the run prints an estimate (calls, tokens, dollars) and stops.
    Re-running with ``--yes`` spends it. No key is read and no client is built before that.
  - The estimate must fit under ``--max-cost-usd`` (default $0.50) or the run is refused.
    While running, real token usage is metered and the run stops before the call that would
    go over, so the cap holds even when the estimate was low.
  - Every model conversation is recorded as a cassette, one per task and segment, keyed by
    model. Running the same task again replays it for free and gets the same answer. A
    cassette whose prompts no longer match is an error, never a silent re-spend: pass
    ``--rerecord`` to pay for a fresh one.
  - Prices below are what was published when this was written; they are an assumption you can
    override with ``--price-in`` and ``--price-out`` (dollars per million tokens). A model with
    no known price and no override is refused, since the cap would mean nothing.

Nothing here is imported by the offline test run, and the default ``pytest`` blocks the one
function every real model call goes through.
"""

import argparse
import asyncio
import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from agent.llm_tools import AssistantTurn, ToolCallingLLMClient, ToolSchema, Usage

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASSETTES = ROOT / "evals/cassettes"
DEFAULT_MAX_TASKS = 5
HARD_MAX_TASKS = 20
HARD_MAX_TRIALS = 3
DEFAULT_MAX_COST = 0.50

# Dollars per million tokens (input, output). An assumption; see the module docstring.
PRICES: Mapping[str, tuple[float, float]] = {
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
}

# A rough shape for one model call in a specialist run, for the estimate only.
_BASE_PROMPT_TOKENS = 1_800
_GROWTH_PER_CALL = 700
_OUTPUT_PER_CALL = 150
_EXTRA_CALLS_PER_SEGMENT = 3  # a policy search, the plan, slack


class LiveRunRefused(Exception):
    """The run was not allowed to start. The message says why and what to change."""


class BudgetExceeded(Exception):
    """The next model call would take the run past its cost cap."""


class CassetteMismatch(Exception):
    """A recorded conversation no longer matches what is being asked."""


# ------------------------------------------------------------------ choosing what to run

@dataclass(frozen=True)
class Plan:
    task_ids: tuple[str, ...]
    trials: int
    max_cost: float
    model: str
    price_in: float
    price_out: float


def choose_tasks(
    spec: Sequence[str],
    *,
    max_tasks: int,
    gradeable: Mapping[str, str],
) -> tuple[str, ...]:
    """Resolve ``--tasks`` into task ids, or refuse. ``gradeable`` maps task id -> cluster."""
    if not spec:
        raise LiveRunRefused("--tasks is required: name task ids or one specialist")
    if max_tasks < 1 or max_tasks > HARD_MAX_TASKS:
        raise LiveRunRefused(f"--max-tasks must be between 1 and {HARD_MAX_TASKS}")
    if any(s.lower() in ("all", "*", "everything") for s in spec):
        raise LiveRunRefused("there is no 'all': name the tasks or one specialist")
    clusters = set(gradeable.values())
    chosen: list[str] = []
    for item in spec:
        if item in clusters:
            chosen += sorted(t for t, c in gradeable.items() if c == item)
        elif re.fullmatch(r"task_\d{3}", item):
            if item not in gradeable:
                raise LiveRunRefused(f"{item} cannot be run live: it is excluded from the offline gate or unknown")
            chosen.append(item)
        else:
            raise LiveRunRefused(f"'{item}' is not a task id like task_082 or a specialist ({', '.join(sorted(clusters))})")
    unique = tuple(dict.fromkeys(chosen))
    if len(unique) > max_tasks:
        raise LiveRunRefused(
            f"{len(unique)} tasks selected but --max-tasks is {max_tasks}: pick fewer, or raise the limit (at most {HARD_MAX_TASKS})"
        )
    return unique


def price_for(model: str, price_in: Optional[float], price_out: Optional[float]) -> tuple[float, float]:
    if price_in is not None and price_out is not None:
        return price_in, price_out
    if model in PRICES:
        return PRICES[model]
    raise LiveRunRefused(f"no price is known for '{model}': pass --price-in and --price-out so the cost cap means something")


@dataclass(frozen=True)
class Estimate:
    tasks: int
    uncached_tasks: int
    calls: int
    input_tokens: int
    output_tokens: int
    dollars: float


def estimate_task(segments: int, reads: int) -> tuple[int, int, int]:
    """(calls, input tokens, output tokens) for one task, roughly."""
    calls = reads + segments * _EXTRA_CALLS_PER_SEGMENT
    input_tokens = sum(_BASE_PROMPT_TOKENS + _GROWTH_PER_CALL * k for k in range(1, calls + 1))
    return calls, input_tokens, calls * _OUTPUT_PER_CALL


def estimate_run(
    shapes: Mapping[str, tuple[int, int]],
    cached: Sequence[str],
    *,
    trials: int,
    price_in: float,
    price_out: float,
) -> Estimate:
    calls = tokens_in = tokens_out = 0
    uncached = 0
    for task_id, (segments, reads) in shapes.items():
        if task_id in cached:
            continue
        uncached += 1
        c, i, o = estimate_task(segments, reads)
        calls, tokens_in, tokens_out = calls + c * trials, tokens_in + i * trials, tokens_out + o * trials
    dollars = (tokens_in * price_in + tokens_out * price_out) / 1_000_000
    return Estimate(len(shapes), uncached, calls, tokens_in, tokens_out, round(dollars, 4))


def check_estimate(estimate: Estimate, plan: Plan, *, confirmed: bool) -> None:
    if estimate.dollars > plan.max_cost:
        raise LiveRunRefused(
            f"estimated ${estimate.dollars:.2f} is over the cap of ${plan.max_cost:.2f}: run fewer tasks or raise --max-cost-usd"
        )
    if not confirmed and estimate.uncached_tasks:
        raise LiveRunRefused(
            f"this would spend about ${estimate.dollars:.2f} on {estimate.uncached_tasks} task(s) ({estimate.calls} model calls); "
            "re-run with --yes to go ahead"
        )


def render_estimate(estimate: Estimate, plan: Plan) -> str:
    return (
        f"{estimate.tasks} task(s), {estimate.uncached_tasks} not yet recorded, {plan.trials} trial(s), model {plan.model}\n"
        f"about {estimate.calls} model calls, {estimate.input_tokens:,} input and {estimate.output_tokens:,} output tokens\n"
        f"estimated cost ${estimate.dollars:.2f} (cap ${plan.max_cost:.2f}; prices are an assumption: "
        f"${plan.price_in}/M in, ${plan.price_out}/M out)"
    )


# ------------------------------------------------------------------ metering and cassettes

class MeteredClient:
    """Counts real usage and refuses the call that would go over the cap."""

    def __init__(self, inner: ToolCallingLLMClient, *, price_in: float, price_out: float, cap: float) -> None:
        self._inner = inner
        self._price = (price_in, price_out)
        self.cap = cap
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.seconds = 0.0

    @property
    def dollars(self) -> float:
        return (self.input_tokens * self._price[0] + self.output_tokens * self._price[1]) / 1_000_000

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[ToolSchema],
        tool_choice: Literal["auto", "required"] = "auto",
        temperature: float = 0.0,
    ) -> AssistantTurn:
        if self.dollars >= self.cap:
            raise BudgetExceeded(f"spent ${self.dollars:.4f} of the ${self.cap:.2f} cap")
        started = time.monotonic()
        turn = await self._inner.chat(messages, tools=tools, tool_choice=tool_choice, temperature=temperature)
        self.seconds += time.monotonic() - started
        self.calls += 1
        self.input_tokens += turn.usage.input_tokens
        self.output_tokens += turn.usage.output_tokens
        return turn


def request_hash(messages: Sequence[Mapping[str, Any]], tools: Sequence[ToolSchema], tool_choice: str) -> str:
    canonical = json.dumps(
        {"messages": [dict(m) for m in messages], "tools": [t.name for t in tools], "choice": tool_choice},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class CassetteClient:
    """Replays a recorded conversation if there is one; otherwise records the real one."""

    def __init__(self, inner: Optional[ToolCallingLLMClient], path: Path, *, rerecord: bool = False) -> None:
        self._inner = inner
        self._path = path
        self._records: list[dict[str, Any]] = []
        self._position = 0
        self.replaying = path.exists() and not rerecord
        if self.replaying:
            self._records = json.loads(path.read_text())

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[ToolSchema],
        tool_choice: Literal["auto", "required"] = "auto",
        temperature: float = 0.0,
    ) -> AssistantTurn:
        wanted = request_hash(messages, tools, tool_choice)
        if self.replaying:
            if self._position >= len(self._records):
                raise CassetteMismatch(f"{self._path.name} has only {len(self._records)} calls recorded; re-record with --rerecord")
            record = self._records[self._position]
            self._position += 1
            if record["request"] != wanted:
                raise CassetteMismatch(f"{self._path.name} was recorded for a different prompt; re-record with --rerecord")
            return AssistantTurn.model_validate(record["turn"])
        if self._inner is None:
            raise CassetteMismatch("no recording and no live client")
        turn = await self._inner.chat(messages, tools=tools, tool_choice=tool_choice, temperature=temperature)
        self._records.append({"request": wanted, "turn": turn.model_dump(mode="json")})
        return turn

    def save(self) -> None:
        """Write a new recording (atomically, so a crash never leaves half a conversation)."""
        if self.replaying or not self._records:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        scratch = self._path.with_suffix(".tmp")
        scratch.write_text(json.dumps(self._records, indent=1))
        scratch.replace(self._path)


def cassette_path(directory: Path, task_id: str, segment: int, model: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    return directory / f"{task_id}.{segment}.{slug}.json"


# ------------------------------------------------------------------ the command line

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent.live", description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", nargs="+", required=True, help="task ids (task_082) or a specialist name; there is no 'all'")
    parser.add_argument("--max-tasks", type=int, default=DEFAULT_MAX_TASKS)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST)
    parser.add_argument("--model", default=None, help="defaults to AGENT_MODEL")
    parser.add_argument("--price-in", type=float, default=None)
    parser.add_argument("--price-out", type=float, default=None)
    parser.add_argument("--cassettes", type=Path, default=DEFAULT_CASSETTES)
    parser.add_argument("--rerecord", action="store_true", help="pay for a fresh recording of tasks that already have one")
    parser.add_argument("--yes", action="store_true", help="spend the estimate")
    parser.add_argument("--out", type=Path, default=None)
    return parser


@dataclass(frozen=True)
class LiveTaskResult:
    task_id: str
    passed: bool
    strict_equal: bool
    calls: int
    input_tokens: int
    output_tokens: int
    dollars: float
    seconds: float
    replayed: bool
    note: str = ""


@dataclass
class LiveReport:
    results: list[LiveTaskResult] = field(default_factory=list)

    @property
    def dollars(self) -> float:
        return sum(r.dollars for r in self.results)

    def render(self) -> str:
        passed = sum(r.passed for r in self.results)
        lines = [f"{len(self.results)} task(s), {passed} passed, ${self.dollars:.4f} spent"]
        for r in self.results:
            how = "replayed" if r.replayed else f"{r.calls} calls, {r.input_tokens + r.output_tokens} tokens, ${r.dollars:.4f}, {r.seconds:.1f}s"
            lines.append(f"- {r.task_id}: {'PASS' if r.passed else 'FAIL'}{' (strict)' if r.passed and r.strict_equal else ''} [{how}]{' ' + r.note if r.note else ''}")
        return "\n".join(lines)


ClientFactory = Callable[[str], ToolCallingLLMClient]
TaskRunner = Callable[[str, Path, "RunContext"], LiveTaskResult]


@dataclass
class RunContext:
    plan: Plan
    make_client: ClientFactory
    rerecord: bool
    meter: Optional[MeteredClient] = None


def prepare(
    argv: Sequence[str],
    *,
    gradeable: Mapping[str, str],
    shapes: Mapping[str, tuple[int, int]],
    default_model: str,
) -> tuple[Plan, Estimate, argparse.Namespace]:
    """Everything decided before a single request: parse, choose, price, estimate, check."""
    args = build_parser().parse_args(list(argv))
    if args.trials < 1 or args.trials > HARD_MAX_TRIALS:
        raise LiveRunRefused(f"--trials must be between 1 and {HARD_MAX_TRIALS}")
    if args.max_cost_usd <= 0:
        raise LiveRunRefused("--max-cost-usd must be above zero")
    chosen = choose_tasks(args.tasks, max_tasks=args.max_tasks, gradeable=gradeable)
    model = args.model or default_model
    price_in, price_out = price_for(model, args.price_in, args.price_out)
    plan = Plan(chosen, args.trials, args.max_cost_usd, model, price_in, price_out)
    cached = [
        t for t in chosen
        if not args.rerecord and any(cassette_path(args.cassettes, t, n, model).exists() for n in range(8))
    ]
    estimate = estimate_run({t: shapes[t] for t in chosen}, cached, trials=args.trials, price_in=price_in, price_out=price_out)
    check_estimate(estimate, plan, confirmed=args.yes)
    return plan, estimate, args


def run_plan(plan: Plan, args: argparse.Namespace, make_client: ClientFactory, run_task: TaskRunner) -> LiveReport:
    """Run the chosen tasks one after another, metered, stopping cleanly when the cap is hit."""
    report = LiveReport()
    context = RunContext(plan=plan, make_client=make_client, rerecord=args.rerecord)
    spent = 0.0
    for task_id in plan.task_ids:
        for _ in range(plan.trials):
            if spent >= plan.max_cost:
                report.results.append(LiveTaskResult(task_id, False, False, 0, 0, 0, 0.0, 0.0, False, "skipped: cost cap reached"))
                continue
            result = run_task(task_id, args.cassettes, context)
            spent += result.dollars
            report.results.append(result)
    return report


# ------------------------------------------------------------------ running a task for real

def gold_secrets(task: Mapping[str, Any]) -> dict[str, str]:
    """Every PIN or security code the gold uses, named pin_a, pin_b ... so each gets its own reference."""
    from agent.tier1 import gold_steps
    from banking_tools.session import REDACTED_ARGUMENTS

    found: dict[str, str] = {}
    for step in gold_steps(task):
        for key in sorted(REDACTED_ARGUMENTS & set(step.args)):
            value = str(step.args[key])
            if value and value not in found.values():
                found[f"{key.replace('new_', '').replace('current_', 'old_')}_{chr(ord('a') + len(found))}"] = value
    return found


def segment_request(task_id: str, number: int) -> Optional[str]:
    """What the customer says for one segment whose request the script only reveals after the
    agent probes (evals/segment_requests.json, hand-written from the persona's own lines).

    When there is one it is the whole request. The persona's full script mentions every part of
    the conversation, so handing it over with a single segment tempts the model to do the other
    parts early or on the wrong card."""
    path = Path(__file__).resolve().parents[2] / "evals" / "segment_requests.json"
    try:
        found = json.loads(path.read_text()).get(task_id, {}).get(str(number))
    except (OSError, ValueError):
        return None
    return str(found) if found else None


def customer_facts(task: Mapping[str, Any]) -> str:
    """What the simulated customer would say if asked, so the single-shot run is not stuck on a question.

    A live conversation would answer the specialist's questions as the persona does. This
    harness has no conversation, so the persona's own notes (who they are, what they want,
    what they give when asked) go in with the opening request. The specialist is still the one
    that decides what to do about it.
    """
    from knowledge.report import opening_message

    notes = str((task.get("user_scenario") or {}).get("instructions") or "").strip()
    return f"{opening_message(task)}\n\nWhat the customer knows and will tell you if you ask:\n{notes}"


# A task that also pays a card from checking is still the credit specialist's: that payment is replayed
# directly, as in the offline replay (the accounts specialist is not built yet).
ALSO_REPLAYED_DIRECTLY: Mapping[str, frozenset[str]] = {"credit": frozenset({"accounts"})}


def gradeable_clusters(configs: Mapping[str, Any]) -> dict[str, str]:
    """task id -> specialist, for the tasks a built specialist can be graded on offline."""
    from banking_tools.manifest import build_manifest, load_tasks

    found: dict[str, str] = {}
    for entry in build_manifest(load_tasks()):
        if entry.excluded_reason is not None:
            continue
        if entry.category == "wholly_one_cluster" and entry.clusters[0] in configs:
            found[entry.task_id] = entry.clusters[0]
        elif entry.category == "spans_clusters":
            for name in configs:
                if name in entry.clusters and set(entry.clusters) - {name} <= ALSO_REPLAYED_DIRECTLY.get(name, frozenset()):
                    found[entry.task_id] = name
    return found


def task_shape(task: Mapping[str, Any], config: Any) -> tuple[int, int]:
    """(segments, reads) in a task, for the estimate."""
    from agent.tier1 import Segment, gold_steps, plan_items

    items = plan_items(gold_steps(task), config)
    segments = [i for i in items if isinstance(i, Segment)]
    return len(segments), sum(len(i.reads) for i in segments)


def run_live_task(
    task: Mapping[str, Any],
    config: Any,
    cassettes: Path,
    context: RunContext,
    *,
    index: Any,
) -> LiveTaskResult:
    """One task, with the model deciding each segment's plan. Costs money unless replayed."""
    from tau2.domains.banking_knowledge.tools import KnowledgeUserTools

    from agent.tier1 import (
        Direct,
        SegmentError,
        gold_steps,
        invalid_gold,
        plan_and_execute,
        plan_items,
        slots_of,
    )
    from agent.specialist import Specialist
    from agent.toolbox import SecretVault
    from banking_tools.gold_replay import read_log_allowlist, replay_on_tau2
    from banking_tools.session import BankingSession, Fidelity, _base_db
    from banking_tools.statehash import compare
    from banking_tools.task_state import apply_initial_state
    from orchestrator.types import Intent

    task_id = str(task["id"])
    plan = context.plan
    steps = gold_steps(task)
    if (broken := invalid_gold(steps)) is not None:
        return LiveTaskResult(task_id, False, False, 0, 0, 0, 0.0, 0.0, False, f"excluded: {broken}")
    user_id = next(str(s.args["user_id"]) for s in steps if s.kind == "verify")
    session = BankingSession.from_task(task, fidelity=Fidelity.TAU2, read_log_allowlist=read_log_allowlist(task))
    start = apply_initial_state(_base_db().model_copy(deep=True), task)
    expected, _ = replay_on_tau2(task)
    customer_tools = KnowledgeUserTools(session.database)
    vault = SecretVault()
    meter: Optional[MeteredClient] = None
    replayed = True
    note = ""

    def direct(step: Any) -> None:
        if step.kind == "user":
            getattr(customer_tools, step.name)(**step.args)
        elif step.kind == "other":
            getattr(session.toolkit, step.name)(**step.args)
        else:
            session.call(step.name, step.args)

    number = 0
    try:
        for item in plan_items(steps, config):
            if isinstance(item, Direct):
                direct(item.step)
                continue
            path = cassette_path(cassettes, task_id, number, plan.model)
            fresh = context.rerecord or not path.exists()
            if fresh and meter is None:
                meter = MeteredClient(context.make_client(plan.model), price_in=plan.price_in, price_out=plan.price_out, cap=plan.max_cost)
            replayed = replayed and not fresh
            client = CassetteClient(meter if fresh else None, path, rerecord=context.rerecord)
            specialist = Specialist(config, llm=client, index=index, database=session.database, vault=vault)
            intent = Intent(
                id=f"{task_id}-{number}",
                kind=item.kind,
                summary=vault.capture(
                    vault.hide(
                        segment_request(task_id, number) or customer_facts(task),
                        gold_secrets(task),
                    )
                ),
                slots=slots_of(item.writes[0]),
            )
            try:
                plan_and_execute(specialist, intent, session, user_id, label=f"segment {number} ({item.kind.value})")
            finally:
                client.save()  # a failed segment is the one worth reading, and was paid for
            number += 1
    except (SegmentError, BudgetExceeded, CassetteMismatch) as exc:
        note = f"{type(exc).__name__}: {exc}"
        # The agent loop turns a failed model call into a quiet stop, so say when the cause
        # was the cap: the meter refused the next call.
        if meter is not None and meter.dollars >= meter.cap:
            note = f"cost cap of ${meter.cap:.2f} reached after ${meter.dollars:.2f}; {note}"
        return LiveTaskResult(
            task_id, False, False,
            meter.calls if meter else 0, meter.input_tokens if meter else 0, meter.output_tokens if meter else 0,
            meter.dollars if meter else 0.0, meter.seconds if meter else 0.0, replayed, note[:300],
        )
    outcome = compare(start, expected, session.database, read_log_names=read_log_allowlist(task))
    return LiveTaskResult(
        task_id, outcome.semantic_equal, outcome.strict_equal,
        meter.calls if meter else 0, meter.input_tokens if meter else 0, meter.output_tokens if meter else 0,
        meter.dollars if meter else 0.0, meter.seconds if meter else 0.0, replayed,
        "" if outcome.semantic_equal else f"{len(outcome.mismatches)} differences",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - the entry point; its parts are tested
    import sys

    from agent.llm_tools import LiteLLMToolClient
    from agent.specialists.accounts import ACCOUNTS_SPECIALIST
    from agent.specialists.credit import CREDIT_SPECIALIST
    from agent.specialists.debit import DEBIT_SPECIALIST
    from agent.specialists.fees import FEES_SPECIALIST
    from banking_tools.manifest import load_tasks
    from knowledge.index import get_index
    from orchestrator.config import agent_model

    configs = {"debit": DEBIT_SPECIALIST, "credit": CREDIT_SPECIALIST, "accounts": ACCOUNTS_SPECIALIST, "fees": FEES_SPECIALIST}
    tasks = {str(t["id"]): t for t in load_tasks()}
    gradeable = gradeable_clusters(configs)
    shapes = {t: task_shape(tasks[t], configs[c]) for t, c in gradeable.items()}
    try:
        plan, estimate, args = prepare(argv if argv is not None else sys.argv[1:], gradeable=gradeable, shapes=shapes, default_model=agent_model())
    except LiveRunRefused as refusal:
        print(f"not started: {refusal}")
        return 2
    print(render_estimate(estimate, plan))
    index = get_index()

    def run_task(task_id: str, cassettes: Path, context: RunContext) -> LiveTaskResult:
        return run_live_task(tasks[task_id], configs[gradeable[task_id]], cassettes, context, index=index)

    from agent import tier1

    loop = asyncio.new_event_loop()  # one loop for the whole run: the model client keeps its connection on it
    tier1.use_loop(loop)
    try:
        report = run_plan(plan, args, lambda model: LiteLLMToolClient(model), run_task)
    finally:
        tier1.use_loop(None)
        loop.close()
    print(report.render())
    if args.out:
        args.out.write_text(json.dumps([r.__dict__ for r in report.results], indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
