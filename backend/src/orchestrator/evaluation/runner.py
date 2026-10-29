"""Run every case several times and score each trial.

A trial passes when the action kinds the decomposer found equal the ones the
message states exactly (``EvalCase.stated``). Read-only kinds and OTHER are ignored on the predicted side, as
the stated side never contains them. If the decomposer fails and hands back a
single OTHER intent, nothing is predicted, so the trial fails.
"""

import asyncio
from collections.abc import Callable, Sequence
from statistics import mean

from pydantic import BaseModel, ConfigDict

from orchestrator.decomposer import Decomposer
from orchestrator.evaluation.metrics import pass_at_k, pass_hat_k
from orchestrator.evaluation.tasks import EvalCase
from orchestrator.plan import READ_ONLY_KINDS
from orchestrator.types import IntentKind

DEFAULT_TRIALS = 3
DEFAULT_CONCURRENCY = 8


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class Trial(_Model):
    task_id: str
    trial: int
    # Matches what the message asks for (graded).
    passed: bool
    # Matches what tau2's policy-driven expected actions imply (reported only).
    policy_passed: bool
    predicted: tuple[IntentKind, ...]
    missed: tuple[IntentKind, ...]
    extra: tuple[IntentKind, ...]
    intent_count: int


class Summary(_Model):
    tasks: int
    trials: int
    # Index k-1 holds the value for k, averaged over tasks.
    pass_at_k: tuple[float, ...]
    pass_hat_k: tuple[float, ...]
    mean_trial_pass_rate: float
    policy_pass_rate: float
    # Share of tasks whose intent count was the same in every trial.
    intent_count_stable: float


async def run_trial(decomposer: Decomposer, case: EvalCase, trial: int) -> Trial:
    interpretation = await decomposer.interpret(case.message, ())
    intents = interpretation.decomposition.intents
    predicted = {i.kind for i in intents} - READ_ONLY_KINDS - {IntentKind.OTHER}
    expected = set(case.stated)
    return Trial(
        task_id=case.task_id,
        trial=trial,
        passed=predicted == expected,
        policy_passed=predicted == set(case.expected),
        predicted=tuple(sorted(predicted)),
        missed=tuple(sorted(expected - predicted)),
        extra=tuple(sorted(predicted - expected)),
        intent_count=len(intents),
    )


async def run_eval(
    cases: Sequence[EvalCase],
    make_decomposer: Callable[[], Decomposer],
    *,
    trials: int = DEFAULT_TRIALS,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[Trial]:
    """Every trial gets a fresh decomposer, so no state carries between trials."""
    gate = asyncio.Semaphore(concurrency)

    async def one(case: EvalCase, trial: int) -> Trial:
        async with gate:
            return await run_trial(make_decomposer(), case, trial)

    return list(
        await asyncio.gather(*(one(c, t) for c in cases for t in range(1, trials + 1)))
    )


def summarize(results: Sequence[Trial], trials: int) -> Summary:
    by_task: dict[str, list[Trial]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)
    passes = [sum(t.passed for t in ts) for ts in by_task.values()]
    return Summary(
        tasks=len(by_task),
        trials=trials,
        pass_at_k=tuple(mean(pass_at_k(trials, c, k) for c in passes) for k in range(1, trials + 1)),
        pass_hat_k=tuple(mean(pass_hat_k(trials, c, k) for c in passes) for k in range(1, trials + 1)),
        mean_trial_pass_rate=mean(c / trials for c in passes),
        policy_pass_rate=mean(r.policy_passed for r in results),
        intent_count_stable=mean(
            len({t.intent_count for t in ts}) == 1 for ts in by_task.values()
        ),
    )


def render(summary: Summary, results: Sequence[Trial]) -> str:
    lines = [f"{summary.tasks} tasks x {summary.trials} trials", ""]
    for n in range(1, summary.trials + 1):
        lines.append(
            f"k={n}  pass@{n} = {summary.pass_at_k[n - 1]:.3f}   pass^{n} = {summary.pass_hat_k[n - 1]:.3f}"
        )
    lines += [
        "",
        f"mean per-trial pass rate: {summary.mean_trial_pass_rate:.3f}",
        f"trials matching tau2's policy-driven actions instead: {summary.policy_pass_rate:.3f}",
        f"intent count identical across trials: {summary.intent_count_stable:.1%} of tasks",
    ]
    missed: dict[IntentKind, int] = {}
    extra: dict[IntentKind, int] = {}
    for r in results:
        for kind in r.missed:
            missed[kind] = missed.get(kind, 0) + 1
        for kind in r.extra:
            extra[kind] = extra.get(kind, 0) + 1
    for title, counts in (("most often missed", missed), ("most often added", extra)):
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        if top:
            lines.append(f"{title}: " + ", ".join(f"{k} x{n}" for k, n in top))
    return "\n".join(lines)
