"""Run every case several times and score each trial.

A trial passes when the action kinds the decomposer found equal the ones the
message states exactly (``EvalCase.stated``). Read-only kinds and OTHER are ignored on the predicted side, as
the stated side never contains them. If the decomposer fails and hands back a
single OTHER intent, nothing is predicted, so the trial fails.
"""

import asyncio
from collections.abc import Callable, Sequence
from statistics import mean
from typing import Optional

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
    # What the message asks for; empty when it asks the bank for nothing.
    stated: tuple[IntentKind, ...]
    # Matches what the message asks for (graded).
    passed: bool
    # Matches what tau2's policy-driven expected actions imply (reported only).
    policy_passed: bool
    predicted: tuple[IntentKind, ...]
    missed: tuple[IntentKind, ...]
    extra: tuple[IntentKind, ...]
    intent_count: int


class KindScore(_Model):
    kind: IntentKind
    true_positive: int
    false_positive: int
    false_negative: int


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
    # A dropped request goes unanswered; an added action may change an account.
    # They are different failures, so they are counted apart (a trial can be both).
    trials_with_missed: float
    trials_with_extra: float
    # On messages that ask the bank for nothing: how often a write was invented.
    # None when the set has no such message.
    no_action_tasks: int
    false_write_rate: Optional[float]
    # Over every action kind together; None when nothing was predicted or stated.
    write_precision: Optional[float]
    write_recall: Optional[float]
    per_kind: tuple[KindScore, ...]


async def run_trial(decomposer: Decomposer, case: EvalCase, trial: int) -> Trial:
    interpretation = await decomposer.interpret(case.message, ())
    intents = interpretation.decomposition.intents
    predicted = {i.kind for i in intents} - READ_ONLY_KINDS - {IntentKind.OTHER}
    expected = set(case.stated)
    return Trial(
        task_id=case.task_id,
        trial=trial,
        stated=tuple(sorted(expected)),
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


def _per_kind(results: Sequence[Trial]) -> tuple[KindScore, ...]:
    tp: dict[IntentKind, int] = {}
    fp: dict[IntentKind, int] = {}
    fn: dict[IntentKind, int] = {}
    for r in results:
        for kind in r.predicted:
            counts = fp if kind in r.extra else tp
            counts[kind] = counts.get(kind, 0) + 1
        for kind in r.missed:
            fn[kind] = fn.get(kind, 0) + 1
    kinds = sorted(set(tp) | set(fp) | set(fn))
    return tuple(
        KindScore(
            kind=k,
            true_positive=tp.get(k, 0),
            false_positive=fp.get(k, 0),
            false_negative=fn.get(k, 0),
        )
        for k in kinds
    )


def summarize(results: Sequence[Trial], trials: int) -> Summary:
    by_task: dict[str, list[Trial]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)
    passes = [sum(t.passed for t in ts) for ts in by_task.values()]
    per_kind = _per_kind(results)
    tp = sum(k.true_positive for k in per_kind)
    fp = sum(k.false_positive for k in per_kind)
    fn = sum(k.false_negative for k in per_kind)
    quiet = [r for r in results if not r.stated]
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
        trials_with_missed=mean(bool(r.missed) for r in results),
        trials_with_extra=mean(bool(r.extra) for r in results),
        no_action_tasks=len({r.task_id for r in quiet}),
        false_write_rate=mean(bool(r.extra) for r in quiet) if quiet else None,
        write_precision=tp / (tp + fp) if tp + fp else None,
        write_recall=tp / (tp + fn) if tp + fn else None,
        per_kind=per_kind,
    )


def _optional(label: str, value: Optional[float]) -> str:
    return f"{label}: {value:.1%}" if value is not None else f"{label}: n/a"


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
        f"trials that dropped a request: {summary.trials_with_missed:.1%}",
        f"trials that added an action: {summary.trials_with_extra:.1%}",
        _optional(
            "write precision", summary.write_precision
        )
        + "   "
        + _optional("write recall", summary.write_recall),
        _optional(
            f"false write rate on {summary.no_action_tasks} no-action messages",
            summary.false_write_rate,
        ),
        "",
        "kind                          precision  recall  (tp/fp/fn)",
    ]
    for k in summary.per_kind:
        called = k.true_positive + k.false_positive
        wanted = k.true_positive + k.false_negative
        precision = f"{k.true_positive / called:.2f}" if called else "  - "
        recall = f"{k.true_positive / wanted:.2f}" if wanted else "  - "
        lines.append(
            f"{k.kind:<30}{precision:>9}  {recall:>6}  "
            f"({k.true_positive}/{k.false_positive}/{k.false_negative})"
        )
    lines.append("")
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
