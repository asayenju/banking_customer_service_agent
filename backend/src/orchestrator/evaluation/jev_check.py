"""Does Jev's confidence tell a right intent from a wrong one?

Every action intent the decomposer produces is scored by Jev and compared with the
labelled answer: right when its kind is among the kinds the message states. This
tells us whether a Jev confidence gate (and a retry on a low score) could catch
real mistakes, and at what threshold. Read-only kinds are not graded, so they are
not counted.
"""

import asyncio
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from orchestrator.decomposer import Decomposer
from orchestrator.evaluation.tasks import EvalCase
from orchestrator.plan import READ_ONLY_KINDS
from orchestrator.scoring import Scorer
from orchestrator.types import IntentKind

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98)


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    task_id: str
    kind: IntentKind
    confidence: float
    kind_confidence: float
    error: bool
    correct: bool


async def observe(
    cases: Sequence[EvalCase], decomposer: Decomposer, scorer: Scorer, *, concurrency: int = 8
) -> list[Observation]:
    gate = asyncio.Semaphore(concurrency)

    async def one(case: EvalCase) -> list[Observation]:
        async with gate:
            decomposition = (await decomposer.interpret(case.message, ())).decomposition
            scores = {s.intent_id: s for s in await scorer.score(case.message, decomposition)}
        return [
            Observation(
                task_id=case.task_id,
                kind=i.kind,
                confidence=scores[i.id].confidence,
                kind_confidence=scores[i.id].kind_confidence,
                error=scores[i.id].error is not None,
                correct=i.kind in case.stated,
            )
            for i in decomposition.intents
            if i.kind not in READ_ONLY_KINDS and i.kind is not IntentKind.OTHER and i.id in scores
        ]

    return [o for rows in await asyncio.gather(*(one(c) for c in cases)) for o in rows]


def auroc(wrong: Sequence[float], right: Sequence[float]) -> float:
    """Chance that a random wrong intent scores lower than a random right one (0.5 = no signal)."""
    if not wrong or not right:
        return float("nan")
    total = sum(1.0 if w < r else 0.5 if w == r else 0.0 for w in wrong for r in right)
    return total / (len(wrong) * len(right))


def render(observations: Sequence[Observation]) -> str:
    usable = [o for o in observations if not o.error]
    right = [o for o in usable if o.correct]
    wrong = [o for o in usable if not o.correct]
    lines = [
        f"{len(observations)} action intents: {len(right)} right, {len(wrong)} wrong, "
        f"{len(observations) - len(usable)} Jev errors",
    ]
    pickers: tuple[tuple[str, Callable[[Observation], float]], ...] = (
        ("overall confidence", lambda o: o.confidence),
        ("kind question only", lambda o: o.kind_confidence),
    )
    for label, pick in pickers:
        lines.append(
            f"AUROC ({label}): {auroc([pick(o) for o in wrong], [pick(o) for o in right]):.3f}"
            f"   mean right {sum(map(pick, right)) / max(len(right), 1):.2f}"
            f"   mean wrong {sum(map(pick, wrong)) / max(len(wrong), 1):.2f}"
        )
    lines.append("")
    lines.append("threshold on the kind question  (overall confidence is dominated by missing slots)")
    lines.append("threshold  wrong caught   right retried")
    for t in THRESHOLDS:
        caught = sum(o.kind_confidence < t for o in wrong) / max(len(wrong), 1)
        retried = sum(o.kind_confidence < t for o in right) / max(len(right), 1)
        lines.append(f"{t:>9.2f}  {caught:>11.0%}   {retried:>12.0%}")
    return "\n".join(lines)
