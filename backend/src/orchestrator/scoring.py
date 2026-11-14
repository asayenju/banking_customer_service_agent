"""How sure we are about each intent, from a swappable scorer.

A scorer (the Jev adapter later, a scripted fake now) answers three separate
questions per intent: is the kind right, are the slots complete, is it in
scope. The intent's confidence is the weakest of the three, so one doubtful
answer cannot hide behind two confident ones.

Scoring never raises. A timeout, an error or a malformed answer becomes a score
of zero with the reason attached, which the bands turn into an escalation.
Only the exception's type is recorded: SDK errors can echo the request, and the
request holds the customer's message.
"""

import asyncio
import inspect
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Optional, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field

from orchestrator.types import Decomposition

SCORER_TIMEOUT_SECONDS = 5.0

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class IntentScore(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    intent_id: Annotated[str, Field(min_length=1)]
    kind_confidence: Probability
    slots_confidence: Probability
    scope_confidence: Probability
    # Set when the intent could not be scored; the confidence is then zero.
    error: Optional[str] = None
    # The kind the scorer found next most likely (a Choice answer) when it is a different reading, not a sibling that
    # routes the same way. Used to ask "X or Y?" instead of a vague "tell me more".
    alternative: Optional[str] = None
    # Extra signals asked beside the kind (name, probability or score). Logged, never used to decide.
    signals: tuple[tuple[str, float], ...] = ()

    @property
    def confidence(self) -> float:
        if self.error is not None:
            return 0.0
        return min(self.kind_confidence, self.slots_confidence, self.scope_confidence)

    @classmethod
    def flat(cls, intent_id: str, confidence: float) -> Self:
        return cls(
            intent_id=intent_id,
            kind_confidence=confidence,
            slots_confidence=confidence,
            scope_confidence=confidence,
        )

    @classmethod
    def unscored(cls, intent_id: str, error: str) -> Self:
        return cls(
            intent_id=intent_id,
            kind_confidence=0.0,
            slots_confidence=0.0,
            scope_confidence=0.0,
            error=error,
        )


@dataclass(frozen=True)
class ScoreContext:
    """What a scorer may know besides the message: the last turns of the conversation, and the products the verified
    customer holds (types only: no ids, balances or amounts). Passed only to scorers that ask for it."""

    recent: tuple[tuple[str, str], ...] = ()
    products: str = ""
    # Facts from application state, as types and flags only ("verified": "yes"), never a value, id, amount or digit.
    facts: tuple[tuple[str, str], ...] = ()


class Scorer(Protocol):
    async def score(
        self, message: str, decomposition: Decomposition
    ) -> Sequence[IntentScore]: ...


class ScriptedScorer:
    """Returns the scores it was given, or fails or stalls on demand."""

    def __init__(
        self,
        scores: Iterable[IntentScore] = (),
        *,
        error: Optional[Exception] = None,
        delay: float = 0.0,
    ) -> None:
        self._scores = tuple(scores)
        self._error = error
        self._delay = delay

    async def score(
        self, message: str, decomposition: Decomposition
    ) -> Sequence[IntentScore]:
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return self._scores


def _call(scorer: Scorer, message: str, decomposition: Decomposition, context: Optional[ScoreContext]) -> Any:
    """Hand the context to a scorer that takes one; every other scorer is called as before."""
    if context is not None and "context" in inspect.signature(scorer.score).parameters:
        return scorer.score(message, decomposition, context=context)  # type: ignore[call-arg]
    return scorer.score(message, decomposition)


async def score_intents(
    scorer: Scorer,
    message: str,
    decomposition: Decomposition,
    *,
    timeout: float = SCORER_TIMEOUT_SECONDS,
    context: Optional[ScoreContext] = None,
) -> tuple[IntentScore, ...]:
    """One score per intent, in the decomposition's order, whatever the scorer does."""
    ids = [intent.id for intent in decomposition.intents]
    if not ids:
        return ()
    try:
        returned = tuple(
            await asyncio.wait_for(_call(scorer, message, decomposition, context), timeout)
        )
        if not all(isinstance(s, IntentScore) for s in returned):
            raise TypeError("scorer returned something other than IntentScore")
    except TimeoutError:
        error = f"scorer timed out after {timeout}s"
        return tuple(IntentScore.unscored(i, error) for i in ids)
    except Exception as exc:
        error = f"scorer failed: {type(exc).__name__}"
        return tuple(IntentScore.unscored(i, error) for i in ids)

    by_id: dict[str, list[IntentScore]] = {}
    for s in returned:
        by_id.setdefault(s.intent_id, []).append(s)

    result: list[IntentScore] = []
    for intent_id in ids:
        found = by_id.get(intent_id, [])
        if len(found) == 1:
            result.append(found[0])
        elif not found:
            result.append(
                IntentScore.unscored(intent_id, "scorer returned no score for this intent")
            )
        else:
            result.append(
                IntentScore.unscored(
                    intent_id, "scorer returned more than one score for this intent"
                )
            )
    return tuple(result)
