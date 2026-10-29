"""Jev (TypeSafe AI) as the per-intent ``Scorer``.

Jev answers yes/no questions with a probability and writes no text. Per intent
we ask two: is the kind right, and is the request something a bank's customer
service handles. The third score, slots, is plain code: complete exactly when
nothing is missing, because Jev is weak on numbers and dates. An intent's
confidence is still the weakest of the three (see ``scoring.IntentScore``).

Checks that need no model run first. A message flagged as a possible injection
is not sent at all (Jev is easily steered by adversarial text), so its intents
score zero and escalate. Only the redacted message, the kind and the summary go
out; slot values never do.

High confidence changes nothing about safety: ``decision`` still requires the
customer's explicit yes for irreversible actions and for unresolved slots.
The model is pinned; ``jev-latest`` moves under us. The SDK is an optional
extra (``uv sync --extra jev``) and is imported only by ``make_jev_scorer``.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from orchestrator.config import JEV_KEY_VAR, require
from orchestrator.sanitize import redact
from orchestrator.scoring import IntentScore
from orchestrator.types import Decomposition, Flag, Intent

JEV_MODEL = "jev-1.13.0"
JEV_TIMEOUT_SECONDS = 4.0

KIND_QUESTION = "kind"
SCOPE_QUESTION = "scope"


class _NoulAnswer(Protocol):
    @property
    def noul(self) -> float: ...


class _Response(Protocol):
    @property
    def nouls(self) -> Mapping[str, _NoulAnswer]: ...


class JevClient(Protocol):
    """The slice of ``typesafe_sdk.AsyncTypeSafeClient`` we use."""

    async def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> _Response: ...


def _words(intent: Intent) -> str:
    return intent.kind.value.lower().replace("_", " ")


def _state(message: str, intent: Intent) -> dict[str, str]:
    return {
        "customer_message": redact(message),
        "request_kind": _words(intent),
        "request_summary": redact(intent.summary),
    }


def _questions(intent: Intent) -> dict[str, dict[str, str]]:
    return {
        KIND_QUESTION: {
            "type": "noul",
            "instructions": (
                f"Does the customer's message contain a request that fits the category "
                f"'{_words(intent)}' ({redact(intent.summary)})?"
            ),
        },
        SCOPE_QUESTION: {
            "type": "noul",
            "instructions": (
                "Is this a request that a retail bank's customer service handles "
                "(cards, accounts, payments, fees, disputes, profile or product questions)?"
            ),
        },
    }


class JevScorer:
    def __init__(
        self,
        client: JevClient,
        *,
        model: str = JEV_MODEL,
        timeout: float = JEV_TIMEOUT_SECONDS,
    ) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout

    async def score(
        self, message: str, decomposition: Decomposition
    ) -> Sequence[IntentScore]:
        if Flag.INJECTION_SUSPECTED in decomposition.flags:
            return tuple(
                IntentScore.unscored(i.id, "not scored: possible prompt injection")
                for i in decomposition.intents
            )
        results = await asyncio.gather(
            *(self._score_one(message, i) for i in decomposition.intents),
            return_exceptions=True,
        )
        scores: list[IntentScore] = []
        for intent, result in zip(decomposition.intents, results, strict=True):
            if isinstance(result, BaseException):
                # Only the type: SDK errors can echo the request, which holds the message.
                scores.append(IntentScore.unscored(intent.id, f"jev failed: {type(result).__name__}"))
            else:
                scores.append(result)
        return tuple(scores)

    async def _score_one(self, message: str, intent: Intent) -> IntentScore:
        response = await self._client.system_one(
            _state(message, intent),
            _questions(intent),
            model=self._model,
            timeout=self._timeout,
        )
        answers = response.nouls
        return IntentScore(
            intent_id=intent.id,
            kind_confidence=answers[KIND_QUESTION].noul,
            slots_confidence=0.0 if intent.missing_slots else 1.0,
            scope_confidence=answers[SCOPE_QUESTION].noul,
        )


def make_jev_scorer(*, model: str = JEV_MODEL) -> JevScorer:
    """A scorer on the real API, with the key from ``JEV_KEY``. Needs the ``jev`` extra."""
    from typesafe_sdk import AsyncTypeSafeClient

    client = AsyncTypeSafeClient(api_key=require(JEV_KEY_VAR), model=model)
    return JevScorer(client, model=model)
