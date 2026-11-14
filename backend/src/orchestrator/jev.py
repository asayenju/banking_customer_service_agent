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
from orchestrator.capabilities import capability
from orchestrator import jev_state
from orchestrator.scoring import IntentScore, ScoreContext
from orchestrator.types import Decomposition, Flag, Intent

JEV_MODEL = "jev-1.13.0"
JEV_TIMEOUT_SECONDS = 4.0

# How likely a different reading must be before it is offered as "X or Y?".
ALTERNATIVE_AT = 0.15
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


def _state(message: str, intent: Intent, context: ScoreContext | None = None) -> dict[str, str]:
    """Everything Jev is shown. The decomposer's finding (the kind, what it understood, the words it pointed to, and
    which details it has or lacks) and what the assistant can really do for that kind, so Jev judges the request against
    what is on offer. Never a detail's value (Jev is weak at numbers and dates, and the values are private)."""
    state = {
        "customer_message": redact(message),
        "request_kind": _words(intent),
        "request_summary": redact(intent.summary),
        "what_the_assistant_can_do_for_this_kind": capability(intent.kind),
    }
    if intent.source_span is not None:
        start, end = intent.source_span
        if 0 <= start < end <= len(message):
            state["customer_words_it_points_to"] = redact(message[start:end])
    if intent.slots or intent.missing_slots:
        state["details_found"] = ", ".join(sorted(intent.slots)) or "none"
        state["details_missing"] = ", ".join(intent.missing_slots) or "none"
    if context is not None:
        if context.recent:
            state["recent_conversation"] = " | ".join(f"{who}: {redact(said)[:300]}" for who, said in context.recent[-4:])
        if context.products:
            state["products_the_customer_holds"] = context.products
        if context.facts:
            state["situation"] = "; ".join(f"{name.replace('_', ' ')}: {value}" for name, value in context.facts)
    return state


def _questions(intent: Intent) -> dict[str, dict[str, str]]:
    return {
        KIND_QUESTION: {
            "type": "noul",
            "instructions": (
                f"The assistant read the customer's message (with the recent conversation, if any) as the category "
                f"'{_words(intent)}': {redact(intent.summary)}. Given what the assistant can do for that category, "
                f"is that what the customer is asking for?"
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
        choice: bool | None = None,
        signals: bool | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout
        self._choice = jev_state.choice_enabled() if choice is None else choice
        self._signals = jev_state.signals_enabled() if signals is None else signals

    async def score(
        self, message: str, decomposition: Decomposition, context: ScoreContext | None = None
    ) -> Sequence[IntentScore]:
        if Flag.INJECTION_SUSPECTED in decomposition.flags:
            return tuple(
                IntentScore.unscored(i.id, "not scored: possible prompt injection")
                for i in decomposition.intents
            )
        results = await asyncio.gather(
            *(self._score_one(message, i, context) for i in decomposition.intents),
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

    async def _score_one(self, message: str, intent: Intent, context: ScoreContext | None = None) -> IntentScore:
        questions: dict[str, Any] = dict(_questions(intent))
        if self._choice:
            questions[jev_state.READING] = jev_state.choice_question()
        if self._signals:
            questions.update(jev_state.signal_questions())
        response = await self._client.system_one(
            _state(message, intent, context),
            questions,
            model=self._model,
            timeout=self._timeout,
        )
        answers = response.nouls
        kind = answers[KIND_QUESTION].noul
        alternative: str | None = None
        if self._choice:
            kind, alternative = self._reading(response, intent, kind)
        return IntentScore(
            intent_id=intent.id,
            kind_confidence=kind,
            slots_confidence=0.0 if intent.missing_slots else 1.0,
            scope_confidence=answers[SCOPE_QUESTION].noul,
            alternative=alternative,
            signals=self._signal_values(response) if self._signals else (),
        )

    @staticmethod
    def _reading(response: Any, intent: Intent, kind: float) -> tuple[float, str | None]:
        """The kind answer, raised to the weight of kinds that run the same way, and the next different reading.
        Any trouble with the extra answer leaves the plain kind answer as it was."""
        try:
            probabilities = dict(response.choices[jev_state.READING].probabilities)
            mass, other = jev_state.runner_up(intent.kind, probabilities)
        except Exception:
            return kind, None
        weight = probabilities.get(other, 0.0) if other else 0.0
        return max(kind, mass), (other if other and weight >= ALTERNATIVE_AT else None)

    @staticmethod
    def _signal_values(response: Any) -> tuple[tuple[str, float], ...]:
        found: list[tuple[str, float]] = []
        try:
            found.append((jev_state.SIGNAL_HUMAN, float(response.nouls[jev_state.SIGNAL_HUMAN].noul)))
            found.append((jev_state.SIGNAL_ANSWER, float(response.nouls[jev_state.SIGNAL_ANSWER].noul)))
            found.append((jev_state.SIGNAL_FRUSTRATION, float(response.scores[jev_state.SIGNAL_FRUSTRATION].score)))
        except Exception:
            pass
        return tuple(found)


def make_jev_scorer(*, model: str = JEV_MODEL) -> JevScorer:
    """A scorer on the real API, with the key from ``JEV_KEY``. Needs the ``jev`` extra."""
    from typesafe_sdk import AsyncTypeSafeClient

    client = AsyncTypeSafeClient(api_key=require(JEV_KEY_VAR), model=model)
    return JevScorer(client, model=model)
