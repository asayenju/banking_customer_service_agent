"""What the customer's message means, as the orchestrator needs it each turn.

The real decomposer (an LLM behind this Protocol) is a later piece. Until then a
scripted one stands in, so the orchestrator's choices can be tested on their own.

An Interpretation covers one message in the middle of a conversation, not just
a first request: new intents, updated ones (the same id with new details), a yes
to something we asked, a withdrawn request, and any identity details given.
Identity claims are untrusted: they are read by ``auth.verify``, which ignores
everything but the four real fields.
"""

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from orchestrator.types import Decomposition, Intent


def _nothing() -> Decomposition:
    return Decomposition(intents=())


class Interpretation(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    # Intents mentioned this turn. An id we already hold means "updated".
    decomposition: Decomposition = Field(default_factory=_nothing)
    # Ids the customer said yes to.
    confirmed: tuple[str, ...] = ()
    # Ids the customer withdrew.
    declined: tuple[str, ...] = ()
    claims: dict[str, str] = Field(default_factory=dict)


class Decomposer(Protocol):
    async def interpret(
        self, message: str, open_intents: Sequence[Intent]
    ) -> Interpretation: ...


class ScriptedDecomposer:
    """Hands out its interpretations in order and records what it was shown."""

    def __init__(self, script: Sequence[Interpretation | Exception]) -> None:
        self._script = list(script)
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def interpret(
        self, message: str, open_intents: Sequence[Intent]
    ) -> Interpretation:
        self.calls.append((message, tuple(i.id for i in open_intents)))
        if not self._script:
            raise AssertionError("decomposer called more often than scripted")
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step
