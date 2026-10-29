"""The seam between the decomposer and a language model.

``LLMClient`` returns the model's raw text for a schema-constrained request, so
parsing and validation stay in our code. ``LiteLLMClient`` is the only place
that touches the network. ``FakeLLMClient`` hands out scripted replies and
records every call for tests.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from orchestrator.config import OPENAI_KEY_VAR, decomposer_model, require

Message = Mapping[str, str]

DEFAULT_TIMEOUT_SECONDS = 30.0


class LLMClient(Protocol):
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        schema: Mapping[str, Any],
        temperature: float = 0.0,
    ) -> str: ...


class LiteLLMClient:
    """Calls a model through LiteLLM with strict JSON-schema output."""

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._model = model or decomposer_model()
        self._api_key = api_key or require(OPENAI_KEY_VAR)
        self._timeout = timeout

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        schema: Mapping[str, Any],
        temperature: float = 0.0,
    ) -> str:
        import litellm

        response = await litellm.acompletion(
            model=self._model,
            messages=[dict(m) for m in messages],
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "interpretation", "schema": dict(schema), "strict": True},
            },
            api_key=self._api_key,
            timeout=self._timeout,
        )
        content = response.choices[0].message.content
        return content if isinstance(content, str) else ""


class FakeLLMClient:
    """Replies in order (a string, or an exception to raise) and records calls."""

    def __init__(self, replies: Sequence[str | Exception]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        schema: Mapping[str, Any],
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(
            {"messages": [dict(m) for m in messages], "schema": dict(schema), "temperature": temperature}
        )
        if not self._replies:
            raise AssertionError("LLM called more often than scripted")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
