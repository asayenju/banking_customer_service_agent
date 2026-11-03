"""Tool calling for the agent loop, kept apart from the decomposer's JSON client.

The decomposer asks a model for one schema-constrained JSON reply (``LLMClient``).
An agent needs the other mode: offer a set of tools, get back text or tool calls, run
the tools, send the results, repeat. That is a different contract, so it is a separate
Protocol and the decomposer, Jev scorer and their fakes are untouched.

Rules this module keeps:
  - ``parallel_tool_calls`` is off. Writes are made one at a time, in an order the
    loop controls.
  - A tool call's arguments arrive as a JSON string from the model and are returned as
    one, unparsed. The loop validates them against the tool's argument model; nothing
    here trusts them.
  - The model, key and timeout come from configuration (``AGENT_MODEL``), never from
    a caller's message.
  - ``FakeToolLLM`` fails loudly on an unscripted call and records the tools it was
    offered, so a test can assert that a tool was never even available.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional, Protocol

from pydantic import BaseModel, ConfigDict

from orchestrator.config import OPENAI_KEY_VAR, agent_model, require

DEFAULT_TIMEOUT_SECONDS = 60.0

Message = Mapping[str, Any]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolSchema(_Model):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(_Model):
    id: str
    name: str
    # The model's raw JSON text. Parsed and validated by the loop, never here.
    arguments_json: str


class Usage(_Model):
    input_tokens: int = 0
    output_tokens: int = 0


class AssistantTurn(_Model):
    text: Optional[str] = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = Usage()


class ToolCallingLLMClient(Protocol):
    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema],
        tool_choice: Literal["auto", "required"] = "auto",
        temperature: float = 0.0,
    ) -> AssistantTurn: ...


def assistant_message(turn: AssistantTurn) -> dict[str, Any]:
    """The turn as a message to put back in the history."""
    message: dict[str, Any] = {"role": "assistant", "content": turn.text or ""}
    if turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json},
            }
            for call in turn.tool_calls
        ]
    return message


def tool_message(call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _tool_payload(tool: ToolSchema) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters},
    }


class LiteLLMToolClient:
    """Calls a model through LiteLLM with tools offered, one tool call at a time."""

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._model = model or agent_model()
        self._api_key = api_key or require(OPENAI_KEY_VAR)
        self._timeout = timeout

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema],
        tool_choice: Literal["auto", "required"] = "auto",
        temperature: float = 0.0,
    ) -> AssistantTurn:
        import litellm

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [dict(m) for m in messages],
            "temperature": temperature,
            "api_key": self._api_key,
            "timeout": self._timeout,
        }
        if tools:
            kwargs["tools"] = [_tool_payload(t) for t in tools]
            kwargs["tool_choice"] = tool_choice
            kwargs["parallel_tool_calls"] = False
        response = await litellm.acompletion(**kwargs)
        message = response.choices[0].message
        calls = tuple(
            ToolCall(id=c.id, name=c.function.name, arguments_json=c.function.arguments or "{}")
            for c in (message.tool_calls or [])
        )
        usage = getattr(response, "usage", None)
        content = message.content
        return AssistantTurn(
            text=content if isinstance(content, str) and content else None,
            tool_calls=calls,
            usage=Usage(
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            ),
        )


class FakeToolLLM:
    """Replies with scripted turns in order (or raises a scripted exception) and records calls."""

    def __init__(self, turns: Sequence[AssistantTurn | Exception]) -> None:
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    @property
    def offered_tools(self) -> list[list[str]]:
        """The tool names offered on each call, in order."""
        return [c["tools"] for c in self.calls]

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema],
        tool_choice: Literal["auto", "required"] = "auto",
        temperature: float = 0.0,
    ) -> AssistantTurn:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "tools": [t.name for t in tools],
                "tool_choice": tool_choice,
                "temperature": temperature,
            }
        )
        if not self._turns:
            raise AssertionError("tool-calling model called more often than scripted")
        turn = self._turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn

    @property
    def remaining(self) -> int:
        return len(self._turns)
