"""The bounded loop a specialist runs: ask the model, run its tool calls, repeat.

The loop never trusts the model to stop. Every way it can run on is capped by code, and
every way it can end is a value in ``StopReason``, so a caller (and a test) can tell a
finished run from one that ran out of steps, money or time, without reading any text the
model wrote.

Budgets (``Budget``):
  - ``max_steps``: model calls.
  - ``max_tool_calls``: tool calls, good or bad.
  - ``max_writes``: writes started (a failed attempt counts).
  - ``max_input_tokens``: total prompt tokens across the run, which bounds spend.
  - ``max_errors``: tool calls that came back as errors, malformed ones included.
  - ``write_margin``: no write is started with less than this many seconds left. A tool
    call is one inline, uninterruptible step, so a write that starts finishes; what must
    not happen is starting one that a timeout would then cut off from its bookkeeping.

Review. A caller may pass ``review``, a function that looks at the way the model wants to end
(a plan, an answer, a question) and returns a reason it is not acceptable, or None. The reason
is handed back to the model as the answer to that call and the run goes on, so a plan the code
would refuse can be corrected instead of thrown away. A refusal counts as an error, so a model
that keeps getting it wrong still stops. Reasons must be written by code, never by the model.

How a run ends: the model calls a terminal pseudo-tool (``finish``, ``ask_customer``,
``escalate``, ``propose_plan``), or a budget runs out, or the model fails. The model is
asked for a tool call every turn (``tool_choice="required"``), so there is no free-text
way out; a reply with no tool call is counted as an error and the model is nudged.

Failures are recorded by exception type only. A model or tool error can echo the
request, and the request holds customer data.
"""

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional

from agent.llm_tools import (
    AssistantTurn,
    ToolCall,
    ToolCallingLLMClient,
    ToolSchema,
    Usage,
    assistant_message,
    tool_message,
)
from agent.toolbox import ScopedToolBox


class StopReason(StrEnum):
    TERMINAL = "terminal"
    STEPS = "steps_exhausted"
    TOOL_CALLS = "tool_calls_exhausted"
    WRITES = "writes_exhausted"
    TOKENS = "tokens_exhausted"
    DEADLINE = "deadline"
    ERRORS = "too_many_errors"
    MODEL_FAILED = "model_failed"


@dataclass(frozen=True)
class Budget:
    max_steps: int = 8
    max_tool_calls: int = 12
    max_writes: int = 4
    max_input_tokens: int = 100_000
    max_errors: int = 4
    write_margin: float = 5.0


@dataclass(frozen=True)
class ToolRecord:
    name: str
    ok: bool
    write: bool


@dataclass(frozen=True)
class AgentResult:
    reason: StopReason
    # Set when the model ended the run itself.
    terminal: Optional[str] = None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    steps: int = 0
    tool_calls: int = 0
    writes: int = 0
    errors: int = 0
    usage: Usage = Usage()
    log: tuple[ToolRecord, ...] = ()

    @property
    def finished(self) -> bool:
        return self.reason is StopReason.TERMINAL


def _terminal(name: str, description: str, properties: dict[str, Any]) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    )


TERMINALS: Mapping[str, ToolSchema] = {
    "finish": _terminal(
        "finish",
        "End the run: tell the customer what was done or answer their question.",
        {"message": {"type": "string", "description": "What to tell the customer."}},
    ),
    "ask_customer": _terminal(
        "ask_customer",
        "End the run to ask the customer for details you still need. Name the details; "
        "never include a value the customer gave.",
        {
            "details": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short names of the missing details, like card_id.",
            }
        },
    ),
    "escalate": _terminal(
        "escalate",
        "End the run and hand the customer to a human agent.",
        {"reason": {"type": "string", "description": "Why a person should take over."}},
    ),
    "propose_plan": _terminal(
        "propose_plan",
        "End the run with the exact changes you would make, in order. Nothing is changed "
        "until the customer approves.",
        {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string"},
                        "arguments_json": {"type": "string", "description": "The tool's arguments as a JSON object."},
                        "description": {"type": "string", "description": "What this does, in the customer's terms."},
                    },
                    "required": ["tool", "arguments_json", "description"],
                    "additionalProperties": False,
                },
            }
        },
    ),
}


Review = Callable[[str, Mapping[str, Any]], Optional[str]]


def _loop_time() -> float:
    return asyncio.get_running_loop().time()


async def run_agent(
    llm: ToolCallingLLMClient,
    toolbox: ScopedToolBox,
    *,
    system: str,
    request: str,
    terminals: Sequence[str] = ("finish", "ask_customer", "escalate"),
    budget: Budget = Budget(),
    deadline: Optional[float] = None,
    clock: Callable[[], float] = _loop_time,
    review: Optional[Review] = None,
) -> AgentResult:
    """Run until the model ends the run or a budget does.

    ``request`` must already have had its secrets swapped for references
    (``SecretVault.capture``); the loop does not look inside it.
    """
    for name in terminals:
        if name not in TERMINALS:
            raise ValueError(f"unknown terminal tool '{name}'")
    offered = toolbox.schemas() + [TERMINALS[n] for n in terminals]
    clash = {t.name for t in toolbox.schemas()} & set(terminals)
    if clash:
        raise ValueError(f"a catalog tool shares a name with a terminal tool: {sorted(clash)}")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": request},
    ]
    steps = tool_calls = writes = errors = input_tokens = output_tokens = 0
    log: list[ToolRecord] = []

    def result(reason: StopReason, terminal: Optional[str] = None, arguments: Optional[Mapping[str, Any]] = None) -> AgentResult:
        return AgentResult(
            reason=reason,
            terminal=terminal,
            arguments=arguments or {},
            steps=steps,
            tool_calls=tool_calls,
            writes=writes,
            errors=errors,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            log=tuple(log),
        )

    def remaining() -> Optional[float]:
        return None if deadline is None else deadline - clock()

    while True:
        left = remaining()
        if left is not None and left <= 0:
            return result(StopReason.DEADLINE)
        if steps >= budget.max_steps:
            return result(StopReason.STEPS)
        if input_tokens >= budget.max_input_tokens:
            return result(StopReason.TOKENS)

        steps += 1
        try:
            turn: AssistantTurn = await asyncio.wait_for(
                llm.chat(messages, tools=offered, tool_choice="required"), left
            )
        except TimeoutError:
            return result(StopReason.DEADLINE)
        except Exception:  # only the fact of the failure is kept, never its message
            return result(StopReason.MODEL_FAILED)
        input_tokens += turn.usage.input_tokens
        output_tokens += turn.usage.output_tokens
        messages.append(assistant_message(turn))

        if not turn.tool_calls:
            errors += 1
            if errors >= budget.max_errors:
                return result(StopReason.ERRORS)
            messages.append(
                {"role": "user", "content": "Reply by calling a tool; if you are done, call finish."}
            )
            continue

        for call in turn.tool_calls:
            if call.name in terminals:
                parsed = _terminal_arguments(call, TERMINALS[call.name])
                if parsed is not None:
                    reason = _reviewed(review, call.name, parsed)
                    if reason is None:
                        return result(StopReason.TERMINAL, call.name, parsed)
                    messages.append(
                        tool_message(call.id, f"Not accepted: {reason}. Correct it and call {call.name} again.")
                    )
                else:
                    messages.append(tool_message(call.id, "Error: the arguments did not match the tool's schema."))
                errors += 1
                if errors >= budget.max_errors:
                    return result(StopReason.ERRORS)
                continue

            if tool_calls >= budget.max_tool_calls:
                return result(StopReason.TOOL_CALLS)
            is_write = toolbox.is_write(call.name)
            if is_write:
                if writes >= budget.max_writes:
                    return result(StopReason.WRITES)
                left = remaining()
                if left is not None and left < budget.write_margin:
                    return result(StopReason.DEADLINE)
            tool_calls += 1
            writes += is_write  # every write started counts, whether or not it worked
            # Inline and atomic: nothing here awaits, so a timeout cannot land mid-write.
            outcome = toolbox.execute(call)
            log.append(ToolRecord(name=call.name, ok=outcome.ok, write=is_write))
            messages.append(tool_message(call.id, outcome.text))
            if not outcome.ok:
                errors += 1
                if errors >= budget.max_errors:
                    return result(StopReason.ERRORS)


def _reviewed(review: Optional[Review], name: str, arguments: Mapping[str, Any]) -> Optional[str]:
    """The reason to send the model back, or None to accept. A review that breaks refuses."""
    if review is None:
        return None
    try:
        return review(name, arguments)
    except Exception:
        return "that could not be checked"


def _terminal_arguments(call: ToolCall, schema: ToolSchema) -> Optional[dict[str, Any]]:
    try:
        parsed = json.loads(call.arguments_json or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or set(parsed) != set(schema.parameters["properties"]):
        return None
    return parsed
