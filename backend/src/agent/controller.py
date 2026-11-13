"""The LLM behind ``orchestrator.controller``: reads one request's situation, may consult Jev, and chooses what to do."""

import json
from collections.abc import Mapping
from typing import Any, Optional

from agent.llm_tools import AssistantTurn, ToolCallingLLMClient, ToolSchema, assistant_message, tool_message
from orchestrator.controller import Choice, ControlCase, ControlChoice

MAX_STEPS = 3

SYSTEM = """You decide what a bank's customer service assistant does next with ONE request, in a live conversation.
You are shown the customer's latest message, the last turns, the request as understood so far, what the assistant can do
for that kind of request, every question already put to the customer about it, and whether a question is on screen.

You choose exactly one:
- run: carry on with the request. Choose this whenever the conversation makes the request reasonably clear, even if the
  customer was brief ("yes", "all cards", "the silver one"): read the recent turns and the question on screen before you
  decide it is unclear. If a detail is still missing or a plan needs the customer's yes, the system asks for that itself,
  so do not hold back for it.
- ask: ask ONE short, specific question, with choices when there are a few. Only when something you cannot work out
  really changes what happens. Never repeat a question that was already asked, never ask for something the records hold,
  and never ask what the customer just answered. If the same thing was asked once, do not ask it again: choose run
  with the most likely reading instead.
- escalate: a person must take over. Only if the customer asks for a person, the request is outside what a bank's
  customer service does, or you have already tried to resolve it and cannot.

Jev is a separate scoring model. You MAY call ask_jev to hear how sure it is that the request is what it looks like
(kind, details, scope). Its number is advice: it only sees a short summary, it is strict with brief or hedged wording,
and a low number alone is never a reason to hand the customer to a person. Weigh it against what the conversation shows.

A read-only request is harmless, so lean to run and offer the other reading afterwards. A change is protected anyway: an
irreversible one is shown as an exact plan and needs the customer's yes, so you do not need to be certain of every
detail, only of what the customer wants. Always end by calling decide."""

ASK_JEV = ToolSchema(
    name="ask_jev",
    description="Ask Jev, an independent scoring model, how sure it is that this request is what the assistant understood. Advice only.",
    parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
)
DECIDE = ToolSchema(
    name="decide",
    description="End with your choice for this request.",
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": [c.value for c in Choice]},
            "question": {"type": "string", "description": "For ask: the one question, in plain friendly words. Otherwise ''."},
            "reason": {"type": "string", "description": "One short sentence on why."},
        },
        "required": ["action", "question", "reason"],
        "additionalProperties": False,
    },
)


def _situation(case: ControlCase) -> str:
    body: dict[str, Any] = {
        "latest_customer_message": case.message,
        "recent_conversation": [f"{who}: {said}" for who, said in case.recent],
        "question_on_screen": case.on_screen or None,
        "questions_already_asked_about_this_request": list(case.already_asked),
        "times_asked_so_far": case.asks_so_far,
        "request_kind": case.request_kind,
        "request_as_understood": case.request_summary,
        "details_found": case.details,
        "details_missing": list(case.missing),
        "what_the_assistant_can_do_for_this_kind": case.what_it_can_do,
        "how_risky": case.safety,
        "products_the_customer_holds": case.products or None,
    }
    return json.dumps(body, ensure_ascii=False)


class LLMController:
    def __init__(self, client: ToolCallingLLMClient, *, max_steps: int = MAX_STEPS) -> None:
        self._client = client
        self._max_steps = max_steps

    async def decide(self, case: ControlCase) -> Optional[ControlChoice]:
        messages: list[Mapping[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _situation(case)},
        ]
        for step in range(self._max_steps):
            tools = [ASK_JEV, DECIDE] if step < self._max_steps - 1 else [DECIDE]
            turn: AssistantTurn = await self._client.chat(messages, tools=tools, tool_choice="required", temperature=0.0)
            if not turn.tool_calls:
                return None
            call = turn.tool_calls[0]
            messages.append(assistant_message(turn))
            if call.name == "decide":
                try:
                    args = json.loads(call.arguments_json or "{}")
                    return ControlChoice(Choice(args["action"]), str(args.get("question") or "").strip(), str(args.get("reason") or "").strip())
                except (ValueError, KeyError):
                    return None
            if call.name == "ask_jev":
                messages.append(tool_message(call.id, await case.consult_jev()))
                continue
            return None
        return None
