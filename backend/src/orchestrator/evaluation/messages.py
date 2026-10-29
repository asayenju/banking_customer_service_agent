"""Write the frozen eval set: one first-person customer message per selected task.

A tau2 user brief is a role-play instruction ("You are playing a customer who..."),
not something a customer types. A model rewrites each brief once, and the result
is saved to ``evals/banking_eval_set.json`` so every later run, and every trial,
sees identical input; only the decomposer varies. Re-run ``build`` to regenerate.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from orchestrator.evaluation.tasks import DATASET_PATH, EvalCase, expected_kinds
from orchestrator.llm import LLMClient
from orchestrator.plan import READ_ONLY_KINDS
from orchestrator.policy import ACTION_POLICY
from orchestrator.types import IntentKind

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You turn a role-play brief for a bank customer into the first message
that customer would type to the bank's support chat.
- First person, natural, as a real customer would write it. Never mention a brief,
  role-play, an agent or a simulation.
- Include everything the customer wants done or asked about in this conversation, even
  when the brief says to reveal details only when asked: the customer's goals must be
  stated, the small details may be left out.
- Do not add requests, account ids, amounts or facts the brief does not contain.
- Keep personality and background only if it is short. At most about 120 words."""


async def write_message(client: LLMClient, brief: str) -> str:
    raw = await client.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"brief": brief})},
        ],
        schema=SCHEMA,
        temperature=0.0,
    )
    message = json.loads(raw)["message"]
    if not isinstance(message, str) or not message.strip():
        raise ValueError("the model returned an empty message")
    return message.strip()


ACTION_KINDS = [k.value for k in IntentKind if k not in READ_ONLY_KINDS and k is not IntentKind.OTHER]

LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"kinds": {"type": "array", "items": {"type": "string", "enum": ACTION_KINDS}}},
    "required": ["kinds"],
    "additionalProperties": False,
}

LABEL_PROMPT = f"""You label a bank customer's message for an evaluation answer key.
List the kinds of ACTION the customer asks the bank to carry out in this message, each
kind once, using only these kinds: {", ".join(ACTION_KINDS)}.

{ACTION_POLICY}

Return an empty list when the customer asks for no action. Label only what the message
says, not steps a bank might take later."""


async def label_message(client: LLMClient, message: str) -> tuple[IntentKind, ...]:
    raw = await client.complete(
        [
            {"role": "system", "content": LABEL_PROMPT},
            {"role": "user", "content": json.dumps({"message": message})},
        ],
        schema=LABEL_SCHEMA,
        temperature=0.0,
    )
    return tuple(sorted({IntentKind(k) for k in json.loads(raw)["kinds"]}))


async def label_cases(client: LLMClient, cases: Sequence[EvalCase]) -> list[EvalCase]:
    return [
        c.model_copy(update={"stated": await label_message(client, c.message)}) for c in cases
    ]


async def build_cases(
    writer: LLMClient, labeler: LLMClient, tasks: Sequence[Mapping[str, Any]]
) -> list[EvalCase]:
    drafts = []
    for task in tasks:
        message = await write_message(writer, task["user_scenario"]["instructions"])
        kinds = tuple(sorted(expected_kinds(task)))
        drafts.append(EvalCase(task_id=task["id"], message=message, stated=(), expected=kinds))
    return await label_cases(labeler, drafts)


def save_cases(cases: Sequence[EvalCase], path: Path = DATASET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.model_dump(mode="json") for c in cases], indent=2) + "\n")


def load_cases(path: Path = DATASET_PATH) -> list[EvalCase]:
    return [EvalCase.model_validate_json(json.dumps(row)) for row in json.loads(path.read_text())]
