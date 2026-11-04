"""What a workflow is, what it can answer, and how its form is shown to a model.

``decide`` is plain code. It never raises on bad facts: it answers with one of

  - ``Plan``     the exact steps to propose (the customer still approves them);
  - ``NeedInfo`` facts only the customer can supply, with the question written by code;
  - ``Refuse``   the facts do not work, with a reason the model can act on and try again;
  - ``Reply``    policy says no, or nothing needs doing: the answer for the customer;
  - ``HandOff``  a person should take over, because the code cannot decide safely.

The model fills in a form built from the workflow's pydantic model. The form is a strict
JSON schema (every field present, no extras), so a made-up field or an out-of-list value
cannot be submitted.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol, get_args, get_origin

from pydantic import BaseModel

from orchestrator.types import IntentKind


@dataclass(frozen=True)
class WorkflowContext:
    """What a workflow may look at: the database as it is now, and who is asking."""

    database: Any
    user_id: str
    today: date


@dataclass(frozen=True)
class Step:
    """One tool call the workflow wants made. Arguments are final, apart from secrets."""

    tool: str
    args: Mapping[str, Any]


@dataclass(frozen=True)
class Plan:
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Ask:
    slot: str
    text: str


@dataclass(frozen=True)
class NeedInfo:
    questions: tuple[Ask, ...]


@dataclass(frozen=True)
class Refuse:
    reason: str


@dataclass(frozen=True)
class Reply:
    text: str


@dataclass(frozen=True)
class HandOff:
    reason: str


Decision = Plan | NeedInfo | Refuse | Reply | HandOff


class Workflow(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def kinds(self) -> frozenset[IntentKind]: ...

    # When to use this one, in a sentence. The model reads it to choose.
    @property
    def summary(self) -> str: ...

    @property
    def facts(self) -> type[BaseModel]: ...

    def decide(self, ctx: WorkflowContext, facts: Any) -> Decision: ...


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A pydantic model as a strict function-calling schema: every field required, no extras,
    nested models written out in place, defaults and titles removed."""
    raw = model.model_json_schema()
    definitions = raw.get("$defs", {})

    def clean(node: Any) -> Any:
        if isinstance(node, list):
            return [clean(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return clean(definitions[node["$ref"].rsplit("/", 1)[-1]])
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in ("title", "default", "$defs", "examples"):
                continue
            out[key] = clean(value)
        if out.get("type") == "object" or "properties" in out:
            properties = out.get("properties", {})
            out["type"] = "object"
            out["properties"] = properties
            out["required"] = list(properties)
            out["additionalProperties"] = False
        return out

    schema = clean(raw)
    assert isinstance(schema, dict)
    return schema


def terminal_name(workflow: Workflow) -> str:
    return f"start_{workflow.name}"


def literal_values(annotation: Any) -> tuple[Any, ...]:
    """The allowed values of a Literal (or an Optional Literal) annotation."""
    if get_origin(annotation) is not None and get_args(annotation):
        for arg in get_args(annotation):
            inner = literal_values(arg)
            if inner:
                return inner
        if str(get_origin(annotation)).endswith("Literal"):
            return tuple(get_args(annotation))
    return ()
