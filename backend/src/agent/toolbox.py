"""What an agent is allowed to see and call, built from the tool catalog.

``ScopedToolBox`` is the only door between a model and the banking tools:

- It offers only the tools it was given. A call to anything else is an error string,
  not an attempt.
- Each tool's schema is generated from the catalog's argument model, made strict (every
  field required, no extra fields), described with the tool's real docstring, and has
  ``user_id`` removed. The verified customer's id is put in by code, so the model cannot
  name another customer, even by mistake.
- PINs and CVVs never pass through the model. The customer's words are first run through
  a ``SecretVault`` that swaps each secret for a reference like ``{{secret:pin}}``. The
  model passes the reference as the argument; the toolbox swaps the real value back in
  at the moment of the call. Everything the model gets back is scrubbed of every
  vault value, of card numbers, and of CVVs, including pydantic's own error messages,
  which would otherwise echo the rejected input.
- A tool result is trimmed to a fixed length, so one huge transaction dump cannot
  swamp the model's context.
"""

import json
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, Optional

from pydantic import ValidationError
from tau2.domains.banking_knowledge.tools import (
    KnowledgeTools,
    parse_discoverable_tool_docstring,
)

from agent.llm_tools import ToolCall, ToolSchema
from banking_tools.catalog import get_registry
from banking_tools.session import REDACTED_ARGUMENTS
from banking_tools.types import ToolKind, ToolSpec
from orchestrator.context import AgentSession
from orchestrator.sanitize import replace_secrets

HIDDEN_ARGUMENT = "user_id"
KEPT_PRIVATE = "(kept private)"
DEFAULT_RESULT_CHARS = 4000
_REF = re.compile(r"^\{\{secret:([a-z_]+)\}\}$")
_SECRET_FIELD_HELP = (
    "A reference to a secret the customer gave, written exactly as it appears in their "
    "message, like {{secret:pin}}. Never a real value."
)
# JSON Schema keywords a strict-mode tool schema may use.
# A bare 3 to 8 digit number: not part of a word, an amount ($12), a date (11/14), a decimal or a range.
_BARE_NUMBER = re.compile(r"(?<![\w$.,/:-])\d{3,8}(?![\w%/:-]|[.,]\d)")
_ALLOWED = frozenset(
    {"type", "properties", "required", "additionalProperties", "description", "enum", "anyOf", "items"}
)


class SecretVault:
    """Holds the secrets a customer typed so the model never has to see them."""

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def _name_for(self, kind: str, value: str) -> str:
        name = kind
        letter = ord("b")
        while name in self._values and self._values[name] != value:
            name = f"{kind}_{chr(letter)}"
            letter += 1
        self._values[name] = value
        return name

    def capture(self, text: str, *, expecting_secret: bool = False) -> str:
        """The text with each PIN, CVV and card number replaced by a reference.

        ``expecting_secret`` is for the message that answers "what PIN would you like?": the
        customer may well type just the digits, with no word "pin" beside them. Then any bare
        number that the cue-based search did not already catch is taken to be the secret.
        Numbers that are clearly something else (an amount, a date, a decimal) are left alone.
        """
        cleaned = replace_secrets(text, lambda kind, value: "{{secret:" + self._name_for(kind, value) + "}}")
        if expecting_secret:
            cleaned = _BARE_NUMBER.sub(lambda m: "{{secret:" + self._name_for("pin", m.group(0)) + "}}", cleaned)
        return cleaned

    def hide(self, text: str, values: Mapping[str, str]) -> str:
        """Replace known secret values (name -> value) wherever they appear, whatever surrounds them."""
        for name, value in values.items():
            if value:
                text = text.replace(value, self.store(name, value))
        return text

    def store(self, name: str, value: str) -> str:
        """Hold a secret that did not come from text (a form field). Returns its reference."""
        if not re.fullmatch(r"[a-z_]+", name):
            raise ValueError("a secret name is lowercase letters and underscores")
        self._values[name] = value
        return "{{secret:" + name + "}}"

    def resolve(self, reference: str) -> Optional[str]:
        match = _REF.match(reference)
        return self._values.get(match.group(1)) if match else None

    def scrub(self, text: str) -> str:
        """Remove anything secret from text on its way to a model, a trace or a customer."""
        for value in sorted(self._values.values(), key=len, reverse=True):
            if len(value) >= 3:
                text = text.replace(value, KEPT_PRIVATE)
        return replace_secrets(text, lambda kind, value: KEPT_PRIVATE)


@dataclass(frozen=True)
class ExtraTool:
    """A tool that is not in the banking catalog (policy search). It only reads, by contract."""

    schema: ToolSchema
    run: Callable[[Mapping[str, Any]], str]


@dataclass(frozen=True)
class ToolExecution:
    tool: str
    ok: bool
    # Already scrubbed and trimmed: safe to hand to the model.
    text: str


@cache
def _documented() -> Mapping[str, Mapping[str, Any]]:
    """tau2's own parsed docstring for every tool, which is the source of truth for what a
    parameter means (the knowledge-base documents use different names)."""
    from banking_tools.session import _base_db

    toolkit = KnowledgeTools(_base_db().model_copy(deep=True))
    discoverable = toolkit.get_discoverable_tools()
    found: dict[str, Mapping[str, Any]] = {}
    for name, spec in get_registry().items():
        method = discoverable.get(name) if spec.source == "discoverable" else getattr(toolkit, name, None)
        if method is None:
            continue
        try:
            found[name] = parse_discoverable_tool_docstring(method)
        except Exception:  # a docstring in another style: fall back to the catalog's line
            continue
    return found


def _strict(spec: ToolSpec) -> dict[str, Any]:
    assert spec.args_model is not None
    schema = spec.args_model.model_json_schema()
    documented = (_documented().get(spec.name) or {}).get("parameters") or {}
    properties: dict[str, Any] = {}
    for name, prop in (schema.get("properties") or {}).items():
        if name == HIDDEN_ARGUMENT:
            continue
        clean = {k: v for k, v in prop.items() if k in _ALLOWED}
        if name in REDACTED_ARGUMENTS:
            clean = {"type": "string", "description": _SECRET_FIELD_HELP}
        else:
            help_text = (documented.get(name) or {}).get("description")
            if help_text:
                clean["description"] = str(help_text)
        properties[name] = clean
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def tool_schema(spec: ToolSpec) -> ToolSchema:
    description = (_documented().get(spec.name) or {}).get("description") or spec.description
    return ToolSchema(name=spec.name, description=str(description), parameters=_strict(spec))


def is_strict_valid(schema: Mapping[str, Any]) -> bool:
    """Would a provider's strict function-calling mode accept this schema?"""
    if schema.get("type") == "object":
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is not False:
            return False
        if set(schema.get("required") or []) != set(properties):
            return False
        return all(is_strict_valid(p) for p in properties.values()) and set(schema) <= _ALLOWED
    if "anyOf" in schema:
        return all(is_strict_valid(p) for p in schema["anyOf"])
    if schema.get("type") == "array":
        return "items" in schema and is_strict_valid(schema["items"])
    return set(schema) <= _ALLOWED and "type" in schema


def _error(tool: str, text: str, vault: SecretVault, limit: int) -> ToolExecution:
    return ToolExecution(tool=tool, ok=False, text=vault.scrub(text)[:limit])


class ScopedToolBox:
    def __init__(
        self,
        session: AgentSession,
        *,
        user_id: Optional[str],
        allowed_tools: Collection[str],
        vault: Optional[SecretVault] = None,
        max_result_chars: int = DEFAULT_RESULT_CHARS,
        extra_tools: Sequence[ExtraTool] = (),
    ) -> None:
        registry = get_registry()
        clash = sorted({t.schema.name for t in extra_tools} & set(registry))
        if clash:
            raise ValueError(f"an extra tool shares a name with a catalog tool: {', '.join(clash)}")
        self._extras = {t.schema.name: t for t in extra_tools}
        unknown = sorted(set(allowed_tools) - set(registry))
        if unknown:
            raise ValueError(f"not in the tool catalog: {', '.join(unknown)}")
        self._session = session
        self._user_id = user_id
        self._allowed = tuple(sorted(allowed_tools))
        self.vault = vault or SecretVault()
        self._limit = max_result_chars

    def is_write(self, name: str) -> bool:
        """Does this tool change anything? Unknown names count as writes: fail closed."""
        if name in self._extras:
            return False
        spec = get_registry().get(name)
        return spec is None or spec.kind is not ToolKind.READ

    def schemas(self) -> list[ToolSchema]:
        registry = get_registry()
        return [tool_schema(registry[name]) for name in self._allowed] + [
            t.schema for t in self._extras.values()
        ]

    def _run_extra(self, tool: ExtraTool, call: ToolCall) -> ToolExecution:
        name = tool.schema.name
        try:
            raw = json.loads(call.arguments_json or "{}")
        except json.JSONDecodeError:
            return _error(name, "Error: the arguments were not valid JSON.", self.vault, self._limit)
        if not isinstance(raw, dict) or set(raw) != set(tool.schema.parameters["properties"]):
            return _error(name, "Error: the arguments did not match the tool's schema.", self.vault, self._limit)
        try:
            text = tool.run(raw)
        except Exception as exc:  # a tool bug must not end the run, and must not leak its message
            return _error(name, f"Error: {type(exc).__name__}", self.vault, self._limit)
        return ToolExecution(tool=name, ok=True, text=self.vault.scrub(text)[: self._limit])

    def execute(self, call: ToolCall) -> ToolExecution:
        """Run one model-requested tool call. Never raises; everything returned is safe to show."""
        name = call.name
        if name in self._extras:
            return self._run_extra(self._extras[name], call)
        if name not in self._allowed:
            return _error(name, f"Error: the tool '{name}' is not available.", self.vault, self._limit)
        spec = get_registry()[name]
        try:
            raw = json.loads(call.arguments_json or "{}")
        except json.JSONDecodeError:
            return _error(name, "Error: the arguments were not valid JSON.", self.vault, self._limit)
        if not isinstance(raw, dict):
            return _error(name, "Error: the arguments must be a JSON object.", self.vault, self._limit)

        arguments: dict[str, Any] = {}
        for key, value in raw.items():
            if key == HIDDEN_ARGUMENT:
                continue  # never the model's to set; filled in below
            if key in REDACTED_ARGUMENTS:
                resolved = self.vault.resolve(value) if isinstance(value, str) else None
                if resolved is None:
                    return _error(
                        name,
                        f"Error: '{key}' must be a secret reference like {{{{secret:pin}}}} "
                        "that appears in the customer's message.",
                        self.vault,
                        self._limit,
                    )
                value = resolved
            arguments[key] = value
        if spec.args_model is not None and HIDDEN_ARGUMENT in spec.args_model.model_fields:
            if self._user_id is None:
                return _error(name, "Error: no verified customer for this action.", self.vault, self._limit)
            arguments[HIDDEN_ARGUMENT] = self._user_id

        try:
            result = self._session.call(name, arguments)
        except ValidationError as exc:  # a session that raises instead of returning a failure
            return _error(name, f"Error: invalid arguments: {exc}", self.vault, self._limit)
        return ToolExecution(
            tool=name,
            ok=bool(result.ok),
            text=self.vault.scrub(str(result.text))[: self._limit],
        )
