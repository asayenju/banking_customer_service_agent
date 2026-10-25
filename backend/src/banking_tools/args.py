"""Pydantic argument models generated from the tau2 method signatures.

tau2's dispatcher parses JSON with parse_int=float so IDs and DB-state hashes do
not depend on how a number was written (tools.py, call_discoverable_agent_tool).
Calling the methods directly must do the same, which is what `call_kwargs` is for.
"""

import inspect
import typing
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, create_model
from tau2.domains.banking_knowledge.tools import KnowledgeTools


def _integral_float_to_int(value: Any) -> Any:
    # JSON clients often send 5000.0 for an int field; accept it only if it is whole.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _field_type(hint: Any) -> Any:
    if hint is int:
        return Annotated[int, BeforeValidator(_integral_float_to_int)]
    return hint


def build_args_model(name: str) -> type[BaseModel]:
    """Strict model for one tool's arguments; unknown or mistyped arguments fail."""
    func = getattr(KnowledgeTools, name)
    hints = typing.get_type_hints(func)
    fields: dict[str, Any] = {}
    for param in inspect.signature(func).parameters.values():
        if param.name == "self":
            continue
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (_field_type(hints[param.name]), default)
    model_name = "".join(part.capitalize() for part in name.split("_")) + "Args"
    return create_model(
        model_name,
        __config__=ConfigDict(extra="forbid", strict=True, frozen=True),
        **fields,
    )


def call_kwargs(args: BaseModel) -> dict[str, Any]:
    """Keyword arguments for the tau2 method, with numbers normalised to float."""
    return {
        key: float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else value
        for key, value in ((k, getattr(args, k)) for k in type(args).model_fields)
    }
