"""Core types shared by every banking tool."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Optional, Self

from pydantic import BaseModel, ConfigDict, model_validator


class ToolKind(StrEnum):
    READ = "read"
    WRITE = "write"
    HANDOFF = "handoff"


class Reversibility(StrEnum):
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


class ToolSpec(BaseModel):
    """Static description of one tool: what it is and how risky it is to call."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: ToolKind
    reversibility: Optional[Reversibility] = None
    description: str
    # "visible" tools are registered with tau2; "discoverable" ones need an unlock there.
    source: Literal["visible", "discoverable"]
    # Set by the argument-model generator.
    args_model: Optional[type[BaseModel]] = None

    @model_validator(mode="after")
    def _reversibility_matches_kind(self) -> Self:
        if self.kind is ToolKind.WRITE and self.reversibility is None:
            raise ValueError(f"write tool '{self.name}' must declare a reversibility")
        if self.kind is not ToolKind.WRITE and self.reversibility is not None:
            raise ValueError(
                f"{self.kind.value} tool '{self.name}' must not declare a reversibility"
            )
        return self


class AuditEntry(BaseModel):
    """One executed write call. Sensitive arguments are redacted before storage."""

    model_config = ConfigDict(frozen=True)

    audit_id: str
    tool: str
    reversibility: Reversibility
    arguments: dict[str, Any]
    ok: bool
    timestamp: datetime


class ToolResult(BaseModel):
    """Outcome of one tool call. tau2 reports failures as strings, never exceptions."""

    model_config = ConfigDict(frozen=True)

    tool: str
    ok: bool
    text: str
    error: Optional[str] = None
    audit_id: Optional[str] = None

    @model_validator(mode="after")
    def _error_matches_ok(self) -> Self:
        if self.ok and self.error is not None:
            raise ValueError("a successful result must not carry an error")
        if not self.ok and not self.error:
            raise ValueError("a failed result must carry an error")
        return self
