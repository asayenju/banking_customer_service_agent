"""Isolated execution of banking tools.

A BankingSession owns a private copy of the banking database, calls the tau2
tool methods directly (no unlock step, no JSON arguments), maps tau2's
"Error: ..." strings to a typed failure, and records every executed write in an
audit log.

Differences from tau2's call_discoverable_agent_tool, on purpose:
- no unlock gate;
- no row in the `agent_discoverable_tools` table; the audit log here replaces it.
"""

import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import cache
from typing import Any, Callable, Optional

from pydantic import ValidationError
from tau2.domains.banking_knowledge.data_model import TransactionalDB
from tau2.domains.banking_knowledge.environment import get_db
from tau2.domains.banking_knowledge.tools import KnowledgeTools

from banking_tools.args import call_kwargs
from banking_tools.catalog import get_registry
from banking_tools.types import AuditEntry, ToolKind, ToolResult

ERROR_PREFIX = "Error:"

# Argument names that must never reach the audit log.
REDACTED_ARGUMENTS = frozenset({"pin", "new_pin", "current_pin", "cvv"})
REDACTED = "***"


@cache
def _base_db() -> TransactionalDB:
    """The pristine database, loaded once. Sessions only ever copy it."""
    db = get_db()
    assert isinstance(db, TransactionalDB)
    return db


def _redact(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: REDACTED if key in REDACTED_ARGUMENTS else value
        for key, value in arguments.items()
    }


class BankingSession:
    def __init__(self, db: Optional[TransactionalDB] = None) -> None:
        self._db = db if db is not None else _base_db().model_copy(deep=True)
        self._toolkit = KnowledgeTools(self._db)
        self._lock = threading.Lock()
        self._audit: list[AuditEntry] = []

    @property
    def audit_log(self) -> tuple[AuditEntry, ...]:
        with self._lock:
            return tuple(self._audit)

    def db_hash(self) -> str:
        with self._lock:
            return str(self._db.get_hash())

    def call(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> ToolResult:
        spec = get_registry().get(name)
        if spec is None or spec.args_model is None:
            return _failure(name, f"Unknown tool '{name}'")
        try:
            args = spec.args_model.model_validate(dict(arguments or {}))
        except ValidationError as exc:
            return _failure(name, f"Invalid arguments: {exc}")

        with self._lock:
            ok, text, error = self._execute(spec.source, name, call_kwargs(args))
            audit_id: Optional[str] = None
            if spec.kind is ToolKind.WRITE and spec.reversibility is not None:
                audit_id = f"audit-{len(self._audit) + 1:06d}"
                self._audit.append(
                    AuditEntry(
                        audit_id=audit_id,
                        tool=name,
                        reversibility=spec.reversibility,
                        arguments=_redact(args.model_dump()),
                        ok=ok,
                        timestamp=datetime.now(UTC),
                    )
                )
        return ToolResult(tool=name, ok=ok, text=text, error=error, audit_id=audit_id)

    def _method(self, source: str, name: str) -> Callable[..., str]:
        if source == "discoverable":
            return self._toolkit.get_discoverable_tools()[name]  # type: ignore[no-any-return]
        return getattr(self._toolkit, name)  # type: ignore[no-any-return]

    def _execute(
        self, source: str, name: str, kwargs: dict[str, Any]
    ) -> tuple[bool, str, Optional[str]]:
        try:
            text = self._method(source, name)(**kwargs)
        except Exception as exc:  # tau2 should not raise, but a tool bug must not escape
            message = f"{type(exc).__name__}: {exc}"
            return False, message, message
        if text.startswith(ERROR_PREFIX):
            return False, text, text[len(ERROR_PREFIX) :].strip() or text
        return True, text, None


def _failure(name: str, error: str) -> ToolResult:
    return ToolResult(tool=name, ok=False, text=f"{ERROR_PREFIX} {error}", error=error)
