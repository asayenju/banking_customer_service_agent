"""Isolated execution of banking tools.

A BankingSession owns a private copy of the banking database, calls the tau2
tool methods directly (no unlock step, no JSON arguments), maps tau2's
"Error: ..." strings to a typed failure, and records every executed write in an
audit log.

Differences from tau2's call_discoverable_agent_tool, on purpose:
- no unlock gate;
- no row in the `agent_discoverable_tools` table; the audit log here replaces it.

Two fidelity modes. DIRECT (the default for ``BankingSession()``, and what the
older tests describe) is the behavior above. TAU2 sends discoverable tools through
tau2's own ``unlock_discoverable_agent_tool`` and ``call_discoverable_agent_tool``,
so the database ends up exactly as in a tau2 run, including the
``agent_discoverable_tools`` audit rows; that is what comparing against tau2's gold
needs. ``from_task`` defaults to TAU2. The audit log of writes is kept in both.

Time is injected. By default it is tau2's locked clock (2025-11-14 03:40 EST), never
wall time, so audit entries are reproducible and comparable with tau2's runs.
"""

import json
import threading
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from functools import cache
from typing import Any, Callable, Optional

from pydantic import ValidationError
from tau2.domains.banking_knowledge.data_model import TransactionalDB
from tau2.domains.banking_knowledge.environment import get_db
from tau2.domains.banking_knowledge.tools import KnowledgeTools
from tau2.domains.banking_knowledge.utils import get_now

from banking_tools.args import call_kwargs
from banking_tools.catalog import get_registry
from banking_tools.task_state import apply_initial_state
from banking_tools.types import AuditEntry, ToolKind, ToolResult

ERROR_PREFIX = "Error:"

# tau2's clock is a naive datetime documented as EST; mid-November is standard time.
_EST = timezone(timedelta(hours=-5))

# Argument names that must never reach the audit log.
REDACTED_ARGUMENTS = frozenset({"pin", "new_pin", "current_pin", "cvv"})
REDACTED = "***"


class Fidelity(StrEnum):
    DIRECT = "direct"
    TAU2 = "tau2"


@cache
def _base_db() -> TransactionalDB:
    """The pristine database, loaded once. Sessions only ever copy it."""
    db = get_db()
    assert isinstance(db, TransactionalDB)
    return db


def tau2_now() -> datetime:
    """tau2's locked "now" with its documented zone attached."""
    now: datetime = get_now()
    return now.replace(tzinfo=_EST)


def _redact(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: REDACTED if key in REDACTED_ARGUMENTS else value
        for key, value in arguments.items()
    }


class BankingSession:
    def __init__(
        self,
        db: Optional[TransactionalDB] = None,
        *,
        clock: Callable[[], datetime] = tau2_now,
        fidelity: Fidelity = Fidelity.DIRECT,
        read_log_allowlist: Iterable[str] = (),
    ) -> None:
        self._db = db if db is not None else _base_db().model_copy(deep=True)
        self._clock = clock
        self._fidelity = fidelity
        self._toolkit = KnowledgeTools(self._db)
        # tau2 logs a discoverable read only if the task's gold trajectory names it.
        self._toolkit.set_read_log_allowlist(set(read_log_allowlist))
        self._lock = threading.Lock()
        self._audit: list[AuditEntry] = []

    @classmethod
    def from_task(
        cls,
        task: Mapping[str, Any],
        *,
        clock: Callable[[], datetime] = tau2_now,
        fidelity: Fidelity = Fidelity.TAU2,
        read_log_allowlist: Iterable[str] = (),
    ) -> "BankingSession":
        """A session on the task's starting database (the base DB plus its overlay)."""
        return cls(
            apply_initial_state(_base_db().model_copy(deep=True), task),
            clock=clock,
            fidelity=fidelity,
            read_log_allowlist=read_log_allowlist,
        )

    def record_verification(
        self,
        *,
        name: str,
        user_id: str,
        address: str,
        email: str,
        phone_number: str,
        date_of_birth: str,
    ) -> ToolResult:
        """Log a successful 2-of-4 identity check the way tau2's policy asks.

        The code that verified the caller calls this once; the time is the session
        clock, formatted as in tau2's gold trajectories. ``verification_history`` is
        part of tau2's database hash, so skipping this makes a run unmatchable.
        """
        return self.call(
            "log_verification",
            {
                "name": name,
                "user_id": user_id,
                "address": address,
                "email": email,
                "phone_number": phone_number,
                "date_of_birth": date_of_birth,
                "time_verified": f"{self._clock():%Y-%m-%d %H:%M:%S} EST",
            },
        )

    @property
    def database(self) -> TransactionalDB:
        """The session's database. For the evaluation harness (comparing states, giving
        tau2's user toolkit the same database); agent code reads and writes through call()."""
        return self._db

    @property
    def toolkit(self) -> KnowledgeTools:
        """The tau2 toolkit behind the session. Evaluation harness only: it skips
        validation, the audit log and the lock, so agent code must use call()."""
        return self._toolkit

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
                        timestamp=self._clock(),
                    )
                )
        return ToolResult(tool=name, ok=ok, text=text, error=error, audit_id=audit_id)

    def _through_tau2_dispatcher(self, name: str, kwargs: dict[str, Any]) -> str:
        """tau2's own unlock + JSON call, so numbers, errors and audit rows are tau2's."""
        if name not in self._toolkit.get_agent_discoverable_tools_state():
            unlocked: str = self._toolkit.unlock_discoverable_agent_tool(name)
            if unlocked.startswith(ERROR_PREFIX):
                return unlocked
        result: str = self._toolkit.call_discoverable_agent_tool(
            name, json.dumps(kwargs, default=str)
        )
        return result

    def _method(self, source: str, name: str) -> Callable[..., str]:
        if source == "discoverable":
            return self._toolkit.get_discoverable_tools()[name]  # type: ignore[no-any-return]
        return getattr(self._toolkit, name)  # type: ignore[no-any-return]

    def _execute(
        self, source: str, name: str, kwargs: dict[str, Any]
    ) -> tuple[bool, str, Optional[str]]:
        try:
            if self._fidelity is Fidelity.TAU2 and source == "discoverable":
                text = self._through_tau2_dispatcher(name, kwargs)
            else:
                text = self._method(source, name)(**kwargs)
        except Exception as exc:  # tau2 should not raise, but a tool bug must not escape
            message = f"{type(exc).__name__}: {exc}"
            return False, message, message
        if text.startswith(ERROR_PREFIX):
            return False, text, text[len(ERROR_PREFIX) :].strip() or text
        return True, text, None


def _failure(name: str, error: str) -> ToolResult:
    return ToolResult(tool=name, ok=False, text=f"{ERROR_PREFIX} {error}", error=error)
