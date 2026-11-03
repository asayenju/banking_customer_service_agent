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
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from functools import cache
from typing import Any, Callable, Optional, Protocol

from pydantic import ValidationError
from tau2.domains.banking_knowledge.data_model import TransactionalDB
from tau2.domains.banking_knowledge.environment import get_db
from tau2.domains.banking_knowledge.tools import KnowledgeTools
from tau2.domains.banking_knowledge.utils import get_now

from banking_tools.args import call_kwargs
from banking_tools.catalog import get_registry
from banking_tools.task_state import apply_initial_state
from banking_tools.types import AuditEntry, CompletedStep, ToolKind, ToolResult

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

    def audit_for(self, intent_id: str) -> tuple[AuditEntry, ...]:
        """The writes made for one intent, successful or not, in order."""
        with self._lock:
            return tuple(e for e in self._audit if e.intent_id == intent_id)

    def completed_for(self, intent_id: str) -> tuple[CompletedStep, ...]:
        """The writes that took effect for an intent, in order, as audited."""
        return tuple(
            CompletedStep(tool=e.tool, arguments=dict(e.arguments), audit_id=e.audit_id)
            for e in self.audit_for(intent_id)
            if e.ok
        )

    def committed_steps_for(self, intent_id: str) -> tuple[str, ...]:
        """Plain-language lines for the writes that actually took effect for an intent."""
        registry = get_registry()
        return tuple(
            registry[e.tool].description.rstrip(".") if e.tool in registry else e.tool
            for e in self.audit_for(intent_id)
            if e.ok
        )

    @property
    def audit_log(self) -> tuple[AuditEntry, ...]:
        with self._lock:
            return tuple(self._audit)

    def db_hash(self) -> str:
        with self._lock:
            return str(self._db.get_hash())

    def call(
        self,
        name: str,
        arguments: Optional[Mapping[str, Any]] = None,
        *,
        intent_id: Optional[str] = None,
    ) -> ToolResult:
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
                        intent_id=intent_id,
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


class ScopedSession:
    """One intent's view of a session.

    Every write it makes is tagged with the intent, so the executor can reconcile what
    happened if the route fails partway.

    Modes. In "plan" mode reads run and every write is refused without being attempted,
    so a route that is only working out what it would do cannot change anything. In
    "execute" mode, if the customer approved a plan, a write that is not one of the
    approved steps is refused: a yes covers what was shown and nothing else.

    Resuming. If the intent was partly done before, pass the writes it had committed as
    ``resume``: when the runner repeats them in the same order they are not made again
    and the original result is reported. Matching is by position, so a legitimate repeat
    (freeze, unfreeze, freeze) is not mistaken for a retry. The first write that differs
    ends the skipping, and everything after it runs.
    """

    def __init__(
        self,
        session: BankingSession,
        intent_id: str,
        resume: Sequence[CompletedStep] = (),
        *,
        mode: str = "execute",
        approved: Optional["ApprovedPlan"] = None,
    ) -> None:
        self._session = session
        self._intent_id = intent_id
        self._resume = tuple(resume)
        self._next = 0
        self._mode = mode
        self._approved = approved

    def call(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> ToolResult:
        spec = get_registry().get(name)
        is_write = spec is not None and spec.kind is ToolKind.WRITE
        if is_write and self._mode == "plan":
            return _refused(name, "this is planning only; changes happen after you approve the plan")
        if is_write and self._approved is not None and not self._approved.allows(
            name, _validated_arguments(name, arguments) or dict(arguments or {})
        ):
            return _refused(name, "that step is not part of the plan the customer approved")
        if self._next < len(self._resume):
            earlier = self._resume[self._next]
            audited = _audited_arguments(name, arguments)
            if audited is not None:
                if earlier.tool == name and earlier.arguments == audited:
                    self._next += 1
                    return ToolResult(
                        tool=name,
                        ok=True,
                        text="Already done earlier; not repeated.",
                        audit_id=earlier.audit_id,
                    )
                self._next = len(self._resume)  # the runner went another way
        return self._session.call(name, arguments, intent_id=self._intent_id)

    def committed_steps(self) -> tuple[str, ...]:
        return self._session.committed_steps_for(self._intent_id)

    def completed(self) -> tuple[CompletedStep, ...]:
        return self._session.completed_for(self._intent_id)


class ApprovedPlan(Protocol):
    """What a scoped session needs from an approved plan."""

    def allows(self, tool: str, arguments: dict[str, Any]) -> bool: ...


def _refused(name: str, why: str) -> ToolResult:
    return ToolResult(tool=name, ok=False, text=f"{ERROR_PREFIX} {why}", error=why)


def _validated_arguments(name: str, arguments: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    """The arguments of a write as validated, secrets included; None if they do not validate."""
    spec = get_registry().get(name)
    if spec is None or spec.args_model is None:
        return None
    try:
        dumped: dict[str, Any] = spec.args_model.model_validate(dict(arguments or {})).model_dump()
        return dumped
    except ValidationError:
        return None


def _audited_arguments(name: str, arguments: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    """The arguments of a write as the audit log would store them; None for anything else."""
    spec = get_registry().get(name)
    if spec is None or spec.kind is not ToolKind.WRITE:
        return None
    validated = _validated_arguments(name, arguments)
    return None if validated is None else _redact(validated)
