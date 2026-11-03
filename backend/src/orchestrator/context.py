"""What a route runner is given besides the intent.

A runner used to receive only the ``Intent``: the customer's words and the details the
decomposer pulled out of them. A runner that acts on an account needs more, and some of
it must not come from those words:

- ``verified_user_id`` is the caller the orchestrator verified (2 of 4 fields matched
  by code). It is copied from ``AuthState`` and nowhere else, so a slot, a summary or a
  model cannot name a different customer. It is None for kinds open to anyone.
- ``session`` is this intent's handle on the banking tools. Writes made through it are
  tagged with the intent, so when a route stalls or fails the executor can tell what was
  already committed and report it as PARTIAL instead of pretending nothing happened.
- ``deadline`` is the loop time by which the route must be done; a runner that loops
  should budget against it instead of waiting to be cancelled.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from banking_tools.types import CompletedStep


class AgentSession(Protocol):
    """The slice of a banking session a runner and the executor rely on."""

    def call(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> Any: ...

    def committed_steps(self) -> Sequence[str]:
        """Plain-language lines for the writes this intent has already committed."""
        ...

    def completed(self) -> Sequence[CompletedStep]:
        """The writes this intent has already committed, in the order they were made."""
        ...


@dataclass(frozen=True)
class RouteContext:
    intent_id: str
    verified_user_id: Optional[str] = None
    session: Optional[AgentSession] = None
    deadline: Optional[float] = None
    # Writes committed by an earlier, partly finished run of this same intent. The
    # session skips them if the runner repeats them in the same order.
    progress: tuple[CompletedStep, ...] = ()
