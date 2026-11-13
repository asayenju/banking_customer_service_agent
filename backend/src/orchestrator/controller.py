"""The step that decides what to do next with a request, and Jev as one of its tools.

The threshold gate used to decide alone: a number below a line meant a question or a person, however clear the
conversation was. Here a model reads the conversation, the question on screen and what has already been asked, may ask
Jev what it thinks (as advice, not a verdict), and then chooses: carry on, ask one specific question, or hand to a
person. Code keeps the rails it always had: identity, a yes to every irreversible plan, ownership and validators. If
the controller fails or breaks a rail, the old threshold decision is used for that request.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, Protocol


class Choice(StrEnum):
    RUN = "run"  # go ahead: the rails still ask for details that are missing and for the customer's yes
    ASK = "ask"  # ask one specific question
    ESCALATE = "escalate"  # a person must take over


@dataclass(frozen=True)
class ControlChoice:
    action: Choice
    question: str = ""  # for ASK: the one question, in plain words
    reason: str = ""


@dataclass(frozen=True)
class ControlCase:
    """Everything the controller may use for one request. Secrets are already hidden."""

    request_kind: str
    request_summary: str
    details: dict[str, str]  # what was found, by name (values redacted)
    missing: tuple[str, ...]
    message: str  # the customer's latest message
    recent: tuple[tuple[str, str], ...]  # the last turns, oldest first
    on_screen: str  # the question the customer is answering now, or ""
    already_asked: tuple[str, ...]  # every question put to the customer about this request
    asks_so_far: int
    what_it_can_do: str  # what the assistant really does for this kind
    safety: str  # "read only" | "reversible change" | "irreversible change (the customer must say yes to a plan)"
    products: str
    consult_jev: Callable[[], Awaitable[str]] = field(repr=False, default=lambda: _no_jev())


async def _no_jev() -> str:
    return "Jev is not available."


class Controller(Protocol):
    async def decide(self, case: ControlCase) -> Optional[ControlChoice]: ...


class ScriptedController:
    """Hands out its choices in order and records the cases it was shown (for tests)."""

    def __init__(self, choices: Sequence[ControlChoice | Exception | None]) -> None:
        self._choices = list(choices)
        self.cases: list[ControlCase] = []

    async def decide(self, case: ControlCase) -> Optional[ControlChoice]:
        self.cases.append(case)
        if not self._choices:
            raise AssertionError("controller called more often than scripted")
        step = self._choices.pop(0)
        if isinstance(step, Exception):
            raise step
        return step
