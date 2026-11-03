"""The exact list of changes a customer is asked to approve with one yes.

A specialist that would change something does not run first and explain later. It
proposes a plan: an ordered list of writes with their arguments. The customer sees that
list, written out by code, and says yes to it. The yes is bound to a fingerprint of the
plan (and, in the orchestrator, to the intent's own fingerprint), so a changed step, a
changed amount or an extra write is a different plan that needs a new yes.

Rules this module enforces:
  - The fingerprint and the customer-facing text are computed here from the steps. A
    runner cannot supply its own, so what was shown is what was hashed.
  - Secrets never enter a plan. A PIN or CVV argument must be a reference like
    ``{{secret:new_pin}}``; the value is substituted at call time. A raw secret is
    rejected, because even a hash of a 4-digit PIN can be reversed.
  - References to the result of an earlier step (``{{step_2.account_id}}``) are part of
    the plan as text. They are resolved by code when the plan runs.
  - Whether a step can be undone comes from the tool catalog, set by whoever builds the
    plan from it, never from the model.
"""

import hashlib
import json
import re
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from banking_tools.session import REDACTED_ARGUMENTS

_SECRET_REF = re.compile(r"^\{\{secret:[a-z_]+\}\}$")
_REF = re.compile(r"^\{\{(secret:[a-z_]+|step_\d+\.[a-z_]+)\}\}$")


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class PlanStep(_Model):
    index: Annotated[int, Field(ge=1)]
    tool: Annotated[str, Field(min_length=1)]
    args: dict[str, Any]
    # From the catalog: can a later tool put things back as they were?
    reversible: bool
    # What this step does, in the customer's terms, written by code.
    description: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def _no_raw_secrets(self) -> Self:
        for key, value in self.args.items():
            if key in REDACTED_ARGUMENTS and not (isinstance(value, str) and _SECRET_REF.match(value)):
                raise ValueError(f"'{key}' must be a secret reference, not a value")
        return self


def is_reference(value: Any) -> bool:
    return isinstance(value, str) and _REF.match(value) is not None


def plan_fingerprint(steps: tuple[PlanStep, ...]) -> str:
    """Stable under argument order. Covers tool and arguments, not the wording."""
    canonical = json.dumps(
        [{"tool": s.tool, "args": s.args} for s in steps],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _shown(value: Any) -> str:
    if isinstance(value, str) and _SECRET_REF.match(value):
        return "(kept private)"
    if isinstance(value, str) and is_reference(value):
        return "(from an earlier step)"
    return str(value)


def render_plan(steps: tuple[PlanStep, ...]) -> str:
    """The text the customer approves. Built here so it always matches the steps."""
    lines = ["Here is exactly what I will do:"]
    for step in steps:
        details = ", ".join(f"{k.replace('_', ' ')}: {_shown(v)}" for k, v in sorted(step.args.items()))
        undo = "can be undone" if step.reversible else "cannot be undone"
        lines.append(f"{step.index}. {step.description}" + (f" ({details})" if details else "") + f" - {undo}")
    lines.append("Shall I go ahead? (yes or no)")
    return "\n".join(lines)


class PlanPreview(_Model):
    intent_id: Annotated[str, Field(min_length=1)]
    # decision.fingerprint(intent) when the plan was proposed; a changed detail makes it stale.
    intent_fp: Annotated[str, Field(min_length=1)]
    steps: Annotated[tuple[PlanStep, ...], Field(min_length=1)]
    plan_fp: str
    rendered: str

    @classmethod
    def build(cls, intent_id: str, intent_fp: str, steps: tuple[PlanStep, ...]) -> Self:
        return cls(
            intent_id=intent_id,
            intent_fp=intent_fp,
            steps=steps,
            plan_fp=plan_fingerprint(steps),
            rendered=render_plan(steps),
        )

    @model_validator(mode="after")
    def _computed_here(self) -> Self:
        if [s.index for s in self.steps] != list(range(1, len(self.steps) + 1)):
            raise ValueError("steps must be numbered 1, 2, 3 ... in order")
        if self.plan_fp != plan_fingerprint(self.steps):
            raise ValueError("plan_fp does not match the steps")
        if self.rendered != render_plan(self.steps):
            raise ValueError("rendered text does not match the steps")
        return self

    @property
    def reversible_only(self) -> bool:
        return all(step.reversible for step in self.steps)

    def allows(self, tool: str, arguments: dict[str, Any]) -> bool:
        """Is this exact call one of the approved steps? Reference fields match anything."""
        given = {k: v for k, v in arguments.items() if v is not None}
        for step in self.steps:
            planned = {k: v for k, v in step.args.items() if v is not None}
            if step.tool == tool and set(planned) == set(given):
                if all(is_reference(v) or given[k] == v for k, v in planned.items()):
                    return True
        return False

