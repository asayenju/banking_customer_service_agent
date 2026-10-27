"""What to do with one intent: run it, ask the customer, or escalate.

The confidence band is necessary but never sufficient. An intent that can do
something irreversible runs only after the customer has said yes to exactly
what will happen, at any confidence. The yes is bound to a fingerprint of the
intent's kind and details; if a detail changes afterwards the yes is stale and
the customer is asked again.

The list of kinds that may run without a yes is explicit, so a new kind
defaults to needing one. tests/orchestrator/test_decision.py checks the list
against the tool catalog: a kind that can reach an irreversible tool may not be
on it.

Known limit: the yes covers the intent's own details. A sub-agent that decides
to take further irreversible steps beyond them is out of this module's reach.
"""

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Optional, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orchestrator.bands import Band, band
from orchestrator.routes import Route, route_for
from orchestrator.scoring import IntentScore
from orchestrator.types import Intent, IntentKind, Question

CONFIRM_SLOT = "confirmation"
CLARIFY_SLOT = "clarification"

# Reads, and writes the catalog labels reversible. Anything not listed needs a yes.
NO_YES_NEEDED = frozenset(
    {
        IntentKind.FREEZE_CARD,
        IntentKind.UNFREEZE_CARD,
        IntentKind.UPDATE_PROFILE,
        IntentKind.ACCOUNT_BALANCE_OR_HISTORY,
        IntentKind.LOOKUP_INFO,
        IntentKind.GENERAL_QUESTION,
        IntentKind.UNCLEAR_OR_GREETING,
    }
)


class Action(StrEnum):
    RUN = "run"
    ASK = "ask"
    ESCALATE = "escalate"


class IntentDecision(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    intent_id: Annotated[str, Field(min_length=1)]
    action: Action
    reason: Annotated[str, Field(min_length=1)]
    questions: tuple[Question, ...] = ()
    # What a yes would bind to. Set only when the question is the confirmation.
    fingerprint: Optional[str] = None

    @model_validator(mode="after")
    def _questions_match_the_action(self) -> Self:
        if self.action is Action.ASK and not self.questions:
            raise ValueError("an ask decision must carry at least one question")
        if self.action is not Action.ASK and self.questions:
            raise ValueError(f"a {self.action.value} decision must not carry questions")
        return self


def needs_confirmation(kind: IntentKind) -> bool:
    return kind not in NO_YES_NEEDED


def fingerprint(intent: Intent) -> str:
    """Identifies exactly what a yes covers. A hash, so a PIN in the slots never
    sits in confirmation state in the clear."""
    canonical = json.dumps(
        {"kind": intent.kind.value, "slots": intent.slots},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _escalate(intent: Intent, reason: str) -> IntentDecision:
    return IntentDecision(intent_id=intent.id, action=Action.ESCALATE, reason=reason)


def _ask_for_details(intent: Intent, reason: str) -> IntentDecision:
    slots = intent.missing_slots or (CLARIFY_SLOT,)
    return IntentDecision(
        intent_id=intent.id,
        action=Action.ASK,
        reason=reason,
        questions=tuple(Question(intent_id=intent.id, slot=s) for s in slots),
    )


def decide(
    intent: Intent, score: IntentScore, confirmed: Mapping[str, str]
) -> IntentDecision:
    """``confirmed`` maps an intent id to the fingerprint the customer said yes to."""
    if route_for(intent.kind) is Route.HAND_OFF:
        return _escalate(intent, "this kind hands off to a human")
    if score.error is not None:
        return _escalate(intent, f"could not be scored: {score.error}")
    confidence_band = band(score.confidence)
    if confidence_band is Band.ESCALATE:
        return _escalate(intent, "confidence below the escalation threshold")
    if intent.missing_slots:
        return _ask_for_details(intent, "details are missing")
    if confidence_band is Band.ASK:
        return _ask_for_details(intent, "not sure enough to act yet")
    if not needs_confirmation(intent.kind):
        return IntentDecision(intent_id=intent.id, action=Action.RUN, reason="ready")

    current = fingerprint(intent)
    said_yes_to = confirmed.get(intent.id)
    if said_yes_to == current:
        return IntentDecision(
            intent_id=intent.id, action=Action.RUN, reason="confirmed by the customer"
        )
    reason = (
        "confirmation is stale: the details changed"
        if said_yes_to is not None
        else "awaiting the customer's yes"
    )
    return IntentDecision(
        intent_id=intent.id,
        action=Action.ASK,
        reason=reason,
        questions=(Question(intent_id=intent.id, slot=CONFIRM_SLOT),),
        fingerprint=current,
    )
