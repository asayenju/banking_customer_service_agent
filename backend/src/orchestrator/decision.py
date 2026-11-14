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

from orchestrator.capabilities import LABEL
from orchestrator.bands import WRITE_KIND_ASK_AT, Band, Risk, band
from orchestrator.plan import READ_ONLY_KINDS
from orchestrator.routes import Route, route_for
from orchestrator.scoring import IntentScore
from orchestrator.types import Intent, IntentKind, Question

# policy_header.md: transfer to a person only after the customer has asked this many times, except when
# the work they came for is finished or the knowledge base offers no action.
HUMAN_AFTER_ASKS = 4
CONFIRM_SLOT = "confirmation"
CLARIFY_SLOT = "clarification"

# Reads, and writes the catalog labels reversible. Anything not listed needs a yes.
NO_YES_NEEDED = frozenset(
    {
        IntentKind.FREEZE_CARD,
        IntentKind.UNFREEZE_CARD,
        IntentKind.ACCOUNT_BALANCE_OR_HISTORY,
        IntentKind.LOOKUP_INFO,
        IntentKind.GENERAL_QUESTION,
        IntentKind.UNCLEAR_OR_GREETING,
    }
)


class Action(StrEnum):
    RUN = "run"
    # Work out exactly what would be done, without doing it, and ask the customer to
    # approve that plan. Used for routes that propose plans.
    PREVIEW = "preview"
    ASK = "ask"
    ESCALATE = "escalate"
    # Noted and not acted on yet: a request for a person while a card-protection plan is open.
    DEFER = "defer"


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


_REVERSIBLE_KINDS = frozenset({IntentKind.FREEZE_CARD, IntentKind.UNFREEZE_CARD})


def risk_of(kind: IntentKind) -> Risk:
    if kind in READ_ONLY_KINDS:
        return Risk.READ
    if kind is IntentKind.REPORT_FRAUD_OR_LOSS:
        return Risk.URGENT
    if kind in _REVERSIBLE_KINDS:
        return Risk.REVERSIBLE
    return Risk.IRREVERSIBLE


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


def _ask_for_details(intent: Intent, reason: str, alternative: Optional[str] = None) -> IntentDecision:
    slots = intent.missing_slots or (CLARIFY_SLOT,)
    text = _either(intent, alternative) if slots == (CLARIFY_SLOT,) else None
    return IntentDecision(
        intent_id=intent.id,
        action=Action.ASK,
        reason=reason,
        questions=tuple(Question(intent_id=intent.id, slot=s, text=text if s == CLARIFY_SLOT else None) for s in slots),
    )


def _either(intent: Intent, alternative: Optional[str]) -> Optional[str]:
    """"Do you want me to X, or Y?" from the two readings the scorer found most likely, in words written here."""
    if not alternative:
        return None
    try:
        other = IntentKind(alternative)
    except ValueError:
        return None
    if other is intent.kind:
        return None
    return f"Do you want me to {LABEL[intent.kind]}, or {LABEL[other]}? If it is neither, tell me more."



def decide(
    intent: Intent, score: IntentScore, confirmed: Mapping[str, str]
) -> IntentDecision:
    """``confirmed`` maps an intent id to the fingerprint the customer said yes to."""
    if route_for(intent.kind) is Route.HAND_OFF:
        return _escalate(intent, "this kind hands off to a human")
    if score.error is not None:
        # The scorer being down says nothing about the customer: ask what they mean, never hand over for it.
        return _ask_for_details(intent.model_copy(update={"missing_slots": ()}), f"could not be scored: {score.error}")
    # Missing details score slots at 0, which drags the minimum to 0. That is a reason
    # to ask for them, not to give up on the request, so only how well the request is
    # understood (kind and scope) can escalate before the missing details are asked for.
    understood = min(score.kind_confidence, score.scope_confidence)
    risk = risk_of(intent.kind)
    if band(understood, risk) is Band.ESCALATE:
        # A first look that is not understood is asked about, with the customer's own words repeated back. A person
        # comes only after the repairs fail (the ask limit), when they ask for one, or for kinds that always hand off.
        return _ask_for_details(intent.model_copy(update={"missing_slots": ()}), "not understood well enough: asking what they mean", score.alternative)
    if intent.missing_slots:
        return _ask_for_details(intent, "details are missing")
    confidence_band = band(score.confidence, risk)
    if confidence_band is Band.ESCALATE:
        return _ask_for_details(intent.model_copy(update={"missing_slots": ()}), "not understood well enough: asking what they mean", score.alternative)
    if confidence_band is Band.ASK:
        return _ask_for_details(intent, "not sure enough to act yet", score.alternative)
    if intent.kind not in READ_ONLY_KINDS and score.kind_confidence < WRITE_KIND_ASK_AT:
        return _ask_for_details(intent, "not sure enough the request is right to act yet")
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
