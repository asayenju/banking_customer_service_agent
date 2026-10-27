"""Who is asking: the verified-caller gate.

Confidence in an intent says nothing about whether the caller may act on the
account, so this is plain code with no scorer and no LLM. A caller is verified
when two of four identity fields (date of birth, email, phone, address) match
the customer's record. tau2 only describes that rule in a tool docstring and
``log_verification`` just writes an audit record, so it is enforced here.

Claims arrive as an untrusted mapping, usually slots an LLM pulled out of the
customer's message. Only the four known field names are read; every other key
("bypass_code", "supervisor_override", "verified") is ignored. Nothing a
customer says about authority, codes or supervisors can verify them.
"""

import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field

from orchestrator.types import Decomposition, HandOff, IntentKind, Question

IDENTITY_SLOT = "identity_verification"
REQUIRED_MATCHES = 2
MAX_FAILED_ATTEMPTS = 3

# Needs no verified caller: greetings, a request for a human, and what we cannot
# place (which hands off anyway). Everything else touches an account.
OPEN_KINDS = frozenset(
    {IntentKind.UNCLEAR_OR_GREETING, IntentKind.REQUEST_HUMAN, IntentKind.OTHER}
)

_DATE_FORMATS = ("%m/%d/%Y", "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y")


class IdentityField(StrEnum):
    DATE_OF_BIRTH = "date_of_birth"
    EMAIL = "email"
    PHONE_NUMBER = "phone_number"
    ADDRESS = "address"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class CustomerRecord(_Model):
    """What the bank has on file, to compare a caller's claims against."""

    user_id: Annotated[str, Field(min_length=1)]
    date_of_birth: str
    email: str
    phone_number: str
    address: str


class AuthState(_Model):
    verified_user_id: Optional[str] = None
    # The record the matches below belong to.
    candidate_user_id: Optional[str] = None
    matched: frozenset[IdentityField] = frozenset()
    failed_attempts: Annotated[int, Field(ge=0)] = 0

    @property
    def verified(self) -> bool:
        return self.verified_user_id is not None

    @property
    def locked(self) -> bool:
        return self.failed_attempts >= MAX_FAILED_ATTEMPTS


class GateResult(_Model):
    """Which intents may go on to scoring, and what to do about the rest."""

    allowed: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()
    # One verification question per blocked intent; empty once the caller is locked out.
    questions: tuple[Question, ...] = ()
    hand_off: Optional[HandOff] = None


def requires_verification(kind: IntentKind) -> bool:
    return kind not in OPEN_KINDS


def _normalize(field: IdentityField, value: str) -> str:
    value = value.strip()
    if field is IdentityField.EMAIL:
        return value.lower()
    if field is IdentityField.PHONE_NUMBER:
        digits = re.sub(r"\D", "", value)
        return digits[-10:]
    if field is IdentityField.ADDRESS:
        return " ".join(re.sub(r"[.,#]", " ", value).lower().split())
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return value.lower()


def _matches(field: IdentityField, claimed: str, record: CustomerRecord) -> bool:
    on_file = _normalize(field, getattr(record, field.value))
    return bool(on_file) and _normalize(field, claimed) == on_file


def verify(
    state: AuthState, record: CustomerRecord, claimed: Mapping[str, str]
) -> AuthState:
    """Apply one round of claims. A wrong field is a failed attempt; too few right
    ones is just "tell me more". Matches add up across rounds for the same record."""
    if state.verified or state.locked:
        return state
    claims = {f: v for f in IdentityField if (v := claimed.get(f.value, "").strip())}
    prior = state.matched if state.candidate_user_id == record.user_id else frozenset()
    hits = frozenset(f for f, v in claims.items() if _matches(f, v, record))
    matched = prior | hits
    if len(matched) >= REQUIRED_MATCHES:
        return AuthState(
            verified_user_id=record.user_id,
            candidate_user_id=record.user_id,
            matched=matched,
            failed_attempts=state.failed_attempts,
        )
    wrong = bool(claims.keys() - hits)
    return AuthState(
        candidate_user_id=record.user_id,
        matched=matched,
        failed_attempts=state.failed_attempts + (1 if wrong else 0),
    )


def gate(state: AuthState, decomposition: Decomposition) -> GateResult:
    allowed: list[str] = []
    blocked: list[str] = []
    for intent in decomposition.intents:
        if state.verified or not requires_verification(intent.kind):
            allowed.append(intent.id)
        else:
            blocked.append(intent.id)
    if not blocked:
        return GateResult(allowed=tuple(allowed))
    if state.locked:
        return GateResult(
            allowed=tuple(allowed),
            blocked=tuple(blocked),
            hand_off=HandOff(
                reason="identity verification failed too many times",
                intent_ids=tuple(blocked),
            ),
        )
    return GateResult(
        allowed=tuple(allowed),
        blocked=tuple(blocked),
        questions=tuple(Question(intent_id=i, slot=IDENTITY_SLOT) for i in blocked),
    )
