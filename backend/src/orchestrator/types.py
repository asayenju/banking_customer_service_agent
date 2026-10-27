"""Typed shapes the orchestrator passes around: intents, a decomposition, a decision.

Models are strict, so model output must come in through ``model_validate_json``
(JSON mode turns strings into enums); passing a bare string for ``kind`` in
Python is rejected on purpose. The intent list is closed: anything the model
cannot place is ``OTHER``, and it may not invent kinds.
"""

from enum import StrEnum
from typing import Annotated, Literal, Optional, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class IntentKind(StrEnum):
    FREEZE_CARD = "FREEZE_CARD"
    UNFREEZE_CARD = "UNFREEZE_CARD"
    ACTIVATE_CARD = "ACTIVATE_CARD"
    ORDER_REPLACEMENT_CARD = "ORDER_REPLACEMENT_CARD"
    CHANGE_OR_RESET_PIN = "CHANGE_OR_RESET_PIN"
    CLOSE_CARD = "CLOSE_CARD"
    CARD_CONTROLS = "CARD_CONTROLS"
    DISPUTE_TRANSACTION = "DISPUTE_TRANSACTION"
    CREDIT_LIMIT_REQUEST = "CREDIT_LIMIT_REQUEST"
    FEE_OR_INTEREST_ADJUSTMENT = "FEE_OR_INTEREST_ADJUSTMENT"
    TRANSFER_FUNDS = "TRANSFER_FUNDS"
    PAY_CREDIT_CARD = "PAY_CREDIT_CARD"
    OPEN_ACCOUNT = "OPEN_ACCOUNT"
    CLOSE_ACCOUNT = "CLOSE_ACCOUNT"
    ACCOUNT_BALANCE_OR_HISTORY = "ACCOUNT_BALANCE_OR_HISTORY"
    LOOKUP_INFO = "LOOKUP_INFO"
    UPDATE_PROFILE = "UPDATE_PROFILE"
    REPORT_FRAUD_OR_LOSS = "REPORT_FRAUD_OR_LOSS"
    REQUEST_HUMAN = "REQUEST_HUMAN"
    UNCLEAR_OR_GREETING = "UNCLEAR_OR_GREETING"
    OTHER = "OTHER"


class Flag(StrEnum):
    INJECTION_SUSPECTED = "injection_suspected"
    POSSIBLE_DROPPED_INTENT = "possible_dropped_intent"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class Intent(_Model):
    """One customer goal, with the details found for it and what it waits on."""

    id: Annotated[str, Field(min_length=1)]
    kind: IntentKind
    # In the customer's own words.
    summary: str
    slots: dict[str, str] = Field(default_factory=dict)
    missing_slots: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    # Character offsets of the part of the message this intent came from.
    source_span: Optional[tuple[int, int]] = None


class Conflict(_Model):
    """Intents that cannot all be honoured, e.g. freeze and unfreeze the same card."""

    intent_ids: tuple[str, ...]
    reason: str


class Decomposition(_Model):
    intents: tuple[Intent, ...]
    conflicts: tuple[Conflict, ...] = ()
    flags: tuple[Flag, ...] = ()
    schema_version: int = 1

    @model_validator(mode="after")
    def _ids_are_unique_and_dependencies_resolve(self) -> Self:
        seen: set[str] = set()
        for intent in self.intents:
            if intent.id in seen:
                raise ValueError(f"duplicate intent id '{intent.id}'")
            seen.add(intent.id)
        for intent in self.intents:
            for dep in intent.depends_on:
                if dep not in seen:
                    raise ValueError(
                        f"intent '{intent.id}' depends on unknown intent '{dep}'"
                    )
        return self


class Question(_Model):
    """A missing detail for one intent. The wording is phrased later by an LLM."""

    intent_id: str
    slot: str
    text: Optional[str] = None


class Proceed(_Model):
    kind: Literal["proceed"] = "proceed"
    intent_ids: tuple[str, ...] = Field(min_length=1)


class AskCustomer(_Model):
    kind: Literal["ask_customer"] = "ask_customer"
    questions: tuple[Question, ...] = Field(min_length=1)


class HandOff(_Model):
    kind: Literal["hand_off"] = "hand_off"
    reason: str
    intent_ids: tuple[str, ...] = ()


Decision = Annotated[Proceed | AskCustomer | HandOff, Field(discriminator="kind")]
