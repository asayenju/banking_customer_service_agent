"""The one runner behind ``Route.SUB_AGENT``: pick the right specialist and hand over.

The orchestrator has a single slot for every kind that needs policy reasoning. This runner
fills it. It looks at the kind and, for the three that span debit and credit products,
at the record the customer named:

  - DISPUTE_TRANSACTION, CLOSE_CARD and ORDER_REPLACEMENT_CARD: a debit card, a checking
    account or a bank transaction means the debit specialist; a credit card account or a
    credit card transaction means the credit specialist. That lookup is code against the
    database, scoped to the verified customer, so another customer's id resolves to nothing.
  - If nothing is named, or both kinds of record are, the customer is asked; the model
    never picks.

A kind with no specialist built yet, or a specialist that is not wired in, returns a plain
failure ("not implemented"), so the system can be rolled out one specialist at a time
without a route that silently does nothing. ``proposes_plans`` tells the orchestrator to
ask this runner for a plan first and take the customer's yes to that plan.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Optional

from orchestrator.context import RouteContext
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.routes import ROUTES, Route
from orchestrator.types import Intent, IntentKind, Question

from agent.specialist import Specialist

K = IntentKind


class Name(StrEnum):
    DEBIT = "debit"
    CREDIT = "credit"
    ACCOUNTS = "accounts"
    FEES = "fees"
    FRAUD = "fraud"
    GENERAL = "general"


# Kinds with one obvious owner.
FIXED: Mapping[IntentKind, Name] = {
    K.ACTIVATE_CARD: Name.DEBIT,
    K.CHANGE_OR_RESET_PIN: Name.DEBIT,
    K.CARD_CONTROLS: Name.DEBIT,
    K.TRANSFER_FUNDS: Name.ACCOUNTS,
    K.PAY_CREDIT_CARD: Name.ACCOUNTS,
    K.OPEN_ACCOUNT: Name.ACCOUNTS,
    K.CLOSE_ACCOUNT: Name.ACCOUNTS,
    K.FEE_OR_INTEREST_ADJUSTMENT: Name.FEES,
    K.REPORT_FRAUD_OR_LOSS: Name.FRAUD,
    K.GENERAL_QUESTION: Name.GENERAL,
}
# Kinds that belong to debit or credit depending on the record the customer means.
BY_RECORD: frozenset[IntentKind] = frozenset(
    {K.DISPUTE_TRANSACTION, K.CLOSE_CARD, K.ORDER_REPLACEMENT_CARD, K.CREDIT_LIMIT_REQUEST}
)


def sub_agent_kinds() -> frozenset[IntentKind]:
    return frozenset(k for k, r in ROUTES.items() if r is Route.SUB_AGENT)


def unmapped_kinds() -> frozenset[IntentKind]:
    """Sub-agent kinds nobody has decided an owner for. Must be empty."""
    return sub_agent_kinds() - set(FIXED) - BY_RECORD


class RecordKinds:
    """Which products the records a customer named belong to, from the database as it is."""

    def __init__(self, database: Any) -> None:
        self._db = database

    def _owned(self, table: str, key: str, user_id: str) -> bool:
        row = getattr(self._db, table).data.get(key)
        return isinstance(row, Mapping) and row.get("user_id") == user_id

    def _by_last_four(self, value: str, user_id: str) -> bool:
        """The customer may have said the card's last four digits, not its id."""
        return value.isdigit() and len(value) == 4 and any(
            c.get("user_id") == user_id and c.get("last_4_digits") == value
            for c in self._db.debit_cards.data.values()
        )

    def products(self, slots: Mapping[str, str], user_id: str) -> frozenset[Name]:
        found: set[Name] = set()
        for name, value in slots.items():
            if not value:
                continue
            if name == "card_id" and (
                self._owned("debit_cards", value, user_id) or self._by_last_four(value, user_id)
            ):
                found.add(Name.DEBIT)
            elif name == "credit_card_account_id" and self._owned("credit_card_accounts", value, user_id):
                found.add(Name.CREDIT)
            elif name == "account_id":
                if self._owned("accounts", value, user_id):
                    found.add(Name.DEBIT)
                elif self._owned("credit_card_accounts", value, user_id):
                    found.add(Name.CREDIT)
            elif name == "transaction_id":
                if self._owned("credit_card_transaction_history", value, user_id):
                    found.add(Name.CREDIT)
                else:
                    row = getattr(self._db, "bank_account_transaction_history").data.get(value)
                    if isinstance(row, Mapping) and self._owned("accounts", str(row.get("account_id")), user_id):
                        found.add(Name.DEBIT)
        return frozenset(found)


class SubAgentDispatcher:
    proposes_plans = True

    def __init__(self, specialists: Mapping[Name, Specialist], records: RecordKinds) -> None:
        self._specialists = dict(specialists)
        self._records = records

    def specialist_name(self, intent: Intent, user_id: Optional[str]) -> Name | list[str]:
        """The specialist for this intent, or the details to ask for when it cannot be told."""
        if intent.kind in FIXED:
            return FIXED[intent.kind]
        if intent.kind not in BY_RECORD:
            raise KeyError(f"no specialist is decided for {intent.kind.value}")
        found = self._records.products(intent.slots, user_id) if user_id else frozenset()
        if len(found) == 1:
            return next(iter(found))
        if intent.kind is K.CREDIT_LIMIT_REQUEST and not found:
            return Name.CREDIT  # a card limit request with no record named is a credit one
        return ["card_or_account_id"]

    async def run(self, intent: Intent, ctx: RouteContext) -> RouteOutcome:
        try:
            chosen = self.specialist_name(intent, ctx.verified_user_id)
        except KeyError:
            return RouteOutcome(
                intent_id=intent.id, status=OutcomeStatus.FAILED, error="no specialist for this kind"
            )
        if isinstance(chosen, list):
            return RouteOutcome(
                intent_id=intent.id,
                status=OutcomeStatus.NEEDS_INFO,
                questions=tuple(Question(intent_id=intent.id, slot=s) for s in chosen),
            )
        specialist = self._specialists.get(chosen)
        if specialist is None:
            return RouteOutcome(
                intent_id=intent.id,
                status=OutcomeStatus.FAILED,
                error=f"the {chosen.value} specialist is not implemented yet",
            )
        return await specialist.run(intent, ctx)
