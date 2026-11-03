"""The code that stands between a model's tool call and the bank, whatever the model says.

A model chooses *within* what this module lets through; it cannot widen it. Nothing here
reads model text, so an instruction hidden in a customer message, a merchant name or a
policy document can change which allowed call is made, never which calls are allowed.

Checks, in order, for every call:
  1. The tool is on the kind's allow-list (writes) or the shared read list.
  2. Every id in the arguments (card, account, credit card account, transaction)
     resolves, in the database as it is now, to a record owned by the verified customer.
     An id that resolves to someone else, or to nothing, is refused. This is what stops
     a transfer to a third party's account and a lookup of another customer's card.
  3. The verified customer is the only ``user_id`` that can appear.

``GuardedSession`` applies those checks around any session. The plan check (a write must
be one of the steps the customer approved) lives in ``ScopedSession``; both can be stacked.

``unapproved_writes`` is the after-the-fact check: it compares what a run committed with
what was approved. The pre-call checks are the enforcement; this is the alarm for a hole
in them, and it cannot undo anything.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional, Protocol

from banking_tools.catalog import get_registry
from banking_tools.types import CompletedStep, ToolKind, ToolResult
from orchestrator.context import AgentSession
from orchestrator.plan_preview import PlanPreview, unapproved_steps
from orchestrator.types import IntentKind

K = IntentKind

# Writes the orchestrator makes itself. No model-driven route may call them.
ORCHESTRATOR_ONLY: frozenset[str] = frozenset({"log_verification"})

# Reads that look up customers by name or email would let a model find anyone.
FORBIDDEN_READS: frozenset[str] = frozenset(
    {"get_user_information_by_name", "get_user_information_by_email"}
)

_CARD_PROTECTION = frozenset(
    {
        "freeze_debit_card_3892",
        "unfreeze_debit_card_3893",
        "close_debit_card_4721",
        "order_debit_card_5739",
        "clear_debit_card_fraud_alert_4892",
    }
)
_DISPUTES = frozenset(
    {"file_credit_card_transaction_dispute_4829", "file_debit_card_transaction_dispute_6281"}
)

# Hand-written: it cannot be derived from taxonomy (which files close_debit_card under
# CLOSE_CARD, while a replacement flow also needs it). Generous inside a cluster on
# purpose, since the customer-approved plan is the tight fence; tight across clusters.
KIND_WRITE_TOOLS: Mapping[IntentKind, frozenset[str]] = {
    K.FREEZE_CARD: frozenset({"freeze_debit_card_3892"}),
    K.UNFREEZE_CARD: frozenset({"unfreeze_debit_card_3893"}),
    K.UPDATE_PROFILE: frozenset({"change_user_email"}),
    K.ACTIVATE_CARD: frozenset(
        {"activate_debit_card_8291", "activate_debit_card_8292", "activate_debit_card_8293"}
    ),
    K.CHANGE_OR_RESET_PIN: frozenset({"change_debit_card_pin_6285", "reset_debit_card_pin_6284"}),
    K.CARD_CONTROLS: frozenset(
        {
            "set_debit_card_recurring_block_7382",
            "clear_debit_card_fraud_alert_4892",
            "request_temporary_debit_card_limit_increase_8374",
        }
    ),
    K.ORDER_REPLACEMENT_CARD: frozenset(
        {
            "close_debit_card_4721",
            "order_debit_card_5739",
            "activate_debit_card_8291",
            "activate_debit_card_8292",
            "activate_debit_card_8293",
            "order_replacement_credit_card_7291",
        }
    ),
    K.DISPUTE_TRANSACTION: _DISPUTES | _CARD_PROTECTION | {"order_replacement_credit_card_7291"},
    K.CLOSE_CARD: frozenset(
        {
            "close_debit_card_4721",
            "log_credit_card_closure_reason_4521",
            "close_credit_card_account_7834",
            "apply_statement_credit_8472",
            "apply_credit_card_account_flag_6147",
            "pay_credit_card_from_checking_9182",
        }
    ),
    K.CREDIT_LIMIT_REQUEST: frozenset(
        {
            "submit_credit_limit_increase_request_7392",
            "approve_credit_limit_increase_5847",
            "deny_credit_limit_increase_5848",
            "request_temporary_debit_card_limit_increase_8374",
            "pay_credit_card_from_checking_9182",
        }
    ),
    K.FEE_OR_INTEREST_ADJUSTMENT: frozenset(
        {
            "update_transaction_rewards_3847",
            "apply_statement_credit_8472",
            "apply_checking_account_credit_5829",
            "apply_savings_account_credit_6831",
            "submit_interest_discrepancy_report_7294",
        }
    ),
    K.TRANSFER_FUNDS: frozenset({"transfer_funds_between_bank_accounts_7291"}),
    K.PAY_CREDIT_CARD: frozenset({"pay_credit_card_from_checking_9182"}),
    K.OPEN_ACCOUNT: frozenset({"open_bank_account_4821", "transfer_funds_between_bank_accounts_7291"}),
    K.CLOSE_ACCOUNT: frozenset({"close_bank_account_7392", "transfer_funds_between_bank_accounts_7291"}),
    # A composer: the protective steps of the debit and credit flows, nothing of its own.
    K.REPORT_FRAUD_OR_LOSS: _CARD_PROTECTION
    | _DISPUTES
    | {"activate_debit_card_8292", "order_replacement_credit_card_7291"},
}


def allowed_writes(kind: IntentKind) -> frozenset[str]:
    """Writes a route for this kind may make. A kind not listed may make none."""
    return KIND_WRITE_TOOLS.get(kind, frozenset())


def shared_reads() -> frozenset[str]:
    return frozenset(
        name
        for name, spec in get_registry().items()
        if spec.kind is ToolKind.READ and name not in FORBIDDEN_READS
    )


def uncovered_catalog_writes() -> list[str]:
    """Catalog writes no kind may use and the orchestrator does not own: a gap to close on purpose."""
    covered = set().union(*KIND_WRITE_TOOLS.values()) | ORCHESTRATOR_ONLY
    return sorted(
        name
        for name, spec in get_registry().items()
        if spec.kind is ToolKind.WRITE and name not in covered
    )


class Why(StrEnum):
    ALLOWED = "allowed"
    NOT_ALLOWED_FOR_KIND = "tool_not_allowed_for_this_request"
    FORBIDDEN_READ = "lookup_of_other_customers_is_not_allowed"
    OTHER_CUSTOMER = "record_belongs_to_another_customer"
    UNKNOWN_RECORD = "record_not_found"
    NO_CUSTOMER = "no_verified_customer"


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    why: Why
    # Which argument tripped it, never its value.
    argument: Optional[str] = None


class Ownership(Protocol):
    def owner_of(self, argument: str, value: str) -> Optional[str]:
        """The user who owns the record ``value`` names for ``argument``; None if there is none."""
        ...


# Argument names that hold a record id, and the order to look for it in.
_ID_ARGUMENTS = (
    "account_id",
    "checking_account_id",
    "source_account_id",
    "destination_account_id",
    "card_id",
    "credit_card_account_id",
    "transaction_id",
)


class DbOwnership:
    """Reads ownership from the session's database as it is right now."""

    def __init__(self, database: Any) -> None:
        self._db = database

    def _row(self, table: str, key: str) -> Optional[Mapping[str, Any]]:
        row = getattr(self._db, table).data.get(key)
        return row if isinstance(row, Mapping) else None

    def _account_owner(self, key: str) -> Optional[str]:
        for table in ("accounts", "credit_card_accounts"):
            row = self._row(table, key)
            if row is not None:
                owner = row.get("user_id")
                return str(owner) if owner is not None else None
        return None

    def owner_of(self, argument: str, value: str) -> Optional[str]:
        if argument == "card_id":
            row = self._row("debit_cards", value)
            return str(row["user_id"]) if row and "user_id" in row else None
        if argument == "credit_card_account_id":
            row = self._row("credit_card_accounts", value)
            return str(row["user_id"]) if row and "user_id" in row else None
        if argument == "transaction_id":
            row = self._row("credit_card_transaction_history", value)
            if row is not None:
                return str(row["user_id"]) if "user_id" in row else None
            row = self._row("bank_account_transaction_history", value)
            return self._account_owner(str(row["account_id"])) if row and "account_id" in row else None
        return self._account_owner(value)


class Guard:
    def __init__(
        self,
        *,
        kind: IntentKind,
        user_id: Optional[str],
        ownership: Ownership,
        extra_reads: Collection[str] = (),
    ) -> None:
        self._kind = kind
        self._user_id = user_id
        self._ownership = ownership
        self._writes = allowed_writes(kind)
        self._reads = shared_reads() | frozenset(extra_reads)

    def check(self, tool: str, arguments: Optional[Mapping[str, Any]] = None) -> Verdict:
        spec = get_registry().get(tool)
        if spec is None:
            return Verdict(False, Why.NOT_ALLOWED_FOR_KIND)
        if tool in FORBIDDEN_READS:
            return Verdict(False, Why.FORBIDDEN_READ)
        permitted = self._reads if spec.kind is ToolKind.READ else self._writes
        if spec.kind is ToolKind.HANDOFF:
            return Verdict(False, Why.NOT_ALLOWED_FOR_KIND)  # hand-offs are the orchestrator's
        if tool not in permitted:
            return Verdict(False, Why.NOT_ALLOWED_FOR_KIND)
        if self._user_id is None and (tool in self._writes or _names_a_customer(spec)):
            return Verdict(False, Why.NO_CUSTOMER)
        args = dict(arguments or {})
        if "user_id" in args and args["user_id"] != self._user_id:
            return Verdict(False, Why.OTHER_CUSTOMER, "user_id")
        for name in _ID_ARGUMENTS:
            value = args.get(name)
            if value is None:
                continue
            owner = self._ownership.owner_of(name, str(value))
            if owner is None:
                return Verdict(False, Why.UNKNOWN_RECORD, name)
            if owner != self._user_id:
                return Verdict(False, Why.OTHER_CUSTOMER, name)
        return Verdict(True, Why.ALLOWED)


def _names_a_customer(spec: Any) -> bool:
    return spec.args_model is not None and any(
        f in spec.args_model.model_fields for f in ("user_id", *_ID_ARGUMENTS)
    )


@dataclass
class GuardedSession:
    """Any session with the guard in front. A refused call returns a failed result and is logged."""

    inner: AgentSession
    guard: Guard
    blocked: list[tuple[str, Why, Optional[str]]] = field(default_factory=list)

    def call(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> ToolResult:
        verdict = self.guard.check(name, arguments)
        if not verdict.allowed:
            self.blocked.append((name, verdict.why, verdict.argument))
            reason = f"that is not permitted ({verdict.why.value})"
            return ToolResult(tool=name, ok=False, text=f"Error: {reason}", error=reason)
        result: ToolResult = self.inner.call(name, arguments)
        return result

    def committed_steps(self) -> Sequence[str]:
        return self.inner.committed_steps()

    def completed(self) -> Sequence[CompletedStep]:
        return self.inner.completed()


def unapproved_writes(
    completed: Sequence[CompletedStep], approved: Optional[PlanPreview]
) -> list[CompletedStep]:
    """See ``orchestrator.plan_preview.unapproved_steps``; named here for the guard's callers."""
    return unapproved_steps(completed, approved)
