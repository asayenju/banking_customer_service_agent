"""The debit-card specialist: cards, replacements, PINs, card limits and debit disputes.

It is a configuration of ``Specialist``, not a class of its own: the instructions below,
the document scope, and the validators that follow. The validators are the bank's rules
written as code, checked at plan time against a copy of the database and again just
before each step runs:

  - lifecycle (this module's first half): which statuses a card must be in for each
    change, and which activation tool fits the card's issue reason;
  - ordering: the published fee values for delivery and card design;
  - PINs, limits and fraud alerts, and disputes: added to ``VALIDATORS`` below.

The bank's tools already refuse many bad calls (unknown card, wrong status). These
validators exist for two reasons: the refusal comes back in words we control and can test,
and they cover what the tools never check (a fee the agent invented, a dispute amount that
does not match the transaction).

Not encoded here, on purpose: the tier-specific fee schedules. The published values are
checked; which of them applies to a given account tier is left to the policy notes the
model reads, because the tier is not a field in the database.
"""

from collections.abc import Mapping
from typing import Any, Optional

from agent.specialist import SpecialistConfig, Validator, ValidationContext
from knowledge.scopes import DEBIT
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.types import Intent, Question
from orchestrator.types import IntentKind as K

KINDS = frozenset(
    {
        K.ACTIVATE_CARD,
        K.CHANGE_OR_RESET_PIN,
        K.CARD_CONTROLS,
        K.CLOSE_CARD,
        K.ORDER_REPLACEMENT_CARD,
        K.DISPUTE_TRANSACTION,
    }
)

INSTRUCTIONS = """You handle debit card requests: closing, replacing and activating cards, PIN changes, card
limits and alerts, and debit transaction disputes.
How to work:
1. Search the policy notes for the procedure for this request before you act. The internal
   procedures are for you; do not quote them to the customer.
2. Find the customer's cards and accounts with your read tools. If the customer did not say
   which card or account and more than one could be meant, ask. Do not choose.
3. A frozen card must be unfrozen before it can be closed. Only one card can be active on a
   checking account, so the old card is closed before a replacement is ordered for it.
4. Replacement cards: the fees are charged to the checking account. Give the exact
   delivery_fee, design_fee and excess_replacement_fee the policy gives for the account's
   tier. STANDARD delivery is always free.
5. Activation: choose the tool from the card's issue_reason. new_account or first_card:
   activate_debit_card_8291. lost, stolen or fraud: activate_debit_card_8292. expired, damaged,
   upgrade or bank_reissue: activate_debit_card_8293.
6. Disputes: fill every field from the transaction and from what the customer said. Never guess
   a date, an amount or a category. The disputed amount must match the transaction.
7. PIN changes: use the secret references that appear in the customer's message."""

CLOSE_REASONS = frozenset({"lost", "stolen", "fraud_suspected", "damaged", "no_longer_needed", "account_closing"})
# Which issue reasons each activation tool is for.
ACTIVATION_REASONS: Mapping[str, frozenset[str]] = {
    "activate_debit_card_8291": frozenset({"new_account", "first_card"}),
    "activate_debit_card_8292": frozenset({"lost", "stolen", "fraud"}),
    "activate_debit_card_8293": frozenset({"expired", "damaged", "upgrade", "bank_reissue"}),
}
# Published fee values; which one applies depends on the account tier (see module docstring).
DELIVERY_FEES: Mapping[str, frozenset[float]] = {
    "STANDARD": frozenset({0.0}),
    "EXPEDITED": frozenset({0.0, 15.0}),
    "RUSH": frozenset({0.0, 35.0}),
}
DESIGN_FEES: Mapping[str, frozenset[float]] = {
    "CLASSIC": frozenset({0.0}),
    "PREMIUM": frozenset({0.0, 10.0}),
    "CUSTOM": frozenset({0.0, 15.0, 25.0}),
}
EXCESS_REPLACEMENT_FEES = frozenset({0.0, 15.0, 25.0})


def _row(v: ValidationContext, table: str, key: Any) -> Optional[Mapping[str, Any]]:
    row = getattr(v.database, table).data.get(str(key))
    return row if isinstance(row, Mapping) else None


def _card(v: ValidationContext, args: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    return _row(v, "debit_cards", args.get("card_id"))


def freeze(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    card = _card(v, args)
    if card is None:
        return "that card does not exist"
    return None if card.get("status") == "ACTIVE" else f"only an active card can be frozen (this one is {card.get('status')})"


def unfreeze(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    card = _card(v, args)
    if card is None:
        return "that card does not exist"
    if card.get("status") != "FROZEN":
        return f"only a frozen card can be unfrozen (this one is {card.get('status')})"
    account = _row(v, "accounts", card.get("account_id"))
    if account is not None and account.get("status") != "OPEN":
        return "the account behind this card is not open"
    return None


def close(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    card = _card(v, args)
    if card is None:
        return "that card does not exist"
    if str(args.get("reason", "")).lower() not in CLOSE_REASONS:
        return "that is not a reason a card can be closed for"
    if card.get("status") == "FROZEN":
        return "a frozen card must be unfrozen before it can be closed"
    if card.get("status") not in ("ACTIVE", "PENDING"):
        return f"only an active or pending card can be closed (this one is {card.get('status')})"
    return None


def activate(tool: str) -> Validator:
    def check(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
        card = _card(v, args)
        if card is None:
            return "that card does not exist"
        if card.get("status") != "PENDING":
            return f"only a card waiting to be activated can be activated (this one is {card.get('status')})"
        reason = str(card.get("issue_reason", "new_account"))
        if reason not in ACTIVATION_REASONS[tool]:
            right = next((t for t, rs in ACTIVATION_REASONS.items() if reason in rs), "another tool")
            return f"this card was issued for '{reason}'; use {right}"
        return None

    return check


def order(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    account = _row(v, "accounts", args.get("account_id"))
    if account is None:
        return "that account does not exist"
    if account.get("class") != "checking":
        return "a debit card can only be ordered for a checking account"
    if account.get("status") != "OPEN":
        return "the account is not open"
    if any(
        c.get("account_id") == args.get("account_id") and c.get("status") == "ACTIVE"
        for c in v.database.debit_cards.data.values()
    ):
        return "this account already has an active card: close it first"
    delivery = str(args.get("delivery_option", "")).upper()
    design = str(args.get("card_design", "")).upper()
    if delivery not in DELIVERY_FEES or design not in DESIGN_FEES:
        return "that delivery option or card design is not offered"
    if float(args.get("delivery_fee") or 0) not in DELIVERY_FEES[delivery]:
        return f"{delivery.lower()} delivery does not cost that"
    if float(args.get("design_fee") or 0) not in DESIGN_FEES[design]:
        return f"the {design.lower()} design does not cost that"
    if float(args.get("excess_replacement_fee") or 0) not in EXCESS_REPLACEMENT_FEES:
        return "that is not a published replacement fee"
    return None


# Kinds where "which card?" must be answered before anything else.
CARD_KINDS = frozenset(
    {K.ACTIVATE_CARD, K.CHANGE_OR_RESET_PIN, K.CARD_CONTROLS, K.CLOSE_CARD, K.DISPUTE_TRANSACTION}
)
# Any of these names a card or account well enough for the model to find the card.
_NAMES_A_RECORD = ("account_id", "transaction_id", "credit_card_account_id")


def _owned_cards(db: Any, user_id: str) -> list[Mapping[str, Any]]:
    return sorted(
        (c for c in db.debit_cards.data.values() if c.get("user_id") == user_id and c.get("status") != "CLOSED"),
        key=lambda c: str(c.get("card_id")),
    )


def _describe(db: Any, card: Mapping[str, Any]) -> str:
    account = db.accounts.data.get(str(card.get("account_id")), {})
    where = f", on your {account['level']}" if account.get("level") else ""
    return f"the card ending {card.get('last_4_digits')} ({str(card.get('status')).lower()}{where})"


def _which_card(intent: Intent, db: Any, cards: list[Mapping[str, Any]]) -> RouteOutcome:
    """The list the customer chooses from, written here: never ids, never the model's words."""
    listing = "\n".join(f"- {_describe(db, c)}" for c in cards)
    return RouteOutcome(
        intent_id=intent.id,
        status=OutcomeStatus.NEEDS_INFO,
        questions=(Question(intent_id=intent.id, slot="card_id", text=f"Which card do you mean?\n{listing}"),),
    )


def prepare(intent: Intent, user_id: str, db: Any) -> Intent | RouteOutcome:
    """Settle which card is meant, in code, before a model gets involved."""
    if intent.kind not in CARD_KINDS:
        return intent
    named = (intent.slots.get("card_id") or "").strip()
    cards = _owned_cards(db, user_id)
    if named:
        if any(c.get("card_id") == named for c in cards):
            return intent
        if named.isdigit() and len(named) == 4:  # said the last four digits
            matches = [c for c in cards if c.get("last_4_digits") == named]
            if len(matches) == 1:
                return intent.model_copy(update={"slots": {**intent.slots, "card_id": str(matches[0]["card_id"])}})
            return _which_card(intent, db, matches or cards) if (matches or cards) else intent
        return _which_card(intent, db, cards) if cards else intent
    if any((intent.slots.get(k) or "").strip() for k in _NAMES_A_RECORD):
        return intent
    return _which_card(intent, db, cards) if len(cards) > 1 else intent


VALIDATORS: dict[str, Validator] = {
    "freeze_debit_card_3892": freeze,
    "unfreeze_debit_card_3893": unfreeze,
    "close_debit_card_4721": close,
    "order_debit_card_5739": order,
    **{tool: activate(tool) for tool in ACTIVATION_REASONS},
}

DEBIT_SPECIALIST = SpecialistConfig(
    name="debit",
    kinds=KINDS,
    instructions=INSTRUCTIONS,
    scope=DEBIT,
    validators=VALIDATORS,
    prepare=prepare,
)
