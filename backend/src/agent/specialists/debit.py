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

import re
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Optional

from agent.cards import owned_cards, which_card
from agent.debit_lookup import NAMES as LOOKUP_NAMES, debit_lookups
from agent.workflows.debit_dispute import PARTIAL_AMOUNT_CATEGORIES, DebitDispute
from agent.workflows.debit_replacement import DebitReplacement
from agent.specialist import SpecialistConfig, Validator, ValidationContext
from tau2.domains.banking_knowledge.tools import _validate_pin

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
2. LOOK THINGS UP BEFORE YOU ASK. find_debit_transactions searches a card's transactions (name the card by its
   id, last four digits or account, and give any merchant, rough amount, date or transaction id the customer
   said); list_debit_cards and look_up_customer_records show their cards and accounts. For a dispute you may not
   ask the customer anything until you have used one of them; ask only if the search finds nothing or several.
   Never ask for a transaction id, card id or account id that a search can find. If the customer did not say
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
   a date, an amount or a category. The disputed amount must match the transaction. Never ask the
   customer for a transaction id, a card id or an account id: read the account's transactions and
   cards and match what they described by amount, merchant and date. Put every transaction they
   want disputed, on every card, in one form. Do not ask the customer yourself for the dispute's details
   (whether they have the card, PIN, merchant contact, police report, date noticed, statement): fill the form
   with what they said and null for the rest, and the workflow asks only what matters for that kind of dispute.
7. PIN changes: use the secret references that appear in the customer's message.
8. Disputes and replacement cards have a workflow: end with start_debit_dispute or
   start_debit_replacement. Report only what the customer said (which transaction, what
   happened, which card, what delivery and design they want); the fees, liability, card
   action and the order of the steps are worked out for you. If the workflow refuses, it says why:
   fix the form or ask the customer.
9. Fraud alerts: clear one only when the customer confirms the flagged transactions are theirs.
   If the customer says a flagged charge is not theirs, or is unsure, leave the alert in place,
   even if they ask for it to be removed."""

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


# A plain answer changes nothing: only a plan the customer approves does. So an answer that says a card has been or
# will be closed, ordered or issued, with no plan behind it, is a false claim (task 082: "new cards will be issued"
# after the replacement workflow refused).
_CLAIMS_AN_ACTION = re.compile(
    r"\b(?:new\s+)?(?:debit\s+)?cards?\b[^.?!]{0,60}\b(?:will\s+be|has\s+been|have\s+been|is\s+being|are\s+being)\s+"
    r"(?:issued|ordered|sent|shipped|closed|replaced|mailed)\b"
    r"|\b(?:I|we)(?:'ve|\s+have)\s+(?:closed|ordered|issued|sent|replaced|shipped|mailed|filed)\b"
    r"|\b(?:I|we)(?:'ll|\s+will)\s+(?:close|order|issue|send|replace|ship|mail|file)\b",
    re.IGNORECASE,
)


def answer_check(answer: str) -> Optional[str]:
    if _CLAIMS_AN_ACTION.search(answer):
        return (
            "a plain answer closes, orders and issues nothing, so do not say a card was or will be closed, ordered "
            "or issued: say only what is true, or fix the form and propose the plan"
        )
    return None


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
    from agent.workflows.debit_replacement import order_blocker

    user = _row(v, "users", account.get("user_id")) or {}
    on_file = str(args.get("shipping_address") or "") == str(user.get("address") or "")
    if (blocker := order_blocker(v.database, account, user, str(user.get("address") or "") if on_file else "", v.today)) is not None:
        return blocker
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
# A dispute is not here: its workflow finds the card from the transaction.
CARD_KINDS = frozenset(
    {K.ACTIVATE_CARD, K.CHANGE_OR_RESET_PIN, K.CARD_CONTROLS, K.CLOSE_CARD}
)
# Any of these names a card or account well enough for the model to find the card.
_NAMES_A_RECORD = ("account_id", "transaction_id", "credit_card_account_id")


def prepare(intent: Intent, user_id: str, db: Any) -> Intent | RouteOutcome:
    """Settle which card is meant, in code, before a model gets involved."""
    if intent.kind not in CARD_KINDS:
        return intent
    named = (intent.slots.get("card_id") or "").strip()
    cards = owned_cards(db, user_id)
    if named:
        if any(c.get("card_id") == named for c in cards):
            return intent
        if named.isdigit() and len(named) == 4:  # said the last four digits
            matches = [c for c in cards if c.get("last_4_digits") == named]
            if len(matches) == 1:
                return intent.model_copy(update={"slots": {**intent.slots, "card_id": str(matches[0]["card_id"])}})
            return which_card(intent, db, matches or cards) if (matches or cards) else intent
        return which_card(intent, db, cards) if cards else intent
    if any((intent.slots.get(k) or "").strip() for k in _NAMES_A_RECORD):
        return intent
    return which_card(intent, db, cards) if len(cards) > 1 else intent


def _pin(value: Any) -> Optional[str]:
    """The bank's own PIN rule (four digits, nothing sequential or repeated), not a copy of it."""
    problem: Optional[str] = _validate_pin(str(value or ""))
    return problem


def _active_card(v: ValidationContext, args: Mapping[str, Any], what: str) -> Optional[str]:
    card = _card(v, args)
    if card is None:
        return "that card does not exist"
    if card.get("status") != "ACTIVE":
        return f"only an active card can have {what} (this one is {card.get('status')})"
    return None


def change_pin(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    if (problem := _active_card(v, args, "its PIN changed")) is not None:
        return problem
    if (problem := _pin(args.get("new_pin"))) is not None:
        return problem
    if str(args.get("current_pin")) == str(args.get("new_pin")):
        return "the new PIN must differ from the current one"
    return None


def reset_pin(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    if (problem := _active_card(v, args, "its PIN reset")) is not None:
        return problem
    card = _card(v, args)
    assert card is not None
    if str(args.get("last_4_digits")) != str(card.get("last_4_digits")):
        return "the last four digits do not match this card"
    return _pin(args.get("new_pin"))


def recurring_block(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    return _active_card(v, args, "recurring payments blocked or allowed")


def limit_blocker(v: ValidationContext, card: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Doc 040's eligibility rules the bank's tool never checks: the account is OPEN and at least 60 days old, and
    it has had no overdraft fee in the last 30 days (task 089: the Blue card's overdraft 13 days ago)."""
    account = _row(v, "accounts", (card or {}).get("account_id"))
    if account is None:
        return "that card's account does not exist"
    if account.get("status") != "OPEN":
        return "the account is not open"
    opened = _day(account.get("date_opened"))
    if opened is not None and (v.today - opened).days < 60:
        return "the account must be at least 60 days old for a temporary limit increase"
    for t in v.database.bank_account_transaction_history.data.values():
        if t.get("account_id") == account.get("account_id") and "OVERDRAFT" in str(t.get("description", "")).upper():
            when = _day(t.get("date"))
            if when is not None and 0 <= (v.today - when).days <= 30:
                return "an overdraft fee in the last 30 days means the account is not eligible for a temporary limit increase"
    return None


def temporary_limit(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    if (problem := _active_card(v, args, "a limit increase")) is not None:
        return problem
    if args.get("limit_type") not in ("atm", "purchase"):
        return "the limit is either atm or purchase"
    if (problem := limit_blocker(v, _card(v, args))) is not None:
        return problem
    limit = args.get("new_limit")
    if not isinstance(limit, (int, float)) or isinstance(limit, bool) or limit <= 0 or limit != int(limit):
        return "the new limit must be a whole number of dollars above zero"
    return None


def _day(text: Any) -> Optional[date]:
    """A date from MM/DD/YYYY, with or without a time after it."""
    try:
        return datetime.strptime(str(text)[:10], "%m/%d/%Y").date()
    except ValueError:
        return None


def dispute(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    """Ground the dispute in the records. The bank's tool files whatever it is given, so
    nothing else stops a made-up transaction, a card from another account, or an amount
    larger than the charge."""
    txn = _row(v, "bank_account_transaction_history", args.get("transaction_id"))
    if txn is None:
        return "that transaction does not exist"
    if txn.get("account_id") != args.get("account_id"):
        return "that transaction is not on that account"
    card = _card(v, args)
    if card is None:
        return "that card does not exist"
    if card.get("account_id") != args.get("account_id"):
        return "that card is not linked to that account"
    charged = abs(float(txn.get("amount") or 0))
    disputed = float(args.get("disputed_amount") or 0)
    if disputed <= 0:
        return "the disputed amount must be more than zero"
    if disputed > charged:
        return "the disputed amount is more than the transaction"
    if args.get("dispute_category") not in PARTIAL_AMOUNT_CATEGORIES and disputed != charged:
        return "this kind of dispute is for the whole amount of the transaction"
    happened, found = _day(args.get("transaction_date")), _day(args.get("discovery_date"))
    if happened is None or found is None:
        return "dates must be written MM/DD/YYYY"
    if happened != _day(txn.get("date")):
        return "the transaction date is not the date on the transaction"
    if found < happened or found > v.today:
        return "the customer cannot have noticed it before it happened or after today"
    if args.get("provisional_credit_eligible") and not args.get("written_statement_provided"):
        return "provisional credit needs the customer's written statement first"
    return None


def clear_alert(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    card = _card(v, args)
    if card is None:
        return "that card does not exist"
    reason = args.get("reason")
    if reason == "velocity_clear":
        return None if card.get("velocity_blocked") else "this card has no velocity block to clear"
    if reason != "customer_verified":
        return "an alert is cleared as customer_verified or velocity_clear"
    if not card.get("fraud_alert_active"):
        return "this card has no active fraud alert"
    if card.get("alert_source") == "bank_initiated":
        return "the bank raised this alert itself; only the security team can clear it, so hand over to a person"
    return None


VALIDATORS: dict[str, Validator] = {
    "file_debit_card_transaction_dispute_6281": dispute,
    "change_debit_card_pin_6285": change_pin,
    "reset_debit_card_pin_6284": reset_pin,
    "set_debit_card_recurring_block_7382": recurring_block,
    "request_temporary_debit_card_limit_increase_8374": temporary_limit,
    "clear_debit_card_fraud_alert_4892": clear_alert,
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
    workflows=(DebitDispute(), DebitReplacement()),
    answer_check=answer_check,
    lookups=debit_lookups,
    lookup_first=frozenset({K.DISPUTE_TRANSACTION}),
    lookup_names=LOOKUP_NAMES,
)
