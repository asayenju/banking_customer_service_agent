"""The note a person receives when a customer is handed over, written by code.

A model never writes it: it could repeat what a customer claimed (a payment that was never made, a
card never owned) or carry details that should not travel. The note is built from what the records
show was done (the audit of completed changes) and from the reasons the workflows gave for what they
declined, in fixed sentences. It names the customer and the card types; it never contains an id,
a contact detail, a PIN or a card number.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from banking_tools.types import AuditEntry


def _words(value: Any) -> str:
    return str(value).replace("_", " ")


def _card(db: Any, arguments: Mapping[str, Any]) -> str:
    account = db.credit_card_accounts.data.get(str(arguments.get("credit_card_account_id")))
    return str(account["card_type"]) if account else "a card"


def _dollars(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "an amount"


def describe(db: Any, entry: AuditEntry) -> str:
    """One sentence for a completed change."""
    a, card = entry.arguments, _card(db, entry.arguments)
    tool = entry.tool
    if tool == "close_credit_card_account_7834":
        return f"{card} was closed."
    if tool == "log_credit_card_closure_reason_4521":
        return f"The reason for wanting to close {card} was recorded ({_words(a.get('closure_reason'))})."
    if tool == "apply_statement_credit_8472":
        return f"A {_dollars(a.get('amount'))} statement credit ({_words(a.get('reason'))}) was applied to {card}."
    if tool == "apply_credit_card_account_flag_6147":
        return f"{_words(a.get('flag_type')).capitalize()} was applied to {card} until {a.get('expiration_date')}."
    if tool == "pay_credit_card_from_checking_9182":
        return f"{_dollars(a.get('amount'))} was paid toward {card} from checking."
    if tool == "file_credit_card_transaction_dispute_4829":
        txn = db.credit_card_transaction_history.data.get(str(a.get("transaction_id")), {})
        return f"A dispute was filed for a {txn.get('merchant_name', 'card')} charge ({_words(a.get('dispute_reason'))})."
    if tool == "order_replacement_credit_card_7291":
        return f"A replacement was ordered for {card} ({_words(a.get('reason'))})."
    if tool == "submit_credit_limit_increase_request_7392":
        return f"A limit increase of {_dollars(a.get('requested_increase_amount'))} was requested for {card}."
    if tool == "approve_credit_limit_increase_5847":
        return f"The limit of {card} was raised to {_dollars(a.get('new_credit_limit'))}."
    if tool == "deny_credit_limit_increase_5848":
        return f"A limit increase for {card} was denied ({_words(a.get('denial_reason'))})."
    if tool == "apply_checking_account_credit_5829":
        return f"A {_dollars(a.get('amount'))} ATM fee credit ({_words(a.get('credit_type'))}) was applied to a checking account."
    if tool == "apply_savings_account_credit_6831":
        return f"A {_dollars(a.get('amount'))} interest correction was applied to a savings account."
    if tool == "submit_interest_discrepancy_report_7294":
        return "An interest discrepancy report was submitted for review (a report only, not a credit)."
    if tool == "update_transaction_rewards_3847":
        txn = db.credit_card_transaction_history.data.get(str(a.get("transaction_id")), {})
        return f"The rewards on a {txn.get('merchant_name', 'card')} purchase were corrected to {a.get('new_rewards_earned')}."
    return "Another change was made to the customer's accounts."


def done_events(db: Any, audit: Sequence[AuditEntry]) -> list[str]:
    return [describe(db, e) for e in audit if e.ok]


def handoff_summary(customer_name: str, done: Sequence[str], declined: Sequence[str], why: str) -> str:
    """``done`` from ``done_events``; ``declined`` the workflows' own explanations of what they would not do;
    ``why`` a short fixed phrase for the hand-over (for example "the customer asked for a person")."""
    parts = [f"Customer {customer_name}: {why}."]
    if done:
        parts.append("Done in this conversation: " + " ".join(done))
    if declined:
        parts.append("Not done, with the reason given to the customer: " + " ".join(declined))
    if not done and not declined:
        parts.append("Nothing was changed in this conversation.")
    return " ".join(parts)
