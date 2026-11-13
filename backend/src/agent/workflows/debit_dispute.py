"""Debit card dispute: the model reports what the customer said, code fills in the rest.

Source: "Internal: Filing a Debit Card Transaction Dispute" and "Debit Card Provisional
Credit Guidelines". The model never writes the transaction's date or amount, the
liability, the provisional-credit flag or the card action: those come from the database
and the rules below. After the disputes, one card action per card, the most severe one.
"""

from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from agent.cards import describe
from agent.workflows.base import (
    Ask,
    Decision,
    NeedInfo,
    Plan,
    Refuse,
    Step,
    WorkflowContext,
)
from orchestrator.types import IntentKind

Category = Literal[
    "unauthorized_transaction",
    "atm_cash_discrepancy",
    "atm_deposit_not_credited",
    "duplicate_charge",
    "incorrect_amount",
    "goods_services_not_received",
    "recurring_charge_after_cancellation",
    "card_present_fraud",
    "card_not_present_fraud",
]
TransactionType = Literal[
    "pin_purchase",
    "signature_purchase",
    "online_purchase",
    "atm_withdrawal",
    "atm_deposit",
    "recurring_payment",
    "person_to_person",
]
ClosureAnswer = Literal["yes", "no", "not_asked"]
PinCompromised = Literal["yes_shared", "yes_observed", "no", "unknown"]

CARD_ACTION: dict[str, str] = {
    "card_present_fraud": "close_and_reissue",
    "card_not_present_fraud": "close_and_reissue",
    "unauthorized_transaction": "freeze_pending_investigation",
}
SEVERITY = ("keep_active", "freeze_pending_investigation", "close_and_reissue")
# Categories where only part of the charge is in question (an ATM that paid $100 of $200, a charge $100 too high).
# Every other category disputes the whole amount. In every reference answer a partial category is below the charge.
PARTIAL_AMOUNT_CATEGORIES = frozenset({"atm_cash_discrepancy", "incorrect_amount", "atm_deposit_not_credited"})
PROVISIONAL_CATEGORIES = frozenset(
    {
        "unauthorized_transaction",
        "card_present_fraud",
        "card_not_present_fraud",
        "atm_cash_discrepancy",
        "duplicate_charge",
    }
)
MAX_OPEN_DISPUTES = {"Entry": 2, "Mid": 3, "Premium": 4, "Elite": 5}
# The bank's rule is stated against the statement date, which the data does not hold. In
# the reference answers a report 9 days or less after the charge carries the $50 limit and
# one 40 days or more after carries $500; 30 days splits them.
LATE_AFTER_DAYS = 30
MAX_AGE_DAYS = 60


class DisputeFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: str
    category: Category = Field(
        description=(
            "Bank policy: if the customer says they did not make the charge and fraud is suspected, "
            "use card_present_fraud for an in-store or physical-card charge (chip, PIN or swipe) or "
            "card_not_present_fraud for an online or phone charge. Use unauthorized_transaction only "
            "when fraud is NOT suspected."
        )
    )
    discovery_date: str = Field(
        description=(
            "MM/DD/YYYY: when the customer first noticed. If the customer gave no year, use the year of the "
            "charge on the transaction record; never guess a different year and do not ask."
        )
    )
    # What the customer disputes: all of the charge, or only part (an ATM that paid $100 short).
    disputed_amount: float = Field(
        description=(
            "The amount in dispute. When the customer disputes the whole charge, use the amount on "
            "the transaction record even if the customer only gave a rounded figure; do not ask. For "
            "incorrect_amount it is only the overcharge (charged minus what it should have been); for "
            "atm_cash_discrepancy it is only the cash that was not paid out. Never the whole charge for these."
        )
    )
    transaction_type: TransactionType = Field(
        description="How the charge was paid, as the customer or the transaction record says. Do not guess."
    )
    card_in_possession: bool
    pin_compromised: PinCompromised
    contacted_merchant: bool
    police_report_filed: bool
    written_statement_provided: bool
    card_closure: ClosureAnswer = Field(
        description=(
            "Only for a card the bank recommends closing and reissuing after this dispute: 'yes' if the "
            "customer said to close or cancel it, 'no' if they want to keep using it (asking for a new number or a "
            "reissue is NOT consent to close it: report 'no'), 'not_asked' if neither. Report 'not_asked' for "
            "every other card."
        )
    )


class DisputeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    disputes: list[DisputeFacts] = Field(
        description="One entry for every transaction the customer wants to dispute, on every card. Do not leave any out."
    )


def parse_date(text: str) -> Optional[date]:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip()[:10], fmt).date()
        except ValueError:
            continue
    return None


def liability(transaction_date: date, discovery: date) -> float:
    return 500.0 if (discovery - transaction_date).days > LATE_AFTER_DAYS else 50.0


def provisional_credit(facts: DisputeFacts) -> bool:
    if facts.pin_compromised == "yes_shared":
        return False
    if facts.category in PROVISIONAL_CATEGORIES:
        return True
    # Doc 032 says the other categories are "not required but may be offered at discretion". The references offer it
    # only for a cancelled subscription the customer already took to the merchant (task 082); tasks 084 and 086 do
    # not for an incorrect amount or goods not received even after the merchant was contacted. Inferred, one task.
    return facts.category == "recurring_charge_after_cancellation" and facts.contacted_merchant


def tier_of(level: str) -> Optional[str]:
    from agent.workflows.tiers import tier_of_level

    return tier_of_level(level)


class DebitDispute:
    name = "debit_dispute"
    kinds = frozenset({IntentKind.DISPUTE_TRANSACTION})
    summary = "The customer disputes one or more debit card transactions."
    facts = DisputeRequest

    def decide(self, ctx: WorkflowContext, facts: DisputeRequest) -> Decision:
        if not facts.disputes:
            return Refuse("list at least one transaction to dispute")
        db = ctx.database
        steps: list[Step] = []
        card_action: dict[str, str] = {}
        cards_of: dict[str, str] = {}
        seen: set[str] = set()
        new_per_account: dict[str, int] = {}
        for item in facts.disputes:
            if item.transaction_id in seen:
                return Refuse(f"{item.transaction_id} is listed twice")
            seen.add(item.transaction_id)
            txn = db.bank_account_transaction_history.data.get(item.transaction_id)
            account = db.accounts.data.get(str(txn.get("account_id"))) if txn else None
            if txn is None or account is None or account.get("user_id") != ctx.user_id:
                return Refuse(f"{item.transaction_id} is not a transaction on this customer's accounts")
            if account.get("status") != "OPEN":
                return Reply_closed(account)
            happened = parse_date(str(txn.get("date")))
            noticed = parse_date(item.discovery_date)
            if happened is None or noticed is None:
                return Refuse("discovery_date must be written MM/DD/YYYY")
            if noticed < happened or noticed > ctx.today:
                return Refuse(
                    f"discovery_date {item.discovery_date} cannot be before the charge on "
                    f"{happened:%m/%d/%Y} or after today, {ctx.today:%m/%d/%Y}. If the customer gave no "
                    "year, use the year of the charge date; do not ask them."
                )
            amount = abs(float(txn.get("amount", 0)))
            if not 1.0 <= item.disputed_amount <= amount:
                return Refuse(
                    f"the disputed amount must be between $1.00 and the transaction's ${amount:.2f}. "
                    f"If the customer is disputing the whole charge, use ${amount:.2f}; do not ask them."
                )
            if item.category in PARTIAL_AMOUNT_CATEGORIES and item.disputed_amount >= amount:
                return Refuse(
                    f"a {item.category.replace('_', ' ')} dispute covers only part of the ${amount:.2f} charge, not "
                    "all of it: the overcharge, or the cash that was not paid out. Work it out from what the customer "
                    "said (what they were charged and what it should have been, or what they received) or ask them"
                )
            if (ctx.today - happened).days > MAX_AGE_DAYS:
                return Refuse(f"{item.transaction_id} is more than {MAX_AGE_DAYS} days old")
            cards = [
                c
                for c in db.debit_cards.data.values()
                if c.get("account_id") == account["account_id"] and c.get("status") != "CLOSED"
            ]
            if len(cards) != 1:
                return NeedInfo(
                    (Ask("card_id", "Which debit card was used for this transaction?"),)
                )
            card = cards[0]
            tier = tier_of(str(account.get("level")))
            account_id = str(account["account_id"])
            open_now = sum(
                1
                for d in db.debit_card_disputes.data.values()
                if d.get("account_id") == account_id and d.get("status") in (None, "OPEN", "PENDING", "UNDER_REVIEW")
            )
            new_per_account[account_id] = new_per_account.get(account_id, 0) + 1
            # An account level with no known tier is not capped here; the tool still checks.
            if tier is not None and open_now + new_per_account[account_id] > MAX_OPEN_DISPUTES[tier]:
                return Refuse(
                    f"the {account.get('level')} allows {MAX_OPEN_DISPUTES[tier]} open disputes and this would pass that"
                )
            action = CARD_ACTION.get(item.category, "keep_active")
            cards_of[item.transaction_id] = str(card["card_id"])
            card_action[str(card["card_id"])] = max(
                (card_action.get(str(card["card_id"]), "keep_active"), action), key=SEVERITY.index
            )
            steps.append(
                Step(
                    "file_debit_card_transaction_dispute_6281",
                    {
                        "transaction_id": item.transaction_id,
                        "account_id": account_id,
                        "card_id": card["card_id"],
                        "dispute_category": item.category,
                        "transaction_date": happened.strftime("%m/%d/%Y"),
                        "discovery_date": noticed.strftime("%m/%d/%Y"),
                        "disputed_amount": item.disputed_amount,
                        "transaction_type": item.transaction_type,
                        "card_in_possession": item.card_in_possession,
                        "pin_compromised": item.pin_compromised,
                        "contacted_merchant": item.contacted_merchant,
                        "police_report_filed": item.police_report_filed,
                        "written_statement_provided": item.written_statement_provided,
                        "provisional_credit_eligible": provisional_credit(item),
                        "card_action": action,
                        "customer_max_liability_amount": liability(happened, noticed),
                    },
                )
            )
        for card_id, action in card_action.items():
            if action == "close_and_reissue":
                answers = {i.card_closure for i in facts.disputes if cards_of[i.transaction_id] == card_id}
                if "yes" in answers and "no" not in answers:
                    lost = any(not i.card_in_possession for i in facts.disputes if cards_of[i.transaction_id] == card_id)
                    steps.append(Step("close_debit_card_4721", {"card_id": card_id, "reason": "lost" if lost else "fraud_suspected"}))
                elif "no" not in answers:
                    label = describe(db, db.debit_cards.data[card_id])
                    return NeedInfo(
                        (
                            Ask(
                                "card_closure",
                                f"For {label}, the bank recommends closing it and issuing a new card. Do you want "
                                "me to close it, or would you rather keep it?",
                            ),
                        )
                    )
            elif action == "freeze_pending_investigation":
                steps.append(Step("freeze_debit_card_3892", {"card_id": card_id}))
        return Plan(tuple(steps))


def Reply_closed(account: Any) -> Decision:
    from agent.workflows.base import Reply

    return Reply(
        "That account is not open, so a debit card dispute cannot be filed on it."
    )
