"""Closing a credit card, with the bank's retention steps.

Source: "Internal: Credit Card Retention Protocol" (``..._account_logistics_003``) and the closure
rules in the account logistics notes. The conversation is a short one, and it lives in the facts the
model reports from the customer's answers, not in this class:

  1. Eligibility, from fresh records: no dispute in progress, no replacement on its way, the account
     at least 60 days old, and a zero balance. A balance can be paid from the customer's checking
     account first (its own plan); nothing else is offered until it is paid.
  2. A closure reason already on file for this card from the last year means retention was tried:
     the card is closed with no offers.
  3. Otherwise the reason is asked. Then at most two offers, each asked as a question:
       - an annual-fee waiver for a year, only when the reason is the annual fee and the customer is
         two years or more old (a customer under two years would be offered a downgrade, which this
         bank cannot apply, so that step is skipped);
       - the tier's statement credit (document, step 5), on top of a line of advice for the reason.
  4. The plan is shown only when the outcome is known: the reason is logged together with the waiver,
     the credit, a decision to keep the card, or the close. A close is never in a plan the customer
     has not seen, and never follows a plan in which an offer was accepted.

Nothing here offers a downgrade or an upgrade, or applies for a card for the customer.
"""

from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

from agent.specialists.credit_rules import (
    MIN_AGE_FOR_CLOSING_DAYS,
    account_disputes,
    active_orders,
    money,
    years_after,
)
from agent.workflows.base import Ask, Decision, HandOff, NeedInfo, Plan, Refuse, Reply, Step, WorkflowContext
from agent.workflows.credit_tiers import RETENTION_CREDIT, tier_of_card_type
from orchestrator.types import IntentKind

Reason = Literal["annual_fee", "not_using_card", "found_better_card", "unhappy_with_rewards", "simplifying_finances", "negative_experience", "other"]
YesNo = Literal["yes", "no"]
WAIVER_MIN_YEARS = 2
RETENTION_LOOKBACK_DAYS = 365

ADVICE: dict[str, str] = {
    "not_using_card": "Setting up a small recurring payment, such as a subscription, on the card keeps it active and builds your credit history.",
    "found_better_card": "If another card caught your eye, we may have one with similar or better benefits that you could apply for instead of closing this one.",
    "unhappy_with_rewards": "Check which bonus categories you can enroll in; they can add up quickly for the way you spend.",
    "negative_experience": "I'm sorry the experience was a poor one, and I will pass your feedback on.",
}
AFTER_CLOSING = (
    "You have 45 days from the closing request to redeem any rewards, after which they are forfeited. "
    "If the annual fee posted recently, a full refund may be available when a card closes within 37 days of it."
)


class ClosureFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    credit_card_account_id: str
    # Why the customer wants to close it, or null if they have not said.
    reason: Optional[Reason]
    # The checking account the customer chose to pay the card's balance from, or null.
    pay_from_checking_account_id: Optional[str]
    # Their answer to the annual-fee waiver offer, or null if it was not offered or not answered.
    waiver_answer: Optional[YesNo]
    # Their answer to the statement credit offer, or null if it was not offered or not answered.
    credit_answer: Optional[YesNo]
    # True if they have decided to keep the card (for example, to apply for another card instead).
    keeps_card: bool


def _day(value: Any) -> Optional[date]:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value)[:10], fmt).date()
        except ValueError:
            continue
    return None


def closure_records(db: Any, account_id: str) -> list[dict[str, Any]]:
    return [r for r in db.credit_card_closure_reasons.data.values() if r.get("credit_card_account_id") == account_id]


def retention_tried_recently(db: Any, account_id: str, today: date) -> bool:
    """A reason logged for this card in the last year, or one whose date cannot be read."""
    for record in closure_records(db, account_id):
        logged = _day(record.get("logged_at"))
        if logged is None or 0 <= (today - logged).days <= RETENTION_LOOKBACK_DAYS:
            return True
    return False


class CreditClosure:
    name = "credit_closure"
    kinds = frozenset({IntentKind.CLOSE_CARD})
    summary = (
        "The customer wants to close a credit card. Report the card, the reason if given, and the answers "
        "to the offers as they come; what to offer and when is worked out."
    )
    facts = ClosureFacts

    def decide(self, ctx: WorkflowContext, facts: ClosureFacts) -> Decision:
        db = ctx.database
        account = db.credit_card_accounts.data.get(facts.credit_card_account_id)
        if account is None or account.get("user_id") != ctx.user_id:
            return Refuse(f"{facts.credit_card_account_id} is not one of this customer's credit card accounts")
        if str(account.get("status") or "").upper() == "CLOSED" or account.get("closed_date"):
            return Reply("That card account is already closed.")
        tier = tier_of_card_type(str(account.get("card_type")))
        if tier is None:
            return HandOff(f"the card type {account.get('card_type')} has no closing rules on file")
        account_id = facts.credit_card_account_id
        if account_disputes(db, account):
            return Reply("There is a dispute in progress on this card. It can be closed once the dispute is resolved.")
        if active_orders(db, account_id):
            return Reply("A replacement card for this account is on its way. The account can be closed once it has been received or cancelled.")
        opened = _day(account.get("date_of_account_open"))
        if opened is None or (ctx.today - opened).days < MIN_AGE_FOR_CLOSING_DAYS:
            return Reply("An account has to be open at least 60 days before it can be closed.")
        balance = money(account.get("current_balance"))
        if balance is None:
            return HandOff("the balance on this card is not on file")
        if balance > 0:
            if not facts.pay_from_checking_account_id:
                return NeedInfo(
                    (
                        Ask(
                            "pay_from_checking_account_id",
                            f"This card has a balance of ${balance:,.2f}, which has to be paid before it can be closed. "
                            "I can pay it from one of your checking accounts. Which one would you like to use?",
                        ),
                    )
                )
            return Plan(
                (
                    Step(
                        "pay_credit_card_from_checking_9182",
                        {"checking_account_id": facts.pay_from_checking_account_id, "credit_card_account_id": account_id, "amount": balance},
                    ),
                )
            )
        if retention_tried_recently(db, account_id, ctx.today):
            return Plan((Step("close_credit_card_account_7834", {"credit_card_account_id": account_id}),))
        if facts.reason is None:
            return NeedInfo((Ask("closure_reason", "I can help with that. May I ask why you would like to close this card?"),))
        log: list[Step] = []
        if not closure_records(db, account_id):  # the bank's tool cannot log a second reason for a card
            log = [Step("log_credit_card_closure_reason_4521", {"credit_card_account_id": account_id, "closure_reason": facts.reason})]
        if facts.keeps_card:
            return Plan(tuple(log)) if log else Reply("Understood, the card stays open as it is.")
        if facts.reason == "annual_fee" and years_after(opened, WAIVER_MIN_YEARS) <= ctx.today:
            if facts.waiver_answer is None:
                return NeedInfo(
                    (
                        Ask(
                            "waiver_answer",
                            "As a thank you for being a customer for more than two years, I can waive this card's annual fee "
                            "for a year. Would you like that instead of closing the card?",
                        ),
                    )
                )
            if facts.waiver_answer == "yes":
                waiver = Step(
                    "apply_credit_card_account_flag_6147",
                    {
                        "credit_card_account_id": account_id,
                        "flag_type": "annual_fee_waived",
                        "expiration_date": years_after(ctx.today, 1).strftime("%m/%d/%Y"),
                        "reason": "loyalty_benefit",
                    },
                )
                return Plan((*log, waiver))
        amount = RETENTION_CREDIT[tier]
        if facts.credit_answer is None:
            advice = ADVICE.get(facts.reason, "")
            return NeedInfo(
                (
                    Ask(
                        "credit_answer",
                        f"{advice} Before you decide, I can add a ${amount:,.0f} statement credit to your account if you keep the card. "
                        "Would you like that?".strip(),
                    ),
                )
            )
        if facts.credit_answer == "yes":
            credit = Step("apply_statement_credit_8472", {"credit_card_account_id": account_id, "amount": amount, "reason": "retention_offer"})
            return Plan((*log, credit))
        return Plan((*log, Step("close_credit_card_account_7834", {"credit_card_account_id": account_id})))

    def after(self, ctx: WorkflowContext, steps: Sequence[Step]) -> str:
        """What to tell the customer once the plan has run."""
        tools = [s.tool for s in steps]
        if tools == ["pay_credit_card_from_checking_9182"]:
            return "The balance is paid. Why would you like to close this card?"
        if "close_credit_card_account_7834" in tools:
            return AFTER_CLOSING
        return ""
