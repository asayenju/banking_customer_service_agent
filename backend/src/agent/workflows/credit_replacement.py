"""Replace a credit card.

Source: "How to Order a Replacement Credit Card (Internal)", the shipping fee and the "why was your
replacement rejected" documents. The model reports the customer's choices (which card, why, whether
they want expedited delivery and accept its fee, where to ship). Code works out the rest: the fee for
the card's tier, the pending-order and 60-day limits the bank's tool never checks, and the address on
file when the customer gave none.

Ordering the replacement cancels the old card at once, so this is only offered for a card that is
open and has no replacement already on its way.
"""

from datetime import date, datetime, timedelta
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

from agent.specialists.credit_rules import active_orders
from agent.workflows.base import Ask, Decision, HandOff, NeedInfo, Plan, Refuse, Reply, Step, WorkflowContext
from agent.workflows.credit_tiers import EXPEDITED_FEE, REPLACEMENT_LIMIT_60_DAYS, tier_of_card_type
from orchestrator.types import IntentKind

Reason = Literal["fraud_suspected", "lost", "stolen", "damaged", "expired", "other"]
WINDOW_DAYS = 60


class NewCreditCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    credit_card_account_id: str
    reason: Reason
    # True if the customer wants delivery in 2 to 3 days instead of 7 to 10.
    expedited: bool
    # True only if the customer has agreed to pay the expedited fee when one applies.
    accepts_expedite_fee: bool
    # The address the customer gave, or null for the one on file.
    shipping_address: Optional[str]


class CreditReplacementFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    replacements: list[NewCreditCard]


def _day(text: Any) -> Optional[date]:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(text)[:10], fmt).date()
        except ValueError:
            continue
    return None


class CreditReplacement:
    name = "credit_replacement"
    kinds = frozenset({IntentKind.ORDER_REPLACEMENT_CARD})
    summary = (
        "The customer wants a replacement credit card (lost, stolen, fraud, damaged or expired). "
        "List each card to replace. The fee and the limits are worked out for the account."
    )
    facts = CreditReplacementFacts

    def decide(self, ctx: WorkflowContext, facts: CreditReplacementFacts) -> Decision:
        if not facts.replacements:
            return Refuse("list at least one card to replace")
        db = ctx.database
        user = db.users.data.get(ctx.user_id, {})
        steps: list[Step] = []
        seen: set[str] = set()
        for item in facts.replacements:
            account = db.credit_card_accounts.data.get(item.credit_card_account_id)
            if account is None or account.get("user_id") != ctx.user_id:
                return Refuse(f"{item.credit_card_account_id} is not one of this customer's credit card accounts")
            if item.credit_card_account_id in seen:
                return Refuse(f"{item.credit_card_account_id} is listed twice")
            seen.add(item.credit_card_account_id)
            if str(account.get("status") or "").upper() == "CLOSED":
                return Reply("That card account is already closed, so a replacement cannot be ordered for it.")
            tier = tier_of_card_type(str(account.get("card_type")))
            if tier is None:
                return HandOff(f"the card type {account.get('card_type')} has no replacement rules on file")
            if active_orders(db, item.credit_card_account_id):
                return Reply(
                    "A replacement for that card is already on its way. Another can be ordered once it has "
                    "been delivered or cancelled."
                )
            recent = [
                o
                for o in db.credit_card_orders.data.values()
                if o.get("credit_card_account_id") == item.credit_card_account_id
                and (placed := _day(o.get("order_date"))) is not None
                and timedelta(0) <= ctx.today - placed <= timedelta(days=WINDOW_DAYS)
            ]
            if len(recent) >= REPLACEMENT_LIMIT_60_DAYS[tier]:
                return Reply(
                    "This card has used all its replacements for the last 60 days. A person can review a case "
                    "that needs one more."
                )
            fee = EXPEDITED_FEE[tier] if item.expedited else 0.0
            if fee > 0 and not item.accepts_expedite_fee:
                return NeedInfo(
                    (Ask("accepts_expedite_fee", f"Expedited delivery for this card costs ${fee:.0f}. Would you like to go ahead with it?"),)
                )
            address = item.shipping_address or str(user.get("address") or "")
            if not address:
                return NeedInfo((Ask("shipping_address", "Where should the new card be sent?"),))
            steps.append(
                Step(
                    "order_replacement_credit_card_7291",
                    {
                        "credit_card_account_id": item.credit_card_account_id,
                        "shipping_address": address,
                        "reason": item.reason,
                        "expedited_shipping": item.expedited,
                    },
                )
            )
        return Plan(tuple(steps))
