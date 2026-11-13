"""Replace a debit card: close the old one if it is still open, then order the new one.

Source: "Internal: Debit Card Replacement Options by Account Tier". The model reports the
customer's choices (which card, why, delivery speed, design, where to ship). Code decides
the fees (tier tables), the limit on replacements in 12 months and the waiting period,
and refuses a speed the tier does not offer.
"""

from collections.abc import Mapping
from datetime import date, datetime, timedelta
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from agent.workflows.base import (
    Ask,
    Decision,
    HandOff,
    NeedInfo,
    Plan,
    Refuse,
    Reply,
    Step,
    WorkflowContext,
)
from agent.workflows.tiers import tier_of_level
from orchestrator.types import IntentKind

Delivery = Literal["STANDARD", "EXPEDITED", "RUSH", "FASTEST_FREE"]
# Slowest to fastest. "FASTEST_FREE" means the fastest of these that costs nothing on the tier.
SPEED = ("STANDARD", "EXPEDITED", "RUSH")
Design = Literal["CLASSIC", "PREMIUM", "CUSTOM", "BEST_FREE"]
# Plainest to fanciest. "BEST_FREE" means the fanciest of these that costs nothing on the tier.
LOOKS = ("CLASSIC", "PREMIUM", "CUSTOM")
ExcessFee = Literal["yes", "no", "not_asked"]
Reason = Literal["lost", "stolen", "fraud_suspected", "damaged"]

DELIVERY_FEE: dict[str, dict[str, float]] = {
    "Entry": {"STANDARD": 0.0},
    "Mid": {"STANDARD": 0.0, "EXPEDITED": 15.0},
    "Premium": {"STANDARD": 0.0, "EXPEDITED": 0.0, "RUSH": 35.0},
    "Elite": {"STANDARD": 0.0, "EXPEDITED": 0.0, "RUSH": 0.0},
}
DESIGN_FEE: dict[str, dict[str, float]] = {
    "Entry": {"CLASSIC": 0.0, "PREMIUM": 10.0, "CUSTOM": 25.0},
    "Mid": {"CLASSIC": 0.0, "PREMIUM": 10.0, "CUSTOM": 25.0},
    "Premium": {"CLASSIC": 0.0, "PREMIUM": 0.0, "CUSTOM": 15.0},
    "Elite": {"CLASSIC": 0.0, "PREMIUM": 0.0, "CUSTOM": 0.0},
}
LIMIT = {"Entry": 2, "Mid": 3, "Premium": 5}
EXCESS_FEE = {"Entry": 25.0, "Mid": 15.0}
COUNTS = frozenset({"lost", "stolen", "fraud", "damaged"})
WINDOW_DAYS = 365
# Entry tier: doc 029 says a new card waits this long after closure. The bank's tool never checks it, and the gold
# of task 091 orders an Entry card the same day, so the debit workflow enforces it only for an account past its limit (docs/fraud-policy-decisions.md).
WAIT_HOURS = 48


class Closure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    card_id: str
    # Why the old card is being closed.
    reason: Reason
    # True only if the customer agreed that a refund still pending on the card is credited to the checking account.
    accepts_refund_to_checking: bool


class NewCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    account_id: str
    delivery_option: Delivery
    card_design: Design = Field(
        description="The design the customer asked for. If they did not mention a design, CLASSIC. BEST_FREE only if they ask for the best free one."
    )
    # The address the customer gave, or null for the one on file.
    shipping_address: Optional[str]
    # The customer's answer to the excess replacement fee: "yes" they pay it, "no" they decline it, or
    # "not_asked" if they have not been asked (or it does not apply).
    accepts_excess_fee: ExcessFee


class ReplacementFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Cards the customer wants closed (empty if they are closed already).
    closures: list[Closure]
    # New cards to order, one per checking account.
    new_cards: list[NewCard]


def _issued(card: dict[str, object]) -> Optional[date]:
    text = str(card.get("issue_date") or card.get("date_issued") or "")[:10]
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def pending_rows(db: Any, account_id: str) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """The account's pending purchases and its pending refunds. The bank's close tool checks neither (doc 025)."""
    rows = [
        t
        for t in db.bank_account_transaction_history.data.values()
        if t.get("account_id") == account_id and str(t.get("status")).lower() == "pending"
    ]
    refunds = [t for t in rows if t.get("type") == "refund"]
    return [t for t in rows if t.get("type") != "refund"], refunds


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


class DebitReplacement:
    name = "debit_replacement"
    kinds = frozenset({IntentKind.ORDER_REPLACEMENT_CARD, IntentKind.CLOSE_CARD})
    summary = (
        "The customer wants debit cards closed, replaced, or both (lost, stolen, damaged or fraud). "
        "List each card to close and each new card to order. If the customer asks for the fastest "
        "free delivery, answer FASTEST_FREE; if they ask for the best free design, answer BEST_FREE. "
        "The right choice is worked out for the account."
    )
    facts = ReplacementFacts

    def decide(self, ctx: WorkflowContext, facts: ReplacementFacts) -> Decision:
        if not facts.closures and not facts.new_cards:
            return Refuse("list at least one card to close or one new card to order")
        db = ctx.database
        steps: list[Step] = []
        notes: list[str] = []
        needs: list[Ask] = []
        closing: set[str] = set()
        closing_accounts: set[str] = set()
        held_accounts: set[str] = set()
        for item in facts.closures:
            card = db.debit_cards.data.get(item.card_id)
            account = db.accounts.data.get(str(card.get("account_id"))) if card else None
            if card is None or account is None or account.get("user_id") != ctx.user_id:
                return Refuse(f"{item.card_id} is not one of this customer's debit cards")
            if item.card_id in closing:
                return Refuse(f"{item.card_id} is listed twice")
            closing.add(item.card_id)
            if card.get("status") == "CLOSED":
                return Refuse(
                    f"{item.card_id} is already closed; leave it out of closures and do not tell the customer "
                    "that anything will be done for it"
                )
            account_id = str(card["account_id"])
            label = f"your {account.get('level') or 'account'} card"
            purchases, refunds = pending_rows(db, account_id)
            if purchases:
                # A card with purchases still pending is held, not closed (doc 025): frozen if it is not yet.
                if card.get("status") == "ACTIVE":
                    steps.append(Step("freeze_debit_card_3892", {"card_id": item.card_id}))
                held_accounts.add(account_id)
                notes.append(
                    f"{_cap(label)} stays frozen, not closed: it has purchases still pending, and a card with "
                    "pending purchases cannot be closed until they settle."
                )
                continue
            if refunds and not item.accepts_refund_to_checking:
                needs.append(
                    Ask(
                        "accepts_refund_to_checking",
                        f"{_cap(label)} has a refund still pending. It cannot be closed until the refund "
                        "processes, unless you agree that it is credited to your checking account instead. "
                        "Do you agree?",
                    )
                )
                continue
            closing_accounts.add(account_id)
            if card.get("status") == "FROZEN":
                steps.append(Step("unfreeze_debit_card_3893", {"card_id": item.card_id}))
            steps.append(Step("close_debit_card_4721", {"card_id": item.card_id, "reason": item.reason}))
        user = db.users.data.get(ctx.user_id, {})
        ordering: set[str] = set()
        for new in facts.new_cards:
            account = db.accounts.data.get(new.account_id)
            if account is None or account.get("user_id") != ctx.user_id:
                return Refuse(f"{new.account_id} is not one of this customer's accounts")
            if new.account_id in ordering:
                return Refuse(f"{new.account_id} is listed twice")
            ordering.add(new.account_id)
            if new.account_id in held_accounts:
                notes.append(f"No new card is ordered for your {account.get('level')} while its card is held.")
                continue
            tier = tier_of_level(str(account.get("level")))
            if tier is None:
                return HandOff(f"the account level {account.get('level')} has no replacement rules on file")
            speed: str = new.delivery_option
            if speed == "FASTEST_FREE":
                speed = [o for o in SPEED if DELIVERY_FEE[tier].get(o) == 0.0][-1]
            design: str = new.card_design
            if design == "BEST_FREE":
                design = [d for d in LOOKS if DESIGN_FEE[tier][d] == 0.0][-1]
            if speed not in DELIVERY_FEE[tier]:
                offered = ", ".join(DELIVERY_FEE[tier])
                return Refuse(f"the {account.get('level')} offers only these delivery options: {offered}")
            address = new.shipping_address or str(user.get("address") or "")
            if not address:
                return NeedInfo((Ask("shipping_address", "Where should the new card be sent?"),))
            excess: Optional[float] = None
            recent = [
                c
                for c in db.debit_cards.data.values()
                if c.get("account_id") == new.account_id
                and c.get("issue_reason") in COUNTS
                and (issued := _issued(c)) is not None
                and 0 <= (ctx.today - issued).days <= WINDOW_DAYS
            ]
            if tier in LIMIT and len(recent) >= LIMIT[tier]:
                if tier not in EXCESS_FEE:
                    return Reply(
                        "This account has used all its replacement cards for the last 12 months. "
                        "A new one can be ordered once the oldest has aged out."
                    )
                excess = EXCESS_FEE[tier]
                if new.accepts_excess_fee == "no":
                    notes.append(
                        f"No new card is ordered for your {account.get('level')}: it is past its replacement "
                        f"limit and you chose not to pay the ${excess:.0f} excess replacement fee."
                    )
                    continue
                if tier == "Entry" and new.account_id in closing_accounts:
                    notes.append(
                        f"A new card for your {account.get('level')} cannot be ordered today: this tier waits "
                        f"{WAIT_HOURS} hours after a card is closed. It can be ordered from "
                        f"{ctx.today + timedelta(hours=WAIT_HOURS):%m/%d/%Y}."
                    )
                    continue
                if new.accepts_excess_fee != "yes":
                    return NeedInfo(
                        (
                            Ask(
                                "accepts_excess_fee",
                                f"This account is past its replacement limit, so a ${excess:.0f} "
                                "excess replacement fee would be charged to your checking account. "
                                "Would you like to go ahead?",
                            ),
                        )
                    )
            order: dict[str, object] = {
                "account_id": new.account_id,
                "delivery_option": speed,
                "delivery_fee": DELIVERY_FEE[tier][speed],
                "card_design": design,
                "design_fee": DESIGN_FEE[tier][design],
                "shipping_address": address,
            }
            if excess is not None:
                order["excess_replacement_fee"] = excess
            steps.append(Step("order_debit_card_5739", order))
        if needs:
            return NeedInfo(tuple(needs))
        if not steps:
            return Reply(" ".join(notes))
        if notes:
            first = steps[0]
            extra = "Not included: " + " ".join(notes)
            steps[0] = Step(first.tool, first.args, f"{first.note}. {extra}" if first.note else extra)
        return Plan(tuple(steps))
