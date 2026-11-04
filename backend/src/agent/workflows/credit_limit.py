"""Credit limit increase: the model reports the amount, code decides.

Source: "How can I request a CLI?", "CLI Eligibility Requirements by Card Tier", "CLI Payment History
and Approval Criteria" and "Processing CLI Approvals and Denials". The order of work is the document's:

  0. The amount is checked against the tier's cap BEFORE anything is submitted. Over the cap is a
     question that names the cap, not a denial.
  1. The request is submitted, which records it.
  2. Every rule is checked, in this order: account age, cooldown, disputes, pending replacement,
     good standing, utilization, payment history. The first one that fails is the denial reason.
  3. Approve (new limit = current limit + the increase) or deny with that reason.

The plan therefore always starts with the submission and ends with an approval or a denial. A
denial is shown to the customer as such before it is recorded. The cooldown counts only requests
that were approved, from their submit date (document 005 over 007: a denied request starts no
cooldown, and neither does the pending record the submission creates).
"""

from datetime import date, datetime, timedelta
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field
from tau2.domains.banking_knowledge.utils import generate_credit_limit_increase_request_id

from agent.specialists.credit_rules import BAD_STANDING, active_orders, money, open_disputes
from agent.workflows.base import Ask, Decision, HandOff, NeedInfo, Plan, Refuse, Reply, Step, WorkflowContext
from agent.workflows.credit_tiers import (
    CLI_COOLDOWN_DAYS,
    CLI_MAX_INCREASE_SHARE,
    CLI_MAX_UTILIZATION,
    CLI_MIN_ACCOUNT_AGE_DAYS,
    CLI_ON_TIME_PAYMENTS,
    tier_of_card_type,
)
from orchestrator.types import IntentKind


class LimitFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    credit_card_account_id: str
    requested_increase_amount: int = Field(
        description=(
            "How many dollars the customer wants the limit raised by. If they name the new limit instead, "
            "subtract the card's current limit. A whole number of dollars above zero."
        )
    )


def _day(value: Any) -> Optional[date]:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value)[:10], fmt).date()
        except ValueError:
            continue
    return None


def cooldown_blocks(db: Any, account_id: str, tier: str, today: date) -> bool:
    """An approved request, submitted inside the tier's cooldown, blocks a new one."""
    for request in db.credit_limit_increase_requests.data.values():
        if request.get("credit_card_account_id") != account_id or str(request.get("status")).lower() != "approved":
            continue
        submitted = _day(request.get("submitted_date") or request.get("submitted_at") or request.get("decision_date"))
        if submitted is None or timedelta(0) <= today - submitted < timedelta(days=CLI_COOLDOWN_DAYS[tier]):
            return True  # a date that cannot be read counts: better to deny than to approve by mistake
    return False


def on_time_streak_ok(db: Any, account_id: str, needed: int) -> bool:
    """The newest `needed` payments of the account are all on time. The bank's history tool counts
    payments, not calendar months, and so do we."""
    rows = [p for p in db.payment_history.data.values() if p.get("credit_card_account_id") == account_id]
    rows.sort(key=lambda p: _day(p.get("payment_date")) or date.min, reverse=True)
    return len(rows) >= needed and all(str(p.get("status")).upper() == "ON_TIME" for p in rows[:needed])


def denial_reason(ctx: WorkflowContext, account: dict[str, Any], tier: str, limit: float) -> Optional[str]:
    """The first policy rule the request fails, as the bank's denial reason code; None if it passes."""
    db, today = ctx.database, ctx.today
    account_id = str(account["account_id"])
    opened = _day(account.get("date_of_account_open"))
    if opened is None or (today - opened).days < CLI_MIN_ACCOUNT_AGE_DAYS[tier]:
        return "insufficient_account_age"
    if cooldown_blocks(db, account_id, tier, today):
        return "cooldown_period_active"
    if open_disputes(db, ctx.user_id):
        return "pending_disputes"
    if active_orders(db, account_id):
        return "pending_replacement_card"
    if str(account.get("account_status") or "").upper() in BAD_STANDING or (money(account.get("past_due_amount")) or 0.0) > 0:
        return "past_due_balance"
    if (money(account.get("current_balance")) or 0.0) / limit >= CLI_MAX_UTILIZATION[tier]:
        return "high_utilization"
    if not on_time_streak_ok(db, account_id, CLI_ON_TIME_PAYMENTS[tier]):
        return "insufficient_payment_history"
    return None


class CreditLimit:
    name = "credit_limit"
    kinds = frozenset({IntentKind.CREDIT_LIMIT_REQUEST})
    summary = "The customer asks for a credit limit increase on a credit card."
    facts = LimitFacts

    def decide(self, ctx: WorkflowContext, facts: LimitFacts) -> Decision:
        db = ctx.database
        account = db.credit_card_accounts.data.get(facts.credit_card_account_id)
        if account is None or account.get("user_id") != ctx.user_id:
            return Refuse(f"{facts.credit_card_account_id} is not one of this customer's credit card accounts")
        if str(account.get("status") or "").upper() == "CLOSED" or str(account.get("account_status") or "").upper() == "CLOSED":
            return Reply("That card account is closed, so its limit cannot be raised.")
        if facts.requested_increase_amount <= 0:
            return Refuse("the increase must be a whole number of dollars above zero")
        tier = tier_of_card_type(str(account.get("card_type")))
        if tier is None:
            return HandOff(f"the card type {account.get('card_type')} has no limit rules on file")
        limit = money(account.get("credit_limit"))
        if limit is None or limit <= 0:
            return HandOff("there is no credit limit on file for this card, so a person has to handle the request")
        cap = limit * CLI_MAX_INCREASE_SHARE[tier]
        if facts.requested_increase_amount > cap:
            return NeedInfo(
                (
                    Ask(
                        "requested_increase_amount",
                        f"This card can be raised by at most ${cap:,.0f} at a time. Would you like to request "
                        f"${cap:,.0f}, or a smaller amount?",
                    ),
                )
            )
        steps: list[Step] = []
        # The bank's tool refuses a second request of the same amount on a card ("a similar request may
        # already exist"), and the first one stays on file. When a request of this amount is already
        # there (task 051: denied for utilization, resubmitted after paying down), it stands as the
        # submission and only the decision is made.
        request_id = generate_credit_limit_increase_request_id(
            facts.credit_card_account_id, ctx.user_id, facts.requested_increase_amount
        )
        if request_id not in db.credit_limit_increase_requests.data:
            steps.append(
                Step(
                    "submit_credit_limit_increase_request_7392",
                    {"credit_card_account_id": facts.credit_card_account_id, "requested_increase_amount": facts.requested_increase_amount},
                )
            )
        reason = denial_reason(ctx, account, tier, limit)
        if reason is None:
            steps.append(
                Step(
                    "approve_credit_limit_increase_5847",
                    {"credit_card_account_id": facts.credit_card_account_id, "new_credit_limit": int(limit) + facts.requested_increase_amount},
                )
            )
        else:
            steps.append(Step("deny_credit_limit_increase_5848", {"credit_card_account_id": facts.credit_card_account_id, "denial_reason": reason}))
        return Plan(tuple(steps))
