"""Credit card dispute: the model reports what the customer said, code fills in the rest.

Source: "Credit card disputes: provisional credit" (``doc_credit_cards_credit_cards_(general)_015``) and
the dispute tool's own list of reasons. The model never writes the transaction's date, the contact
details, the provisional-credit flag or whether a charge is on a card the customer owns: those come
from the records and the rules below. The bank's tool checks none of that.

Provisional credit is given only when every condition of the document holds: the account is at
least 60 days old; the reason is an unauthorized or duplicate charge, or goods not received on a
purchase more than 30 days old; the amount is at least $25 and within the card's cap; the customer
has not filed more than 2 disputes in the last 12 months (counting the ones earlier in this
request, in the order the customer gave them); and, for anything but fraud, the customer tried the
merchant first. This rule reproduces every reference answer that files credit disputes.
"""

from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from agent.workflows.base import Ask, Decision, HandOff, NeedInfo, Plan, Refuse, Reply, Step, WorkflowContext
from agent.workflows.credit_tiers import PROVISIONAL_CREDIT_CAP
from orchestrator.types import IntentKind

Reason = Literal[
    "unauthorized_fraudulent_charge",
    "duplicate_charge",
    "incorrect_amount",
    "goods_services_not_received",
    "goods_services_not_as_described",
    "canceled_subscription_still_charging",
    "refund_never_processed",
]
PROVISIONAL_REASONS = frozenset({"unauthorized_fraudulent_charge", "duplicate_charge", "goods_services_not_received"})
MIN_ACCOUNT_AGE_DAYS = 60
NOT_RECEIVED_AFTER_DAYS = 30
MIN_PROVISIONAL_AMOUNT = 25.0
MAX_DISPUTES_IN_A_YEAR = 2


class CreditDisputeFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: str
    reason: Reason
    # MM/DD/YYYY: when the customer first noticed the problem.
    noticed_date: str
    contacted_merchant: bool
    # The last four digits of the card the customer says were used, or null if they did not say.
    card_last_4_digits: Optional[str]
    # Null when the customer disputes the whole charge (use null; do not ask). Otherwise the
    # amount they say they are owed, which must be less than the charge.
    partial_refund_amount: Optional[float] = Field(default=None)
    # True only if the customer asks for the card to be cancelled and reissued.
    cancel_and_reissue: bool


class CreditDisputeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    disputes: list[CreditDisputeFacts]


def parse_date(text: Any) -> Optional[date]:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(text).strip()[:10], fmt).date()
        except ValueError:
            continue
    return None


def money(value: Any) -> float:
    try:
        return float(str(value).replace("$", "").replace(",", ""))
    except ValueError:
        return 0.0


def account_of(db: Any, txn: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The credit account a transaction is on: by its id, or by the customer's one card of that type."""
    if txn.get("credit_card_account_id"):
        found = db.credit_card_accounts.data.get(str(txn["credit_card_account_id"]))
        return found if isinstance(found, dict) else None
    same = [
        a
        for a in db.credit_card_accounts.data.values()
        if a.get("user_id") == txn.get("user_id") and a.get("card_type") == txn.get("credit_card_type")
    ]
    return same[0] if len(same) == 1 else None


def disputes_in_last_year(db: Any, user_id: str, today: date) -> int:
    count = 0
    for d in db.transaction_disputes.data.values():
        if d.get("user_id") != user_id:
            continue
        filed = parse_date(d.get("submitted_at"))
        if filed is None or 0 <= (today - filed).days <= 365:
            count += 1  # a date that cannot be read counts: better to withhold a credit than to give it by mistake
    return count


def provisional_credit(
    account: dict[str, Any], facts: CreditDisputeFacts, amount: float, purchased: date, today: date, earlier: int
) -> bool:
    opened = parse_date(account.get("date_of_account_open"))
    cap = PROVISIONAL_CREDIT_CAP.get(str(account.get("card_type")))
    if opened is None or cap is None or (today - opened).days < MIN_ACCOUNT_AGE_DAYS:
        return False
    if facts.reason not in PROVISIONAL_REASONS:
        return False
    if facts.reason == "goods_services_not_received" and (today - purchased).days <= NOT_RECEIVED_AFTER_DAYS:
        return False
    if not MIN_PROVISIONAL_AMOUNT <= amount <= cap:
        return False
    if earlier > MAX_DISPUTES_IN_A_YEAR:
        return False
    return facts.reason == "unauthorized_fraudulent_charge" or facts.contacted_merchant


class CreditDispute:
    name = "credit_dispute"
    kinds = frozenset({IntentKind.DISPUTE_TRANSACTION})
    summary = "The customer disputes one or more credit card transactions."
    facts = CreditDisputeRequest

    def decide(self, ctx: WorkflowContext, facts: CreditDisputeRequest) -> Decision:
        if not facts.disputes:
            return Refuse("list at least one transaction to dispute")
        db = ctx.database
        user = db.users.data.get(ctx.user_id) or {}
        contact = {
            "full_name": user.get("name"),
            "phone": user.get("phone_number"),
            "email": user.get("email"),
            "address": user.get("address"),
        }
        if not all(contact.values()):
            return HandOff("the contact details the bank needs for a dispute are not on file")
        already = {d.get("transaction_id") for d in db.transaction_disputes.data.values() if d.get("user_id") == ctx.user_id}
        earlier = disputes_in_last_year(db, ctx.user_id, ctx.today)
        steps: list[Step] = []
        seen: set[str] = set()
        for item in facts.disputes:
            if item.transaction_id in seen:
                return Refuse(f"{item.transaction_id} is listed twice")
            seen.add(item.transaction_id)
            txn = db.credit_card_transaction_history.data.get(item.transaction_id)
            if txn is None or txn.get("user_id") != ctx.user_id:
                return Refuse(f"{item.transaction_id} is not a credit card transaction of this customer")
            if item.transaction_id in already:
                return Reply("A dispute is already open for one of those charges, so it cannot be filed again.")
            account = account_of(db, txn)
            if account is None:
                return HandOff("the card account for one of those charges cannot be told from the records")
            purchased, noticed = parse_date(txn.get("transaction_date")), parse_date(item.noticed_date)
            if purchased is None or noticed is None:
                return Refuse("noticed_date must be written MM/DD/YYYY")
            if noticed < purchased or noticed > ctx.today:
                return Refuse(
                    f"noticed_date {item.noticed_date} cannot be before the charge on {purchased:%m/%d/%Y} or after "
                    f"today, {ctx.today:%m/%d/%Y}. If the customer gave no year, use the year of the charge; do not ask them."
                )
            last_four = (item.card_last_4_digits or "").strip()
            if not last_four:
                return NeedInfo((Ask("card_last_4_digits", "What are the last four digits of the card used for this charge?"),))
            on_file = account.get("card_last_4_digits")
            if not (last_four.isdigit() and len(last_four) == 4) or (on_file and on_file != last_four):
                return Refuse("the last four digits do not match the card the charge is on; ask the customer to check them")
            charged = abs(money(txn.get("transaction_amount")))
            partial = item.partial_refund_amount
            if partial is not None and not 0 < partial < charged:
                return Refuse(
                    f"a partial refund must be more than zero and less than the charge of ${charged:.2f}. "
                    "If the customer disputes the whole charge, use null."
                )
            disputed = partial if partial is not None else charged
            args: dict[str, Any] = {
                "transaction_id": item.transaction_id,
                "card_action": "cancel_and_reissue" if item.cancel_and_reissue else "keep_active",
                "card_last_4_digits": last_four,
                **contact,
                "contacted_merchant": item.contacted_merchant,
                "purchase_date": purchased.strftime("%m/%d/%Y"),
                "issue_noticed_date": noticed.strftime("%m/%d/%Y"),
                "dispute_reason": item.reason,
                "resolution_requested": "full_refund" if partial is None else "partial_refund",
                "eligible_for_provisional_credit": provisional_credit(account, item, disputed, purchased, ctx.today, earlier),
            }
            if partial is not None:
                args["partial_refund_amount"] = partial
            steps.append(Step("file_credit_card_transaction_dispute_4829", args))
            earlier += 1
        return Plan(tuple(steps))
