"""The credit write tools' rules, as code.

The bank's credit tools check almost nothing: closing, statement credit, account flags,
replacements and disputes accept whatever they are given, and the limit approval refuses with one
generic message. These validators are the policy documents written as code (see
``docs/credit-policy-decisions.md``). Each is checked at plan time and again just before the step
runs. A validator returns None to accept, or a short reason written here.

Another customer's account, a made-up id and a closed account all read the same to the customer:
"that credit card account does not exist" (or is closed), so nothing about anyone else leaks.
"""

from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any, Optional

from agent.specialist import ValidationContext
from agent.workflows.credit_tiers import (
    CLI_MAX_INCREASE_SHARE,
    REPLACEMENT_LIMIT_60_DAYS,
    RETENTION_CREDIT,
    tier_of_card_type,
)

OPEN_DISPUTE_STATUSES = frozenset({"SUBMITTED", "UNDER_REVIEW", "PENDING"})
# The tool itself stores a new order as ORDERED and only blocks an approval for the others, so
# a replacement ordered in the same visit would not stop it. The policy says it should.
ACTIVE_ORDER_STATUSES = frozenset({"ORDERED", "PENDING", "PROCESSING", "SHIPPED"})
BAD_STANDING = frozenset({"PAST_DUE", "DELINQUENT", "COLLECTIONS", "CLOSED"})
CLOSURE_REASONS = frozenset(
    {"annual_fee", "not_using_card", "found_better_card", "unhappy_with_rewards", "simplifying_finances", "negative_experience", "other"}
)
DISPUTE_REASONS = frozenset(
    {
        "unauthorized_fraudulent_charge", "duplicate_charge", "incorrect_amount", "goods_services_not_received",
        "goods_services_not_as_described", "canceled_subscription_still_charging", "refund_never_processed",
    }
)
REPLACEMENT_REASONS = frozenset({"fraud_suspected", "lost", "stolen", "damaged", "expired", "other"})
CREDIT_REASONS = frozenset(
    {
        "goodwill_adjustment", "promotional_credit", "annual_fee_reversal", "late_fee_reversal", "interest_charge_reversal",
        "dispute_resolution", "price_match", "retention_offer", "error_correction", "other",
    }
)
FLAG_TYPES = frozenset({"annual_fee_waived", "promotional_apr", "rewards_bonus", "other"})
FLAG_REASONS = frozenset({"retention_offer", "loyalty_benefit", "promotional", "error_correction", "other"})
DENIAL_REASONS = frozenset(
    {
        "insufficient_account_age", "cooldown_period_active", "pending_disputes", "pending_replacement_card",
        "past_due_balance", "high_utilization", "insufficient_payment_history", "requested_amount_exceeds_limit", "other",
    }
)
MIN_AGE_FOR_CLOSING_DAYS = 60


def money(value: Any) -> Optional[float]:
    """"$1,234.56", "1234.56" or a number as dollars; None if it is not an amount."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None


def day(text: Any) -> Optional[date]:
    """A date written MM/DD/YYYY or YYYY-MM-DD, with or without a time after it."""
    from datetime import datetime

    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(text)[:10], fmt).date()
        except ValueError:
            continue
    return None


def years_after(start: date, years: int) -> date:
    try:
        return start.replace(year=start.year + years)
    except ValueError:  # 29 February
        return start.replace(year=start.year + years, day=28)


def _closed(account: Mapping[str, Any]) -> bool:
    return (
        str(account.get("status") or "").upper() == "CLOSED"
        or str(account.get("account_status") or "").upper() == "CLOSED"
        or bool(account.get("closed_date"))
    )


def account_row(v: ValidationContext, account_id: Any) -> Mapping[str, Any] | str:
    """The customer's open credit account, or the reason there is none."""
    row = v.database.credit_card_accounts.data.get(str(account_id))
    if not isinstance(row, Mapping) or row.get("user_id") != v.user_id:
        return "that credit card account does not exist"
    if _closed(row):
        return "that credit card account is already closed"
    return row


def active_orders(db: Any, account_id: str) -> list[Mapping[str, Any]]:
    return [
        o
        for o in db.credit_card_orders.data.values()
        if o.get("credit_card_account_id") == account_id and str(o.get("status")).upper() in ACTIVE_ORDER_STATUSES
    ]


def open_disputes(db: Any, user_id: str) -> list[Mapping[str, Any]]:
    return [
        d
        for d in db.transaction_disputes.data.values()
        if d.get("user_id") == user_id and str(d.get("status")).upper() in OPEN_DISPUTE_STATUSES
    ]


def account_disputes(db: Any, account: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Open disputes of credit transactions on this account. A dispute of a bank-account
    transaction is no reason to keep a card open."""
    found = []
    for dispute in open_disputes(db, str(account.get("user_id"))):
        txn = db.credit_card_transaction_history.data.get(str(dispute.get("transaction_id")))
        if not isinstance(txn, Mapping):
            continue
        if txn.get("credit_card_account_id") == account.get("account_id") or (
            not txn.get("credit_card_account_id") and txn.get("credit_card_type") == account.get("card_type")
        ):
            found.append(dispute)
    return found


def _tier(account: Mapping[str, Any]) -> Optional[str]:
    return tier_of_card_type(str(account.get("card_type")))


def _no_tier() -> str:
    return "this card type has no tier in the policy notes, so a person must decide this"


# ---------------------------------------------------------------- closing and retention

def log_reason(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    account = account_row(v, args.get("credit_card_account_id"))
    if isinstance(account, str):
        return account
    if args.get("closure_reason") not in CLOSURE_REASONS:
        return "that is not a closure reason the bank records"
    if any(r.get("credit_card_account_id") == account.get("account_id") for r in v.database.credit_card_closure_reasons.data.values()):
        # The tool would report success and store nothing.
        return "a closure reason is already on file for this card, so nothing new can be logged"
    return None


def close(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    account = account_row(v, args.get("credit_card_account_id"))
    if isinstance(account, str):
        return account
    balance = money(account.get("current_balance"))
    if balance is None:
        return "the balance on this card is not on file"
    if balance != 0:
        return f"the card has a balance of ${balance:,.2f} that must be paid off first"
    if account_disputes(v.database, account):
        return "the card has a dispute in progress; it can be closed when that is resolved"
    if active_orders(v.database, str(account.get("account_id"))):
        return "a replacement card is on its way; the card can be closed once it is received or cancelled"
    opened = day(account.get("date_of_account_open"))
    if opened is None:
        return "the date this account was opened is not on file"
    if (v.today - opened).days < MIN_AGE_FOR_CLOSING_DAYS:
        return "the account must be open at least 60 days before it can be closed"
    return None


def close_readback(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    row = v.database.credit_card_accounts.data.get(str(args.get("credit_card_account_id")))
    return None if isinstance(row, Mapping) and _closed(row) else "the account is not closed"


def statement_credit(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    account = account_row(v, args.get("credit_card_account_id"))
    if isinstance(account, str):
        return account
    amount = money(args.get("amount"))
    if amount is None or amount <= 0:
        return "a statement credit must be more than zero"
    reason = args.get("reason")
    if reason not in CREDIT_REASONS:
        return "that is not a reason a statement credit can be given for"
    if reason == "retention_offer":
        tier = _tier(account)
        if tier is None:
            return _no_tier()
        if amount != RETENTION_CREDIT[tier]:
            return f"the retention credit for this card is ${RETENTION_CREDIT[tier]:,.0f}"
    return None


def flag(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    account = account_row(v, args.get("credit_card_account_id"))
    if isinstance(account, str):
        return account
    if args.get("flag_type") not in FLAG_TYPES or args.get("reason") not in FLAG_REASONS:
        return "that is not a flag type or reason the bank uses"
    expires = day(args.get("expiration_date"))
    if expires is None:
        return "the expiry must be written MM/DD/YYYY"
    if expires <= v.today:
        return "the expiry must be in the future"
    if args.get("flag_type") == "annual_fee_waived" and expires != years_after(v.today, 1):
        return "an annual fee waiver lasts one year from today"
    if expires > years_after(v.today, 1):
        return "a flag cannot last more than a year"
    if args.get("reason") == "loyalty_benefit":
        opened = day(account.get("date_of_account_open"))
        if opened is None or years_after(opened, 2) > v.today:
            return "the loyalty benefit is for customers of two years or more"
    return None


def pay(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    account = account_row(v, args.get("credit_card_account_id"))
    if isinstance(account, str):
        return account
    checking = v.database.accounts.data.get(str(args.get("checking_account_id")))
    if not isinstance(checking, Mapping) or checking.get("user_id") != v.user_id:
        return "that checking account does not exist"
    if checking.get("class") != "checking" or checking.get("status") != "OPEN":
        return "the payment must come from an open checking account"
    amount = money(args.get("amount"))
    if amount is None or amount <= 0:
        return "a payment must be more than zero"
    balance = money(account.get("current_balance")) or 0.0
    if amount > balance:
        return f"the card's balance is ${balance:,.2f}, which is less than that"
    funds = money(checking.get("current_holdings")) or 0.0
    if amount > funds:
        return f"the checking account has ${funds:,.2f}, which is not enough"
    return None


# ---------------------------------------------------------------- replacement and disputes

def replacement(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    account = account_row(v, args.get("credit_card_account_id"))
    if isinstance(account, str):
        return account
    if args.get("reason") not in REPLACEMENT_REASONS:
        return "that is not a reason a card can be replaced for"
    if not str(args.get("shipping_address") or "").strip():
        return "a shipping address is needed"
    if active_orders(v.database, str(account.get("account_id"))):
        return "a replacement card has already been ordered for this account"
    tier = _tier(account)
    if tier is None:
        return _no_tier()
    recent = [
        o
        for o in v.database.credit_card_orders.data.values()
        if o.get("credit_card_account_id") == account.get("account_id")
        and (placed := day(o.get("order_date"))) is not None
        and timedelta(0) <= v.today - placed <= timedelta(days=60)
    ]
    if len(recent) >= REPLACEMENT_LIMIT_60_DAYS[tier]:
        return "this account has used all its replacement cards for the last 60 days"
    return None


def dispute(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    """Ground a credit dispute in the records. The bank's tool files whatever it is given."""
    txn = v.database.credit_card_transaction_history.data.get(str(args.get("transaction_id")))
    if not isinstance(txn, Mapping) or txn.get("user_id") != v.user_id:
        return "that transaction does not exist"
    if args.get("dispute_reason") not in DISPUTE_REASONS:
        return "that is not a dispute reason the bank uses"
    if args.get("card_action") not in ("keep_active", "cancel_and_reissue"):
        return "the card action is keep_active or cancel_and_reissue"
    if not (str(args.get("card_last_4_digits") or "").isdigit() and len(str(args.get("card_last_4_digits"))) == 4):
        return "the last four digits of the card are needed"
    for field_name in ("full_name", "phone", "email", "address"):
        if not str(args.get(field_name) or "").strip():
            return f"the {field_name.replace('_', ' ')} is needed to file a dispute"
    if any(
        d.get("user_id") == v.user_id and d.get("transaction_id") == args.get("transaction_id")
        for d in v.database.transaction_disputes.data.values()
    ):
        return "this transaction has already been disputed"
    charged = abs(money(txn.get("transaction_amount")) or 0.0)
    resolution = args.get("resolution_requested")
    partial = args.get("partial_refund_amount")
    if resolution == "partial_refund":
        amount = money(partial)
        if amount is None or amount <= 0 or amount >= charged:
            return "a partial refund must be more than zero and less than the transaction"
    elif resolution == "full_refund":
        if partial is not None:
            return "a full refund has no partial amount"
    else:
        return "the resolution is full_refund or partial_refund"
    bought, noticed = day(args.get("purchase_date")), day(args.get("issue_noticed_date"))
    if bought is None or noticed is None:
        return "dates must be written MM/DD/YYYY"
    if bought != day(txn.get("transaction_date")):
        return "the purchase date is not the date on the transaction"
    if noticed < bought or noticed > v.today:
        return "the customer cannot have noticed it before it happened or after today"
    return None


# ---------------------------------------------------------------- limit increases

def _limit(account: Mapping[str, Any]) -> Optional[float]:
    return money(account.get("credit_limit"))


def submit(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    account = account_row(v, args.get("credit_card_account_id"))
    if isinstance(account, str):
        return account
    amount = args.get("requested_increase_amount")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0 or amount != int(amount):
        return "the increase must be a whole number of dollars above zero"
    tier = _tier(account)
    if tier is None:
        return _no_tier()
    limit = _limit(account)
    if limit is None or limit <= 0:
        return "there is no credit limit on file for this card, so a person must handle the request"
    cap = limit * CLI_MAX_INCREASE_SHARE[tier]
    if amount > cap:
        return f"this card can be raised by at most ${cap:,.0f} at a time"
    return None


def approve(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    """The bank's tool refuses for three hidden reasons with one generic message. Each is named here."""
    account = account_row(v, args.get("credit_card_account_id"))
    if isinstance(account, str):
        return account
    if open_disputes(v.database, v.user_id):
        return "a dispute is in progress, which blocks an approval until it is resolved"
    if active_orders(v.database, str(account.get("account_id"))):
        return "a replacement card is on its way, which blocks an approval until it is received or cancelled"
    standing = str(account.get("account_status") or "").upper()
    if standing in BAD_STANDING:
        return "the account is not in good standing"
    if (money(account.get("past_due_amount")) or 0.0) > 0:
        return "the account has a past-due balance"
    tier = _tier(account)
    if tier is None:
        return _no_tier()
    limit = _limit(account)
    if limit is None or limit <= 0:
        return "there is no credit limit on file for this card, so a person must handle the request"
    new = args.get("new_credit_limit")
    if isinstance(new, bool) or not isinstance(new, (int, float)) or new != int(new):
        return "the new limit must be a whole number of dollars"
    if new <= limit:
        return "the new limit must be higher than the current one"
    if new - limit > limit * CLI_MAX_INCREASE_SHARE[tier]:
        return f"this card can be raised by at most ${limit * CLI_MAX_INCREASE_SHARE[tier]:,.0f} at a time"
    return None


def approve_readback(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    row = v.database.credit_card_accounts.data.get(str(args.get("credit_card_account_id")))
    now = money(row.get("credit_limit")) if isinstance(row, Mapping) else None
    return None if now == float(args["new_credit_limit"]) else "the limit was not changed"


def deny(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    account = account_row(v, args.get("credit_card_account_id"))
    if isinstance(account, str):
        return account
    if args.get("denial_reason") not in DENIAL_REASONS:
        return "that is not a denial reason the bank uses"
    return None
