"""The fee, interest and rewards write tools' rules, as code.

The bank's credit tools check a positive amount, the account class and an open status, and nothing else:
not who owns the account, not how large the credit is, not whether one was just given, and a repeat call
with the same arguments adds the balance again (docs/fees-policy-decisions.md, section 1, with a test for
each statement). These validators are the policy documents written as code. Each is checked at plan time
and again just before the step runs. A validator returns None to accept, or a short reason written here.

Another customer's account, a made-up id and a closed account all read the same to the customer.
"""

from collections.abc import Mapping
from typing import Any, Optional

from tau2.domains.banking_knowledge.utils import _deterministic_id, get_today_str

from agent import bank_accounts as accounts
from agent.bank_accounts import Product
from agent.fee_review import cents as money_cents, recent_credit, review_atm_fees
from agent.specialist import ValidationContext
from agent.specialists.accounts_rules import dollars
from agent.specialists.credit_rules import money
from agent.workflows.fee_products import ATM_FEES

NOT_FOUND = "that account does not exist"
CHECKING_CREDIT_TOOL = "apply_checking_account_credit_5829"
CHECKING_CREDIT_TYPES = frozenset({"rebate_credit", "fee_refund"})


def _two_places(amount: float) -> bool:
    return abs(amount * 100 - round(amount * 100)) < 1e-6


def _checking_credit_id(args: Mapping[str, Any]) -> str:
    """The id the bank's tool gives the credit's transaction row (it is a hash of these arguments)."""
    amount = float(args["amount"])
    seed = f"checking_credit:{args['account_id']}:{args['credit_type']}:{amount}:{get_today_str()}"
    return f"txn_{_deterministic_id(seed)}"


def checking_credit(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    account = accounts.get(v.database, v.user_id, args.get("account_id"))
    if account is None:
        return NOT_FOUND
    if account.product is not Product.CHECKING:
        return "ATM fee credits apply to a personal checking account"
    if not account.is_open:
        return "that account is not open"
    if str(args.get("credit_type")) not in CHECKING_CREDIT_TYPES:
        return "that is not a type of checking credit"
    amount = money(args.get("amount"))
    if amount is None or amount != amount or amount in (float("inf"), float("-inf")) or money_cents(amount) <= 0:
        return "a credit must be a dollar amount above zero"
    if not _two_places(amount):
        return "a credit is in dollars and cents"
    if account.name not in ATM_FEES:
        return f"there is no fee schedule on file for a {account.name}, so I cannot confirm a credit"
    if (last := recent_credit(v.database, account, v.today)) is not None:
        return f"a fee credit was already applied to this account on {last:%m/%d/%Y}; the policy allows one every 14 days"
    review = review_atm_fees(v.database, account)
    if review.cannot_decide is not None:
        return f"I cannot confirm the fees on this account: {review.cannot_decide}"
    if money_cents(review.credit) <= 0:
        return "the fees on this account match the account's fee schedule, so there is nothing to credit"
    if money_cents(amount) != money_cents(review.credit):
        return f"the fees charged above the schedule come to {dollars(review.credit)}, not {dollars(amount)}"
    if str(args.get("credit_type")) != review.credit_type:
        return f"this credit is a {review.credit_type.replace('_', ' ')}"
    if _checking_credit_id(args) in v.database.bank_account_transaction_history.data:
        # The bank would add the balance a second time and rewrite the same row.
        return "this exact credit was already applied today"
    return None


def checking_credit_readback(v: ValidationContext, args: Mapping[str, Any]) -> Optional[str]:
    """The tool's failures are loud; what must hold afterwards is the posted row for this amount."""
    row = v.database.bank_account_transaction_history.data.get(_checking_credit_id(args))
    if not isinstance(row, Mapping) or row.get("account_id") != args.get("account_id"):
        return "the credit is not on the account's transactions"
    posted = money(row.get("amount"))
    if posted is None or money_cents(posted) != money_cents(float(args["amount"])):
        return "the credit posted for a different amount"
    return None
