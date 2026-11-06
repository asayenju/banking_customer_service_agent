"""What a savings account's latest monthly interest should have been, against what was paid. Pure code.

Expected APY = base APY for the balance tier + the highest checking boost + the highest credit card bonus
(+ the relationship bonus where the ruling in docs/fees-policy-decisions.md section 2 grants it). The
boosts and the card bonuses never add up with each other, only the highest of each applies (documents
045 and 046). The credit is the expected month's interest (balance x expected APY / 12) minus the
interest actually credited. That is how tau2's gold computes it: it matches all nine gold credits, where
balance x APY gap / 12 is a cent off on one of them from rounding the actual APY.

The balance is the current one; if anything other than the interest changed it since the interest was
paid, the review does not decide (the month's interest was earned on a different balance).
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from agent.bank_accounts import BankAccount, Product, accounts_of
from agent.fee_records import Txn, boost_for, cards_held, level_key, transactions_of
from agent.workflows.fee_products import (
    CARD_BONUS,
    RELATIONSHIP_BONUS,
    RELATIONSHIP_LEVELS,
    SAVINGS_BASE,
)

INTEREST_KIND = "interest_credit"
CORRECTION_KIND = "interest_correction"


@dataclass(frozen=True)
class InterestReview:
    level: str
    balance: float
    base: float
    checking_boost: float
    card_bonus: float
    relationship_bonus: float
    expected_apy: float
    actual_apy: float
    expected_interest: float
    paid_interest: float
    credit: float  # 0.0 when nothing is owed
    paid_on: Optional[date]
    corrected: bool  # an interest correction was already posted after that payment
    cannot_decide: Optional[str] = None

    def working(self) -> str:
        """The arithmetic in words, for the plan the customer approves."""
        return (
            f"${self.balance:,.2f} at {self.expected_apy:g}% for a month is ${self.expected_interest:,.2f}; "
            f"${self.paid_interest:,.2f} was paid"
        )


def _undecided(reason: str, level: str = "") -> InterestReview:
    return InterestReview(level, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, False, reason)


def cents(value: float) -> float:
    return round(value + 1e-9, 2)


def _base_apy(level: str, balance: float) -> float:
    apy = SAVINGS_BASE[level][0][0][1]
    for floor, rate in SAVINGS_BASE[level][0]:
        if balance >= floor:
            apy = rate
    return apy


def review_interest(db: Any, account: BankAccount, user_id: str) -> InterestReview:
    if account.product is not Product.SAVINGS:
        return _undecided("this is not a personal savings account")
    level = level_key(account.name, SAVINGS_BASE)
    if level is None:
        return _undecided(f"there is no interest table on file for a {account.name}")
    if account.balance is None or account.balance <= 0:
        return _undecided("the account's balance is not on file", level)
    rows: list[Txn] = transactions_of(db, account.account_id)
    interest = [t for t in rows if t.kind == INTEREST_KIND and t.amount > 0 and t.when is not None]
    if not interest:
        return _undecided("no monthly interest payment is on the account's transactions", level)
    paid = interest[-1]
    assert paid.when is not None
    moved = [t for t in rows if t.when is not None and t.when > paid.when and t.kind not in (CORRECTION_KIND,)]
    if moved:
        return _undecided("the balance changed after the interest was paid, so I cannot tell what it was earned on", level)
    # A correction already posted raised the balance; the month's interest was earned without it.
    posted = sum(t.amount for t in rows if t.kind == CORRECTION_KIND and t.when is not None and t.when >= paid.when and t.amount > 0)
    balance = cents(account.balance - posted)
    if balance <= 0:
        return _undecided("the account's balance is not on file", level)

    others = accounts_of(db, user_id)
    boosts = [boost_for(a.name, level) for a in others if a.is_open and a.product is Product.CHECKING]
    if any(b is None for b in boosts):
        return _undecided("a checking account of the customer's has no boost table on file", level)
    checking_boost = max((b for b in boosts if b is not None), default=0.0)
    cards, _business_or_unknown = cards_held(db, user_id)
    card_bonus = max((CARD_BONUS[level][c] for c in cards), default=0.0)
    has_checking = any(a.is_open and a.product is Product.CHECKING for a in others)
    relationship = RELATIONSHIP_BONUS if (level in RELATIONSHIP_LEVELS and has_checking and card_bonus <= 0.0) else 0.0

    base = _base_apy(level, balance)
    expected_apy = round(base + checking_boost + card_bonus + relationship, 4)
    actual_apy = round(paid.amount * 1200.0 / balance, 3)
    expected_interest = balance * expected_apy / 1200.0
    credit = max(0.0, cents(expected_interest - paid.amount))
    corrected = any(t.kind == CORRECTION_KIND and t.when is not None and t.when >= paid.when for t in rows)
    return InterestReview(
        level=level, balance=balance, base=base, checking_boost=checking_boost, card_bonus=card_bonus,
        relationship_bonus=relationship, expected_apy=expected_apy, actual_apy=actual_apy,
        expected_interest=cents(expected_interest), paid_interest=paid.amount, credit=credit,
        paid_on=paid.when, corrected=corrected,
    )
