"""What a checking account's ATM fees should have been, against what was charged. Pure code, no model.

For each calendar month: the withdrawals are walked in date order, the bank's own ATMs cost nothing,
the free withdrawals of the class are used up (out-of-network and foreign counted separately), and
every other withdrawal costs what the class's schedule says. The expected net cost is the expected
fees less the rebate the class gives (up to its cap). The credit is the net actually paid (fees
charged less rebates and refunds already posted) minus the expected net, summed over the months,
never below zero. Over-charges and under-charges on one account net out (docs/fees-policy-decisions.md,
section 3): a fee on the wrong withdrawal that totals the right amount is no credit, and a missed fee
is never billed to the customer, only set against an over-charge.

A class with no schedule, or a withdrawal whose place cannot be told, is not guessed: the review
says why it cannot decide and the caller hands off.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from agent.bank_accounts import BankAccount
from agent.fee_records import is_foreign, is_rho_atm, level_key, transactions_of
from agent.workflows.fee_products import ATM_FEES, CHECKING_CREDIT_COOLDOWN_DAYS

CREDIT_KINDS = frozenset({"fee_refund", "rebate_credit"})


@dataclass(frozen=True)
class FeeReview:
    credit: float  # dollars owed to the customer, 0.0 when nothing is owed
    over_charged: float  # fees charged above the schedule, net of rebates
    under_charged: float  # fees the schedule required that were not charged
    months: tuple[str, ...] = ()
    cannot_decide: Optional[str] = None  # why the review stops short (hand off), else None
    notes: tuple[str, ...] = field(default=())


def _undecided(reason: str) -> FeeReview:
    return FeeReview(credit=0.0, over_charged=0.0, under_charged=0.0, cannot_decide=reason)


def cents(value: float) -> float:
    return round(value + 1e-9, 2)


def review_atm_fees(db: Any, account: BankAccount) -> FeeReview:
    level = level_key(account.name, ATM_FEES)
    if level is None:
        return _undecided(f"there is no fee schedule on file for a {account.name}")
    fees = ATM_FEES[level]
    by_month: dict[tuple[int, int], list[Any]] = defaultdict(list)
    for t in transactions_of(db, account.account_id):
        if t.when is not None:
            by_month[(t.when.year, t.when.month)].append(t)

    total_over = total_under = 0.0
    months: list[str] = []
    for (year, month), rows in sorted(by_month.items()):
        oon_seen = foreign_seen = 0
        expected_oon = expected_foreign = 0.0
        charged = rebated = 0.0
        for t in rows:
            if t.kind == "atm_withdrawal" and t.amount < 0:
                if is_rho_atm(t.description):
                    continue
                abroad = is_foreign(t.description)
                if abroad is None:
                    return _undecided("a withdrawal's place could not be told from its description")
                amount = abs(t.amount)
                if abroad:
                    foreign_seen += 1
                    if foreign_seen > fees.foreign.free_per_month:
                        expected_foreign += fees.foreign.fee(amount)
                else:
                    oon_seen += 1
                    if oon_seen > fees.out_of_network.free_per_month:
                        expected_oon += fees.out_of_network.fee(amount)
            elif t.kind == "atm_fee" and t.amount < 0:
                charged += -t.amount
            elif (t.kind == "fee_rebate" or t.kind in CREDIT_KINDS) and t.amount > 0:
                rebated += t.amount
        expected_rebate = min(expected_oon, fees.rebate_cap) if fees.rebate_cap is not None else 0.0
        expected_net = expected_foreign + expected_oon - expected_rebate
        paid_net = charged - rebated
        gap = paid_net - expected_net
        if gap > 0:
            total_over += gap
        else:
            total_under += -gap
        if expected_net or charged or rebated:
            months.append(f"{year:04d}-{month:02d}")
    # Over- and under-charges net out per account over the whole review: a mis-timed fee is no credit.
    credit = max(0.0, cents(total_over - total_under))
    return FeeReview(
        credit=credit,
        over_charged=cents(total_over),
        under_charged=cents(total_under),
        months=tuple(months),
    )


def recent_credit(db: Any, account: BankAccount, today: date) -> Optional[date]:
    """The date of a fee refund or rebate credit posted in the last 14 days, if there is one."""
    latest: Optional[date] = None
    for t in transactions_of(db, account.account_id):
        if t.kind in CREDIT_KINDS and t.amount > 0 and t.when is not None:
            if 0 <= (today - t.when).days < CHECKING_CREDIT_COOLDOWN_DAYS and (latest is None or t.when > latest):
                latest = t.when
    return latest
