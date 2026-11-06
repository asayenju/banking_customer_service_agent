"""What a card purchase should have earned, against what was recorded. Pure code, no model.

Points = floor(amount x rate x multiplier) for the card's rate on the purchase's category (document 007: always
rounded down, whole points), where a merchant on the card's exclusion list earns the default rate, and a card
that doubles its rewards for its first months does so only for a purchase inside them (document 012 of the
Business Silver card). The stored value and any ``expected_rewards`` in a dispute are never trusted
(document 004). A card, category or opening date we cannot place is not guessed: the review says why.
"""

import re
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_FLOOR, Decimal
from typing import Any, Mapping, Optional

from agent.cards import owned_credit_accounts
from agent.specialists.credit_rules import day, money
from agent.workflows.fee_products import REWARDS, RewardRule

_POINTS = re.compile(r"^\s*(\d+)\s+points\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class RewardsReview:
    expected: int
    stored: Optional[int]
    rate: float
    multiplier: int
    cannot_decide: Optional[str] = None

    @property
    def wrong(self) -> bool:
        return self.cannot_decide is None and self.stored != self.expected


def stored_points(row: Mapping[str, Any]) -> Optional[int]:
    found = _POINTS.match(str(row.get("rewards_earned") or ""))
    return int(found.group(1)) if found else None


def points_text(points: int) -> str:
    return f"{points} points"


def _undecided(reason: str) -> RewardsReview:
    return RewardsReview(0, None, 0.0, 1, reason)


def _months_after(start: date, months: int) -> date:
    total = start.month - 1 + months
    year, month = start.year + total // 12, total % 12 + 1
    for last in (start.day, 30, 29, 28):
        try:
            return date(year, month, last)
        except ValueError:
            continue
    return date(year, month, 28)


def _merchant_excluded(rule: RewardRule, merchant: str) -> bool:
    name = merchant.strip().lower()
    return any(name == m or name.startswith(m + " ") for m in rule.excluded_merchants)


def review_transaction(db: Any, row: Mapping[str, Any], user_id: str) -> RewardsReview:
    card_type = str(row.get("credit_card_type") or "")
    rule = REWARDS.get(card_type)
    if rule is None:
        return _undecided(f"there is no rewards table on file for a {card_type or 'card of unknown type'}")
    amount = money(row.get("transaction_amount"))
    when = day(row.get("transaction_date"))
    if amount is None or amount <= 0 or when is None:
        return _undecided("the purchase's amount or date is not on file")
    category = str(row.get("category") or "").strip().lower()
    rate = rule.default if _merchant_excluded(rule, str(row.get("merchant_name") or "")) else rule.rates.get(category, rule.default)
    multiplier = 1
    if rule.promo_months and rate > 0 and rule.promo_window is not None:
        accounts = [a for a in owned_credit_accounts(db, user_id) if str(a.get("card_type")) == card_type]
        opened = {day(a.get("date_of_account_open")) for a in accounts}
        if len(opened) != 1 or None in opened:
            return _undecided("I cannot tell when this card was opened, which decides whether its promotion applies")
        start = next(iter(opened))
        assert start is not None
        lo, hi = (date.fromisoformat(x) for x in rule.promo_window)
        if lo <= start <= hi and start <= when <= _months_after(start, rule.promo_months):
            multiplier = 2
    points = (Decimal(str(amount)) * Decimal(str(rate)) * multiplier).to_integral_value(rounding=ROUND_FLOOR)
    return RewardsReview(int(points), stored_points(row), rate, multiplier)
