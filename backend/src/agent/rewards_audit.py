"""Which purchases earned the wrong rewards, found by the agent itself.

Every purchase on the customer's credit cards is recalculated from the card's earning rules (``rewards_review``, pure code:
the stored value is never trusted). The customer is not asked to pick or to remember anything. This only finds them; the
cash back dispute that must come before a correction is submitted by the rewards workflow, as a plan the customer approves.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent.rewards_review import RewardsReview, review_transaction
from orchestrator.types import Intent, IntentKind

_REWARDS = re.compile(r"\b(?:rewards?|cash ?back|points)\b", re.IGNORECASE)
WHICH_PURCHASE = ("purchase", "transaction", "charge")


@dataclass(frozen=True)
class Audit:
    wrong: tuple[tuple[Mapping[str, Any], RewardsReview], ...]  # biggest difference first
    right: int
    undecided: int


def audit(db: Any, user_id: str, rows: list[Mapping[str, Any]]) -> Audit:
    wrong: list[tuple[Mapping[str, Any], RewardsReview]] = []
    right = undecided = 0
    for row in rows:
        review = review_transaction(db, row, user_id)
        if review.cannot_decide is not None:
            undecided += 1
        elif review.wrong:
            wrong.append((row, review))
        else:
            right += 1
    wrong.sort(key=lambda w: -abs(w[1].expected - (w[1].stored or 0)))
    return Audit(tuple(wrong), right, undecided)


def is_rewards_complaint(intent: Intent) -> bool:
    return intent.kind is IntentKind.FEE_OR_INTEREST_ADJUSTMENT and bool(
        _REWARDS.search(f"{intent.summary} {' '.join(str(v) for v in intent.slots.values())}")
    )
