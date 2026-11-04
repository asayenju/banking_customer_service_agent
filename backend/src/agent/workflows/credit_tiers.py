"""Credit card type to tier. The credit policy documents speak of entry, mid and premium tiers
(retention offers, limit-increase rules, provisional credit caps). Only one place says which card
is which: the replacement-fee documents ("Credit Card Replacement ..."), which list the card types
of each tier. That table is copied here. A card type no document places gets no tier, and a workflow
that needs the tier hands over instead of guessing.

The Crypto-Cash Back card is not in the replacement table. The provisional-credit document
(``doc_credit_cards_credit_cards_(general)_015``) lists it with the entry tier, which is the only place
any document names its tier, so it is entry here. The reference answers disagree about it: task 049
offers its owner the entry-tier $5 retention credit (policy-correct), task 048 the mid-tier $20 for the
same card type (a departure from policy, recorded as an exclusion).
"""

from typing import Optional

ENTRY = "Entry"
MID = "Mid"
PREMIUM = "Premium"  # "Premium-tier and above" in the documents: Gold, Platinum, Diamond Elite.

TIER_OF_CARD_TYPE: dict[str, str] = {
    "Bronze Rewards Card": ENTRY,
    "EcoCard": ENTRY,
    "Business Bronze Rewards Card": ENTRY,
    "Silver Rewards Card": MID,
    "Business Silver Rewards Card": MID,
    "Green Rewards Card": MID,
    "Silver Zoom Card": MID,
    "Gold Rewards Card": PREMIUM,
    "Business Gold Rewards Card": PREMIUM,
    "Platinum Rewards Card": PREMIUM,
    "Business Platinum Rewards Card": PREMIUM,
    "Diamond Elite Card": PREMIUM,
    "Crypto-Cash Back": ENTRY,  # placed only by the provisional-credit document
}

# Card types the documents leave without a tier, on purpose.
UNMAPPED_CARD_TYPES: frozenset[str] = frozenset()


def tier_of_card_type(card_type: str) -> Optional[str]:
    return TIER_OF_CARD_TYPE.get(card_type)


# What the policy documents give per tier ("Credit Card Retention Protocol", "CLI Eligibility
# Requirements by Card Tier", "CLI Payment History and Approval Criteria", replacement documents).
RETENTION_CREDIT: dict[str, float] = {ENTRY: 5.0, MID: 20.0, PREMIUM: 50.0}
# Replacements allowed in 60 days: entry 2, mid 3, premium and above 4.
REPLACEMENT_LIMIT_60_DAYS: dict[str, int] = {ENTRY: 2, MID: 3, PREMIUM: 4}
# Largest increase per request, as a share of the current limit.
CLI_MAX_INCREASE_SHARE: dict[str, float] = {ENTRY: 0.25, MID: 0.50, PREMIUM: 0.50}
# Expedited delivery of a replacement card: the fee the customer must agree to. No bank tool charges it.
EXPEDITED_FEE: dict[str, float] = {ENTRY: 15.0, MID: 10.0, PREMIUM: 0.0}

# Largest amount that can be credited provisionally while a dispute is investigated, by card type
# ("Credit card disputes: provisional credit", which splits the premium tier and above more finely).
PROVISIONAL_CREDIT_CAP: dict[str, float] = {
    "Bronze Rewards Card": 2500.0, "EcoCard": 2500.0, "Business Bronze Rewards Card": 2500.0, "Crypto-Cash Back": 2500.0,
    "Silver Rewards Card": 5000.0, "Business Silver Rewards Card": 5000.0, "Green Rewards Card": 5000.0, "Silver Zoom Card": 5000.0,
    "Gold Rewards Card": 10000.0, "Business Gold Rewards Card": 10000.0,
    "Platinum Rewards Card": 15000.0, "Business Platinum Rewards Card": 15000.0,
    "Diamond Elite Card": 25000.0,
}

# "CLI Eligibility Requirements by Card Tier" and "CLI Payment History and Approval Criteria".
CLI_MIN_ACCOUNT_AGE_DAYS: dict[str, int] = {ENTRY: 120, MID: 90, PREMIUM: 60}
CLI_COOLDOWN_DAYS: dict[str, int] = {ENTRY: 120, MID: 90, PREMIUM: 60}
# Utilization must be below this share of the limit.
CLI_MAX_UTILIZATION: dict[str, float] = {ENTRY: 0.70, MID: 0.80, PREMIUM: 0.90}
CLI_ON_TIME_PAYMENTS: dict[str, int] = {ENTRY: 6, MID: 3, PREMIUM: 3}
