"""Credit card type to tier. The credit policy documents speak of entry, mid and premium tiers
(retention offers, limit-increase rules, provisional credit caps). Only one place says which card
is which: the replacement-fee documents ("Credit Card Replacement ..."), which list the card types
of each tier. That table is copied here. A card type it does not list gets no tier, and a workflow
that needs the tier hands over instead of guessing.

The Crypto-Cash Back card is not in that table. The reference answers disagree about it (task 048
offers its owner the mid-tier $20 credit, task 049 the entry-tier $5 credit for the same card type),
so the policy does not decide it and neither do we.
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
}

# Card types the documents leave without a tier, on purpose.
UNMAPPED_CARD_TYPES: frozenset[str] = frozenset({"Crypto-Cash Back"})


def tier_of_card_type(card_type: str) -> Optional[str]:
    return TIER_OF_CARD_TYPE.get(card_type)
