"""The fee and interest policy as data: ATM fee schedules, savings APY tiers, card and checking bonuses.

Source: the product pages (``doc_checking_accounts_*``, ``doc_savings_accounts_*``) and
``doc_bank_accounts_bank_accounts_(general)_012``, ``_045`` and ``_046``. Decisions, rulings on
documents that disagree, and the evidence for each value are in ``docs/fees-policy-decisions.md``.
Pure data, no logic: ``tests/agent/test_fees_products.py`` checks every value against the document
text it names, so a value cannot drift from its source. A product that is not in a table here is a
hand-off, never a guess.
"""

from dataclasses import dataclass
from typing import Optional

INF = float("inf")


@dataclass(frozen=True)
class AtmRule:
    """What one ATM withdrawal costs: a flat fee or a percentage with a floor and a ceiling, or tiers
    by amount, after some free withdrawals a month."""

    flat: float = 0.0
    pct: float = 0.0  # percent of the withdrawal, e.g. 1.0 for 1%
    minimum: float = 0.0
    maximum: float = INF
    tiers: tuple[tuple[float, float], ...] = ()  # (largest amount in the tier, fee); amounts up to and including it
    free_per_month: int = 0
    docs: tuple[str, ...] = ()

    def fee(self, amount: float) -> float:
        """The fee for one withdrawal of ``amount`` once the free ones are used up."""
        if self.tiers:
            for ceiling, fee in self.tiers:
                if amount <= ceiling:
                    return fee
            return self.tiers[-1][1]
        if self.pct:
            return round(min(max(amount * self.pct / 100.0, self.minimum), self.maximum), 2)
        return self.flat


@dataclass(frozen=True)
class AtmFees:
    out_of_network: AtmRule
    foreign: AtmRule
    rebate_cap: Optional[float] = None  # out-of-network fees are rebated up to this much a month


ATM_FEES: dict[str, AtmFees] = {
    "Blue Account": AtmFees(
        AtmRule(pct=1.0, maximum=3.0, docs=("checking_accounts_blue_account_001",)),
        AtmRule(pct=3.0, minimum=5.0, docs=("checking_accounts_blue_account_012",)),
    ),
    "Bluest Account": AtmFees(
        AtmRule(flat=2.0, docs=("checking_accounts_bluest_account_010",)),
        AtmRule(docs=("checking_accounts_bluest_account_003",)),
        rebate_cap=50.0,
    ),
    "Dark Green Account": AtmFees(
        AtmRule(pct=1.0, minimum=1.5, docs=("checking_accounts_dark_green_account_001",)),
        AtmRule(pct=2.5, maximum=6.0, docs=("checking_accounts_dark_green_account_002",)),
    ),
    "Evergreen Account": AtmFees(
        AtmRule(pct=1.0, maximum=2.5, docs=("checking_accounts_evergreen_account_008",)),
        AtmRule(pct=2.0, minimum=3.0, docs=("checking_accounts_evergreen_account_001",)),
    ),
    "Green Account": AtmFees(
        AtmRule(flat=3.0, docs=("checking_accounts_green_account_(checking)_001",)),
        AtmRule(pct=3.0, minimum=5.0, docs=("checking_accounts_green_account_(checking)_012",)),
    ),
    "Light Blue Account": AtmFees(
        AtmRule(flat=2.5, free_per_month=2, docs=("checking_accounts_light_blue_account_004",)),
        AtmRule(flat=4.0, free_per_month=2, docs=("checking_accounts_light_blue_account_006",)),
    ),
    "Light Green Account": AtmFees(
        AtmRule(flat=1.5, free_per_month=4, docs=("checking_accounts_light_green_account_001",)),
        AtmRule(tiers=((100.0, 2.0), (300.0, 3.5), (INF, 5.0)), docs=("checking_accounts_light_green_account_013",)),
    ),
    "Purple Account": AtmFees(
        AtmRule(flat=2.5, docs=("checking_accounts_purple_account_012",)),
        AtmRule(docs=("checking_accounts_purple_account_001",)),
        rebate_cap=30.0,
    ),
    # Gold Years and Green Fee-Free are not here: their ATM terms are not in the documents we read, so a
    # fee review on either is a hand-off.
}

# Savings base APY by balance: (smallest balance in the tier, APY percent), the document that says so.
SAVINGS_BASE: dict[str, tuple[tuple[tuple[float, float], ...], str]] = {
    "Bronze Account": (((0.0, 2.0),), "savings_accounts_bronze_account_001"),
    "Silver Account": (((0.0, 2.5), (10000.0, 4.0)), "savings_accounts_silver_account_003"),
    "Silver Plus Account": (((0.0, 3.0), (15000.0, 4.5)), "savings_accounts_silver_plus_account_002"),
    "Gold Account": (((0.0, 5.5),), "savings_accounts_gold_account_013"),
    "Gold Plus Account": (((0.0, 6.0),), "savings_accounts_gold_plus_account_002"),
    "Platinum Account": (((0.0, 6.5),), "savings_accounts_platinum_account_001"),
    "Platinum Plus Account": (((0.0, 7.0),), "savings_accounts_platinum_plus_account_002"),
    "Diamond Elite Account": (((0.0, 7.5),), "savings_accounts_diamond_elite_account_001"),
}

CARD_NAMES = (
    "Bronze Rewards Card", "Silver Rewards Card", "Gold Rewards Card", "Platinum Rewards Card",
    "Diamond Elite Card", "EcoCard", "Green Rewards Card", "Crypto-Cash Back Card",
)

# APY bonus a credit card adds to a savings account, by savings level then card (percent). Only the
# highest applies (document 045). The document each table is read from is in CARD_BONUS_DOC.
CARD_BONUS: dict[str, dict[str, float]] = {
    "Bronze Account": {
        "Bronze Rewards Card": 0.1, "Silver Rewards Card": 0.0, "Gold Rewards Card": 0.25,
        "Platinum Rewards Card": 0.55, "Diamond Elite Card": 0.15, "EcoCard": 0.0,
        "Green Rewards Card": 0.2, "Crypto-Cash Back Card": 0.3,
    },
    "Silver Account": {
        "Bronze Rewards Card": 0.0, "Silver Rewards Card": 0.1, "Gold Rewards Card": 0.5,
        "Platinum Rewards Card": 0.2, "Diamond Elite Card": 0.25, "EcoCard": 2.2,
        "Green Rewards Card": 0.0, "Crypto-Cash Back Card": 0.5,
    },
    "Silver Plus Account": {
        "Bronze Rewards Card": 0.15, "Silver Rewards Card": 0.15, "Gold Rewards Card": 0.2,
        "Platinum Rewards Card": 0.15, "Diamond Elite Card": 0.4, "EcoCard": 0.45,
        "Green Rewards Card": 0.1, "Crypto-Cash Back Card": 0.0,
    },
    "Gold Account": {
        "Bronze Rewards Card": 0.15, "Silver Rewards Card": 0.2, "Gold Rewards Card": 0.025,
        "Platinum Rewards Card": 0.15, "Diamond Elite Card": 0.3, "EcoCard": 0.6,
        "Green Rewards Card": 0.35, "Crypto-Cash Back Card": 0.0,
    },
    "Gold Plus Account": {
        "Bronze Rewards Card": 0.15, "Silver Rewards Card": 0.1, "Gold Rewards Card": 0.35,
        "Platinum Rewards Card": 0.2, "Diamond Elite Card": 0.25, "EcoCard": 0.1,
        "Green Rewards Card": 0.05, "Crypto-Cash Back Card": 0.3,
    },
    "Platinum Account": {
        "Bronze Rewards Card": 0.0, "Silver Rewards Card": 0.0, "Gold Rewards Card": 0.15,
        "Platinum Rewards Card": 0.25, "Diamond Elite Card": 0.35, "EcoCard": 0.0,
        "Green Rewards Card": 0.0, "Crypto-Cash Back Card": 0.0,
    },
    "Platinum Plus Account": {
        "Bronze Rewards Card": 0.0, "Silver Rewards Card": 0.0, "Gold Rewards Card": 0.1,
        "Platinum Rewards Card": 0.4, "Diamond Elite Card": 0.6, "EcoCard": 0.05,
        "Green Rewards Card": 0.0, "Crypto-Cash Back Card": 0.25,
    },
    "Diamond Elite Account": {
        "Bronze Rewards Card": 0.0, "Silver Rewards Card": 0.0, "Gold Rewards Card": 0.0,
        "Platinum Rewards Card": 0.1, "Diamond Elite Card": 0.5, "EcoCard": 0.0,
        "Green Rewards Card": 0.0, "Crypto-Cash Back Card": 0.15,
    },
}
CARD_BONUS_DOC: dict[str, str] = {
    "Bronze Account": "savings_accounts_bronze_account_007",
    "Silver Account": "savings_accounts_silver_account_006",
    "Silver Plus Account": "savings_accounts_silver_plus_account_009",
    "Gold Account": "savings_accounts_gold_account_014",
    "Gold Plus Account": "savings_accounts_gold_plus_account_009",
    "Platinum Account": "savings_accounts_platinum_account_010",
    "Platinum Plus Account": "savings_accounts_platinum_plus_account_009",
    "Diamond Elite Account": "savings_accounts_diamond_elite_account_011",
}

# Checking-to-savings APY boost (percent): the 18 pairings of document 012, never any other
# combination. Only the highest applies (document 046). Value and evidence document per pairing.
CHECKING_BOOST: dict[tuple[str, str], tuple[float, str]] = {
    ("Green Account", "Gold Account"): (0.75, "checking_accounts_green_account_(checking)_001"),
    ("Green Account", "Silver Account"): (0.25, "checking_accounts_green_account_(checking)_001"),
    ("Green Fee-Free Account", "Bronze Account"): (0.4, "checking_accounts_green_fee-free_account_005"),
    ("Green Fee-Free Account", "Gold Plus Account"): (0.35, "checking_accounts_green_fee-free_account_005"),
    ("Evergreen Account", "Green Account"): (0.55, "checking_accounts_evergreen_account_001"),
    ("Evergreen Account", "Diamond Elite Account"): (0.15, "checking_accounts_evergreen_account_001"),
    ("Blue Account", "Silver Plus Account"): (0.35, "checking_accounts_blue_account_003"),
    ("Blue Account", "Platinum Account"): (0.8, "checking_accounts_blue_account_003"),
    ("Light Green Account", "Diamond Elite Account"): (0.2, "checking_accounts_light_green_account_002"),
    ("Light Green Account", "Platinum Account"): (0.65, "checking_accounts_light_green_account_002"),
    ("Dark Green Account", "Bronze Account"): (0.85, "checking_accounts_dark_green_account_004"),
    ("Dark Green Account", "Platinum Plus Account"): (0.45, "checking_accounts_dark_green_account_004"),
    ("Bluest Account", "Silver Account"): (0.45, "checking_accounts_bluest_account_003"),
    ("Bluest Account", "Bronze Account"): (0.7, "checking_accounts_bluest_account_003"),
    ("Purple Account", "Platinum Plus Account"): (0.3, "checking_accounts_purple_account_001"),
    ("Purple Account", "Gold Account"): (0.1, "checking_accounts_purple_account_001"),
    ("Gold Years Account", "Gold Plus Account"): (0.5, "checking_accounts_gold_years_account_001"),
    ("Gold Years Account", "Silver Account"): (0.6, "checking_accounts_gold_years_account_001"),
}

# The relationship bonus (percent). The documents say a customer "may qualify" and give no criteria for
# Silver and Silver Plus (silver 002 and 005, silver plus 002, document 044); for Gold and Silver (with a
# Platinum card) the same 0.025 is tied to a card, and document 045 counts the Gold one among the card
# bonuses that do not stack. Ruling (docs/fees-policy-decisions.md, section 2): grant it only to a Silver
# or Silver Plus holder with a linked checking account and no card bonus; never to anyone else.
RELATIONSHIP_BONUS = 0.025
RELATIONSHIP_LEVELS = frozenset({"Silver Account", "Silver Plus Account"})

# Hand-off thresholds the tools do not enforce (documents 017 and 043).
GOODWILL_SUPERVISOR_LIMIT = 25.0
CHECKING_CREDIT_COOLDOWN_DAYS = 14


# ---------------------------------------------------------------- credit card rewards
# Points earned = floor(amount x rate x multiplier), the rate in percent (1 point = $0.01, so 4% is 4 points
# a dollar; the EcoCard's "5 points a dollar" is 5 here too). Documents credit_cards_(general)_004, _006 and
# _007 (floor, whole points), the card pages named per rule, and the merchants that do not earn the higher
# rate. A card not in this table is a hand-off.
@dataclass(frozen=True)
class RewardRule:
    rates: dict[str, float]  # transaction category (lowercase) -> percent
    default: float
    excluded_merchants: frozenset[str] = frozenset()  # lowercase; these earn the default rate
    promo_months: int = 0  # the card doubles its rewards for this many months after opening ...
    promo_window: Optional[tuple[str, str]] = None  # ... if opened inside this window (ISO dates)
    docs: tuple[str, ...] = ()


REWARDS: dict[str, RewardRule] = {
    "Business Silver Rewards Card": RewardRule(
        rates={"travel": 10.0, "software": 10.0},
        default=1.0,
        excluded_merchants=frozenset(
            {
                "concur", "sap concur", "expensify", "navan", "apple", "microsoft", "dell", "xbox game pass",
                "playstation plus", "nintendo switch online", "coursera", "udemy", "linkedin learning", "skillshare",
                "pluralsight",
            }
        ),
        promo_months=6,
        promo_window=("2024-11-14", "2025-11-14"),
        docs=(
            "business_credit_cards_business_silver_rewards_card_002",
            "business_credit_cards_business_silver_rewards_card_005",
            "business_credit_cards_business_silver_rewards_card_012",
        ),
    ),
    "Silver Rewards Card": RewardRule(
        rates={"travel": 4.0, "software": 4.0}, default=1.0, docs=("credit_cards_silver_rewards_card_002",)
    ),
    "Business Platinum Rewards Card": RewardRule(
        rates={"travel": 4.0, "software": 4.0, "media": 4.0}, default=1.5,
        docs=("business_credit_cards_business_platinum_rewards_card_002",),
    ),
    "Crypto-Cash Back": RewardRule(rates={}, default=2.0, docs=("credit_cards_crypto-cash_back_001",)),
    "EcoCard": RewardRule(
        rates={"green": 5.0}, default=1.0, excluded_merchants=frozenset({"target", "walmart", "amazon", "thredup"}),
        docs=("credit_cards_ecocard_002", "credit_cards_ecocard_004"),
    ),
}
