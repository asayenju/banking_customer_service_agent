"""The account products as data: the exact class strings, the caps and the closing tiers.

Source: ``doc_bank_accounts_bank_accounts_(general)_001`` to ``_008`` and the product pages
(``doc_checking_accounts_*``, ``doc_savings_accounts_*``, ``doc_business_checking_accounts_*``,
``doc_business_savings_accounts_*``). Decisions and the places the documents are silent are in
``docs/accounts-policy-decisions.md`` (sections 1 and 8). Pure data, no logic, so the rules module
and the later catalogue read the same table.
"""

from dataclasses import dataclass
from typing import Optional

from agent.bank_accounts import Product

# The exact class string for each product: what ``open_bank_account_4821`` is given as ``account_class``.
# Personal products and business savings end in "Account"; business checking is the bare name
# (documents 001, 002 and 004 say "ending with Account", document 003 and every gold open say otherwise
# for business checking).
CLASSES: dict[Product, tuple[str, ...]] = {
    Product.CHECKING: (
        "Blue Account", "Bluest Account", "Dark Green Account", "Evergreen Account", "Gold Years Account",
        "Green Account", "Green Fee-Free Account", "Light Blue Account", "Light Green Account", "Purple Account",
    ),
    Product.SAVINGS: (
        "Bronze Account", "Diamond Elite Account", "Gold Account", "Gold Plus Account", "Green Account",
        "Platinum Account", "Platinum Plus Account", "Silver Account", "Silver Plus Account",
    ),
    Product.BUSINESS_CHECKING: (
        "Beige", "Cobalt Blue", "Hunter Green", "Lime Green", "Navy Blue", "Sky Blue", "True Blue", "World Blue",
    ),
    Product.BUSINESS_SAVINGS: (
        "Bronze Saver Account", "Diamond Vault Account", "Emerald Saver Account", "Gold Plus Saver Account",
        "Gold Saver Account", "Platinum Reserve Account", "Silver Plus Saver Account", "Silver Saver Account",
    ),
}

# The most accounts of a product a customer may hold: opening is allowed while they hold fewer than this
# (documents 001 to 004).
CAP: dict[Product, int] = {
    Product.CHECKING: 4,
    Product.SAVINGS: 5,
    Product.BUSINESS_CHECKING: 6,
    Product.BUSINESS_SAVINGS: 4,
}
MIN_CHECKING_AGE_FOR_SAVINGS_DAYS = 14
MIN_BUSINESS_CHECKING_AGE_DAYS = 30
MIN_PERSONAL_CHECKING_BALANCE_FOR_BUSINESS = 500.0
MIN_BUSINESS_CHECKING_BALANCE_FOR_SAVINGS = 2500.0
MIN_AGE_FOR_PERSONAL_CHECKING = 18


@dataclass(frozen=True)
class Closing:
    fee: int  # dollars, owed when the account is closed inside the window
    window_days: int
    notice_days: int  # listed by the documents; not enforced (docs/accounts-policy-decisions.md, section 7)
    approval: str = ""  # "supervisor" or "manager": a person must approve, so the agent hands off


def _tier(names: tuple[str, ...], fee: int, window: int, notice: int, approval: str = "") -> dict[str, Closing]:
    return {name: Closing(fee, window, notice, approval) for name in names}


# Documents 005 to 008. A name that is not here has no closing rules on file.
CLOSING: dict[Product, dict[str, Closing]] = {
    Product.CHECKING: {
        **_tier(("Light Blue Account", "Light Green Account", "Green Fee-Free Account"), 15, 30, 0),
        **_tier(("Blue Account", "Green Account"), 25, 60, 3),
        **_tier(("Evergreen Account",), 50, 90, 7),
        **_tier(("Bluest Account",), 100, 180, 14),
    },
    Product.SAVINGS: {
        **_tier(("Bronze Account",), 20, 60, 1),
        **_tier(("Silver Account", "Silver Plus Account"), 35, 90, 5),
        **_tier(("Gold Account", "Gold Plus Account", "Gold Years Account"), 75, 180, 10),
        **_tier(("Platinum Account", "Platinum Plus Account", "Diamond Elite Account"), 150, 270, 21, "manager"),
    },
    Product.BUSINESS_CHECKING: {
        **_tier(("Navy Blue",), 50, 60, 7),
        **_tier(("True Blue", "Sky Blue"), 100, 90, 14),
        **_tier(("Cobalt Blue",), 200, 180, 21, "supervisor"),
    },
    Product.BUSINESS_SAVINGS: {
        **_tier(("Bronze Saver Account",), 75, 90, 10),
        **_tier(("Silver Saver Account",), 125, 120, 14),
        **_tier(("Gold Saver Account",), 250, 180, 21, "supervisor"),
        **_tier(("Platinum Reserve Account",), 500, 270, 30, "manager"),
    },
}


def closing_rules(product: Product, name: str) -> Optional[Closing]:
    return CLOSING.get(product, {}).get(name)


def is_known_class(product: Product, name: str) -> bool:
    return name in CLASSES.get(product, ())


# ---------------------------------------------------------------- the catalogue: pages, ages, pairings, promotions

# The product page for each class: the prefix of its document ids (``<prefix>_001`` and so on). A test checks
# that every prefix is a real document, so a candidate always cites something that exists.
PAGE: dict[tuple[Product, str], str] = {
    (Product.CHECKING, "Blue Account"): "doc_checking_accounts_blue_account",
    (Product.CHECKING, "Bluest Account"): "doc_checking_accounts_bluest_account",
    (Product.CHECKING, "Dark Green Account"): "doc_checking_accounts_dark_green_account",
    (Product.CHECKING, "Evergreen Account"): "doc_checking_accounts_evergreen_account",
    (Product.CHECKING, "Gold Years Account"): "doc_checking_accounts_gold_years_account",
    (Product.CHECKING, "Green Account"): "doc_checking_accounts_green_account_(checking)",
    (Product.CHECKING, "Green Fee-Free Account"): "doc_checking_accounts_green_fee-free_account",
    (Product.CHECKING, "Light Blue Account"): "doc_checking_accounts_light_blue_account",
    (Product.CHECKING, "Light Green Account"): "doc_checking_accounts_light_green_account",
    (Product.CHECKING, "Purple Account"): "doc_checking_accounts_purple_account",
    (Product.SAVINGS, "Bronze Account"): "doc_savings_accounts_bronze_account",
    (Product.SAVINGS, "Diamond Elite Account"): "doc_savings_accounts_diamond_elite_account",
    (Product.SAVINGS, "Gold Account"): "doc_savings_accounts_gold_account",
    (Product.SAVINGS, "Gold Plus Account"): "doc_savings_accounts_gold_plus_account",
    (Product.SAVINGS, "Green Account"): "doc_savings_accounts_green_account_(savings)",
    (Product.SAVINGS, "Platinum Account"): "doc_savings_accounts_platinum_account",
    (Product.SAVINGS, "Platinum Plus Account"): "doc_savings_accounts_platinum_plus_account",
    (Product.SAVINGS, "Silver Account"): "doc_savings_accounts_silver_account",
    (Product.SAVINGS, "Silver Plus Account"): "doc_savings_accounts_silver_plus_account",
    (Product.BUSINESS_CHECKING, "Beige"): "doc_business_checking_accounts_beige",
    (Product.BUSINESS_CHECKING, "Cobalt Blue"): "doc_business_checking_accounts_cobalt_blue",
    (Product.BUSINESS_CHECKING, "Hunter Green"): "doc_business_checking_accounts_hunter_green",
    (Product.BUSINESS_CHECKING, "Lime Green"): "doc_business_checking_accounts_lime_green",
    (Product.BUSINESS_CHECKING, "Navy Blue"): "doc_business_checking_accounts_navy_blue",
    (Product.BUSINESS_CHECKING, "Sky Blue"): "doc_business_checking_accounts_sky_blue",
    (Product.BUSINESS_CHECKING, "True Blue"): "doc_business_checking_accounts_true_blue",
    (Product.BUSINESS_CHECKING, "World Blue"): "doc_business_checking_accounts_world_blue",
    (Product.BUSINESS_SAVINGS, "Bronze Saver Account"): "doc_business_savings_accounts_bronze_saver_account",
    (Product.BUSINESS_SAVINGS, "Diamond Vault Account"): "doc_business_savings_accounts_diamond_vault",
    (Product.BUSINESS_SAVINGS, "Emerald Saver Account"): "doc_business_savings_accounts_emerald_saver",
    (Product.BUSINESS_SAVINGS, "Gold Plus Saver Account"): "doc_business_savings_accounts_gold_plus_saver",
    (Product.BUSINESS_SAVINGS, "Gold Saver Account"): "doc_business_savings_accounts_gold_saver_account",
    (Product.BUSINESS_SAVINGS, "Platinum Reserve Account"): "doc_business_savings_accounts_platinum_reserve_account",
    (Product.BUSINESS_SAVINGS, "Silver Plus Saver Account"): "doc_business_savings_accounts_silver_plus_saver",
    (Product.BUSINESS_SAVINGS, "Silver Saver Account"): "doc_business_savings_accounts_silver_saver_account",
}

# Holder age limits stated on a product page: (youngest, oldest), None for no limit on that side.
# The only ones the pages state as numbers (Gold Years: "at least 62"; Dark Green: 17 through 26).
AGE_LIMITS: dict[tuple[Product, str], tuple[Optional[int], Optional[int]]] = {
    (Product.CHECKING, "Gold Years Account"): (62, None),
    (Product.CHECKING, "Dark Green Account"): (17, 26),
}

# Document 012: a checking and a savings account held together raise the savings APY.
APY_PAIRINGS: tuple[tuple[str, str], ...] = (
    ("Green Account", "Gold Account"), ("Green Fee-Free Account", "Bronze Account"),
    ("Evergreen Account", "Green Account"), ("Blue Account", "Silver Plus Account"),
    ("Light Green Account", "Diamond Elite Account"), ("Dark Green Account", "Bronze Account"),
    ("Bluest Account", "Silver Account"), ("Purple Account", "Platinum Plus Account"),
    ("Gold Years Account", "Gold Plus Account"), ("Light Green Account", "Platinum Account"),
    ("Green Account", "Silver Account"), ("Evergreen Account", "Diamond Elite Account"),
    ("Bluest Account", "Bronze Account"), ("Purple Account", "Gold Account"),
    ("Gold Years Account", "Silver Account"), ("Blue Account", "Platinum Account"),
    ("Green Fee-Free Account", "Gold Plus Account"), ("Dark Green Account", "Platinum Plus Account"),
)


@dataclass(frozen=True)
class Promotion:
    """Documents 013 to 016: between two dates, when several accounts meet a customer's needs, recommend
    these first, in this order."""

    document: str
    label: str
    product: Product
    start: str  # MM/DD/YYYY, inclusive
    end: str  # MM/DD/YYYY, inclusive
    priority: tuple[str, ...]


PROMOTIONS: tuple[Promotion, ...] = (
    Promotion("doc_bank_accounts_bank_accounts_(general)_014", "October 2025 business checking promotion", Product.BUSINESS_CHECKING, "10/12/2025", "11/12/2025", ("Lime Green", "Hunter Green")),
    Promotion("doc_bank_accounts_bank_accounts_(general)_013", "November 2025 business checking promotion", Product.BUSINESS_CHECKING, "11/01/2025", "11/30/2025", ("Sky Blue", "Lime Green")),
    Promotion("doc_bank_accounts_bank_accounts_(general)_016", "October 2025 business savings promotion", Product.BUSINESS_SAVINGS, "10/12/2025", "11/12/2025", ("Gold Plus Saver Account", "Silver Plus Saver Account")),
    Promotion("doc_bank_accounts_bank_accounts_(general)_015", "November 2025 business savings promotion", Product.BUSINESS_SAVINGS, "11/01/2025", "11/30/2025", ("Gold Saver Account", "Silver Plus Saver Account")),
)
