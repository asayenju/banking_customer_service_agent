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
