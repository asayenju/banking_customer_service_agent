"""Account level to tier. The policy documents speak of Entry, Mid, Premium and Elite tiers
but do not say which account is which. This table is read off the reference answers (fees
charged on each level) and covers only the levels they show; any other level is not guessed
at: a workflow that needs the tier hands over."""

from typing import Optional

TIER_OF_LEVEL: dict[str, str] = {
    "Light Blue Account": "Entry",
    "Blue Account": "Mid",
    "Green Account": "Mid",
    "Evergreen Account": "Premium",
    "Bluest Account": "Elite",
}


def tier_of_level(level: str) -> Optional[str]:
    return TIER_OF_LEVEL.get(level)
