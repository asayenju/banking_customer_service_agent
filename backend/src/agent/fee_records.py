"""Reading the records a fee or interest review needs: an account's transactions, which withdrawals were
foreign or at the bank's own ATMs, the customer's credit cards, and a savings or checking level as a
table key. Settled in code, never by a model; another customer's account never resolves (the account
itself comes from ``agent/bank_accounts.py``).

A withdrawal's description ends with a place ("... CHICAGO IL", "... LONDON UK"). Some two-letter
endings are both a US state and a country (CA, DE, IN, ...), so those are decided by the city and,
when the city is not one we know, the withdrawal is ``None`` (unknown): the caller hands off rather
than guess.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from agent.cards import owned_credit_accounts
from agent.specialists.credit_rules import day, money
from agent.workflows.fee_products import ATM_FEES, CARD_NAMES, CHECKING_BOOST, SAVINGS_BASE

US_STATES = frozenset(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK "
    "OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split()
)
FOREIGN_CODES = frozenset(
    "UK GB KR JP CN HK SG MY TH IN ID PH VN AU NZ FR DE IT ES PT NL BE CH AT SE NO DK FI IE PL GR TR MX BR AR CL CO PE "
    "CA ZA EG AE SA IL".split()
)
# Codes that are both: only these cities settle them.
FOREIGN_CITIES = {
    "DE": {"FRANKFURT", "BERLIN", "MUNICH", "HAMBURG", "COLOGNE", "DUSSELDORF"},
    "CA": {"TORONTO", "VANCOUVER", "MONTREAL", "OTTAWA", "CALGARY"},
    "IN": {"MUMBAI", "DELHI", "BANGALORE", "CHENNAI", "KOLKATA"},
    "CO": {"BOGOTA", "MEDELLIN", "CARTAGENA"},
    "AR": {"BUENOS AIRES"},
    "PE": {"LIMA", "CUSCO"},
    "IL": {"TEL AVIV", "JERUSALEM"},
    "ID": {"JAKARTA", "BALI"},
}
# For these codes a US city is the usual reading; an unrecognised city is unknown, not foreign.
_BOTH = FOREIGN_CODES & US_STATES


@dataclass(frozen=True)
class Txn:
    transaction_id: str
    account_id: str
    when: Optional[date]
    description: str
    amount: float
    kind: str  # "atm_withdrawal", "atm_fee", "fee_rebate", "fee_refund", "rebate_credit", ...


def transactions_of(db: Any, account_id: str) -> list[Txn]:
    """Posted transactions of one account, oldest first (then by id), so counting free withdrawals is stable."""
    rows = []
    for row in db.bank_account_transaction_history.data.values():
        if not isinstance(row, Mapping) or row.get("account_id") != account_id:
            continue
        if str(row.get("status") or "posted").lower() != "posted":
            continue
        amount = money(row.get("amount"))
        if amount is None:
            continue
        rows.append(
            Txn(
                transaction_id=str(row.get("transaction_id") or ""),
                account_id=account_id,
                when=day(row.get("date")),
                description=str(row.get("description") or ""),
                amount=amount,
                kind=str(row.get("type") or ""),
            )
        )
    return sorted(rows, key=lambda t: (t.when or date.min, t.transaction_id))


def is_rho_atm(description: str) -> bool:
    """A withdrawal at the bank's own ATM, which never carries a fee."""
    return "RHO-BANK" in description.upper() or "RHO BANK" in description.upper()


def is_foreign(description: str) -> Optional[bool]:
    """True for a withdrawal abroad, False at home, None when the ending cannot tell."""
    text = description.strip().upper()
    code = text[-2:] if len(text) >= 2 and text[-3:-2] == " " else ""
    if not code.isalpha():
        return None
    if code in _BOTH:
        city_part = text[: -3].rsplit(" - ", 1)[-1]
        if any(city in city_part for city in FOREIGN_CITIES.get(code, ())):
            return True
        # A US city for an overlapping code: home. Unknown cities for a code that is only ever a country
        # (not a state) are decided below; for an overlapping one we cannot tell.
        return False if _looks_us(city_part) else None
    if code in US_STATES:
        return False
    if code in FOREIGN_CODES:
        return True
    return None


def _looks_us(city_part: str) -> bool:
    # Descriptions at US ATMs carry a branch number ("#4521") or a well-known US bank; an overseas one
    # names a city only. This is a hint for the overlapping codes, never for a plain state code.
    return "#" in city_part


def card_key(card_type: Any) -> Optional[str]:
    """A credit card's type as the table key ("Crypto-Cash Back" and "Crypto-Cash Back Card" are one card)."""
    text = str(card_type or "").strip()
    for name in CARD_NAMES:
        if text.lower() == name.lower():
            return name
    if text.lower() == "crypto-cash back":
        return "Crypto-Cash Back Card"
    return None


def cards_held(db: Any, user_id: str) -> tuple[list[str], list[str]]:
    """The customer's open credit cards as table keys, and the card types we do not know."""
    known: list[str] = []
    unknown: list[str] = []
    for account in owned_credit_accounts(db, user_id):
        key = card_key(account.get("card_type"))
        (known if key else unknown).append(key or str(account.get("card_type")))
    return known, unknown


def level_key(name: str, table: Mapping[str, Any]) -> Optional[str]:
    """An account's level as a table key, or None when the level is not one we have a table for."""
    for key in table:
        if key.lower() == name.strip().lower():
            return key
    return None


def has_fee_schedule(level: str) -> bool:
    return level_key(level, ATM_FEES) is not None


def has_savings_table(level: str) -> bool:
    return level_key(level, SAVINGS_BASE) is not None


CHECKING_LEVELS = frozenset(ATM_FEES) | {checking for checking, _ in CHECKING_BOOST}


def boost_for(checking_level: str, savings_level: str) -> Optional[float]:
    """The pairing's boost, 0.0 for a combination that does not qualify, None if either level is unknown."""
    checking = level_key(checking_level, dict.fromkeys(CHECKING_LEVELS))
    savings = level_key(savings_level, SAVINGS_BASE)
    if checking is None or savings is None:
        return None
    pair = CHECKING_BOOST.get((checking, savings))
    return pair[0] if pair else 0.0
