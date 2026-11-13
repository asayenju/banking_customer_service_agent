"""The PIN lock investigation (doc 041): a fraud risk score worked out from the records, in code.

The document is a procedure for a card with ``pin_locked`` true. Everything it measures is in the card, the
declined transactions and the account, so the model never does the arithmetic and never sees the score: it
reports what the customer says to the questions code asks. The score is never put in a reply (037, 041).
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

STAMP = "%m/%d/%Y %H:%M:%S"


@dataclass(frozen=True)
class Score:
    flags: Mapping[str, int]
    total: int
    level: str
    escalation: Optional[str] = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def _stamp(text: Any) -> Optional[datetime]:
    for fmt in (STAMP, "%m/%d/%Y"):
        try:
            return datetime.strptime(str(text), fmt)
        except ValueError:
            continue
    return None


def _place(description: str) -> tuple[str, str]:
    """City and state (or country) from 'ATM WITHDRAWAL DECLINED - CHASE ATM LAS VEGAS NV'."""
    tail = description.split(" - ", 1)[-1].strip()
    words = tail.split()
    if len(words) >= 2 and re.fullmatch(r"[A-Z]{2}", words[-1]):
        return " ".join(words[-3:-1]) if len(words) >= 3 else words[-2], words[-1]
    return " ".join(words[-2:]), words[-1] if words else ""


def declines(db: Any, account_id: str) -> list[Mapping[str, Any]]:
    rows = [
        t
        for t in db.bank_account_transaction_history.data.values()
        if t.get("account_id") == account_id and t.get("type") in ("atm_withdrawal_declined", "pos_declined")
    ]
    return sorted(rows, key=lambda t: _stamp(t.get("date")) or datetime.min)


US_STATES = frozenset(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split()
)
_ADDRESS = re.compile(r",\s*([A-Za-z .'-]+),\s*([A-Z]{2})\s+\d{5}")


def home_place(address: str) -> tuple[str, str]:
    match = _ADDRESS.search(address or "")
    return (match.group(1).strip().upper(), match.group(2)) if match else ("", "")


def _bands(value: float, bands: Sequence[tuple[float, int]], last: int) -> int:
    for upper, points in bands:
        if value < upper:
            return points
    return last


def level_of(total: int) -> str:
    if total <= 4:
        return "LOW"
    if total <= 7:
        return "MEDIUM"
    if total <= 10:
        return "HIGH"
    if total <= 14:
        return "VERY_HIGH"
    return "CRITICAL"


def score_card(db: Any, card: Mapping[str, Any], today: date, removed: frozenset[str] = frozenset()) -> Score:
    """The doc 041 flags for one locked card. ``removed`` holds flag groups the customer's answers cleared
    ("location", "amount", "time"): the document says to drop them and recalculate."""
    account = db.accounts.data[str(card["account_id"])]
    user = db.users.data[str(card["user_id"])]
    rows = declines(db, str(card["account_id"]))
    home_city, home_state = home_place(str(user.get("address")))
    places = [_place(str(t.get("description"))) for t in rows]
    flags: dict[str, int] = {}

    def at(key: str, points: int, group: str) -> None:
        flags[key] = 0 if group in removed else points

    # A: location
    worst = 0
    for city, state in places:
        if home_city and home_city in str(rows[places.index((city, state))].get("description")).upper():
            continue
        worst = max(worst, 1 if state == home_state else 2 if state in US_STATES else 3)
    at("A1", worst, "location")
    distinct = len(set(places))
    at("A2", 0 if distinct <= 1 else 1 if distinct == 2 else 2, "location")
    flags["A3"] = 0
    # B: time
    hours = [(_stamp(t.get("date")) or datetime.min).hour for t in rows]
    at("B1", max((0 if 6 <= h < 22 else 1 if h >= 22 else 2 if h < 2 else 3 for h in hours), default=0), "time")
    last_ok = _stamp(card.get("last_successful_pin_transaction_date"))
    locked = _stamp(card.get("pin_lock_timestamp")) or datetime.combine(today, datetime.min.time())
    since = (locked - last_ok).days if last_ok else 0
    flags["B2"] = 0 if since < 7 else 1 if since <= 30 else 2
    times = [_stamp(t.get("date")) for t in rows]
    gaps = [(b - a).total_seconds() / 60 for a, b in zip(times, times[1:]) if a and b]
    flags["B3"] = 0 if not gaps else _bands(min(gaps), ((1, 3), (2, 2), (5, 1)), 0)
    # C: amount
    amounts = [abs(float(t.get("amount") or 0)) for t in rows]
    decreasing = len(amounts) >= 2 and all(a > b for a, b in zip(amounts, amounts[1:]))
    at("C1", 2 if decreasing else 0, "amount")
    flags["C2"] = 1 if amounts and all(a % 100 == 0 for a in amounts) else 0
    ok_atm = [
        abs(float(t.get("amount") or 0))
        for t in db.bank_account_transaction_history.data.values()
        if t.get("account_id") == card.get("account_id") and t.get("type") == "atm_withdrawal"
    ]
    average = sum(ok_atm) / len(ok_atm) if ok_atm else 0.0
    biggest = max(amounts, default=0.0)
    flags["C3"] = 0 if not average or biggest <= 2 * average else 1 if biggest <= 5 * average else 2
    limit = float(card.get("daily_atm_limit") or 0)
    atm = [abs(float(t.get("amount") or 0)) for t in rows if t.get("type") == "atm_withdrawal_declined"]
    if not atm or not limit:
        flags["C4"] = 0
    elif sum(atm) > limit and len(atm) > 1:
        flags["C4"] = 2
    else:
        ratio = max(atm) / limit
        flags["C4"] = 1 if ratio >= 0.8 else 0
    # D: card history
    flags["D1"] = min(int(card.get("pin_locks_last_90_days") or 0), 3)
    issued = _stamp(card.get("issue_date"))
    months = ((today - issued.date()).days / 30.4) if issued else 99
    flags["D2"] = 2 if months < 1 else 1 if months < 3 else 0
    others = [
        c
        for c in db.debit_cards.data.values()
        if c.get("user_id") == card.get("user_id") and c.get("card_id") != card.get("card_id")
    ]
    flags["D3"] = (1 if any(c.get("velocity_blocked") for c in others) else 0) + (
        2 if any(c.get("fraud_alert_active") for c in others) else 0
    )
    # E: account
    opened = _stamp(account.get("date_opened"))
    age_months = ((today - opened.date()).days / 30.4) if opened else 99
    flags["E1"] = 2 if age_months < 3 else 1 if age_months < 6 else 0
    overdrafts = sum(
        1
        for t in db.bank_account_transaction_history.data.values()
        if t.get("account_id") == card.get("account_id") and "OVERDRAFT" in str(t.get("description", "")).upper()
    )
    flags["E2"] = min(overdrafts, 2)
    try:
        balance = float(str(account.get("current_holdings") or "0").replace("$", "").replace(",", ""))
    except ValueError:
        balance = 0.0
    flags["E3"] = 2 if balance < 50 else 1 if balance < 100 else 0
    total = sum(flags.values())
    escalation = None
    if card.get("pin_lock_reason") == "security_hold":
        escalation = "security_hold"
    elif any(v >= 3 for v in flags.values()):
        escalation = "single_flag"
    return Score(flags, total, level_of(total), escalation)
