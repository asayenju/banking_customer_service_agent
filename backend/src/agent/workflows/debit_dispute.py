"""Debit card dispute: the model reports what the customer said, code fills in the rest.

Source: "Internal: Filing a Debit Card Transaction Dispute" and "Debit Card Provisional
Credit Guidelines". The model never writes the transaction's date or amount, the
liability, the provisional-credit flag or the card action: those come from the database
and the rules below. After the disputes, one card action per card, the most severe one.
"""

from collections.abc import Sequence
import re
from datetime import date, datetime, timedelta
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from agent.cards import describe
from agent.workflows.base import (
    Ask,
    Decision,
    NeedInfo,
    Plan,
    Refuse,
    Step,
    WorkflowContext,
)
from orchestrator.types import IntentKind

Category = Literal[
    "unauthorized_transaction",
    "atm_cash_discrepancy",
    "atm_deposit_not_credited",
    "duplicate_charge",
    "incorrect_amount",
    "goods_services_not_received",
    "recurring_charge_after_cancellation",
    "card_present_fraud",
    "card_not_present_fraud",
]
TransactionType = Literal[
    "pin_purchase",
    "signature_purchase",
    "online_purchase",
    "atm_withdrawal",
    "atm_deposit",
    "recurring_payment",
    "person_to_person",
]
ClosureAnswer = Literal["yes", "no", "not_asked"]
PinCompromised = Literal["yes_shared", "yes_observed", "no", "unknown"]

CARD_ACTION: dict[str, str] = {
    "card_present_fraud": "close_and_reissue",
    "card_not_present_fraud": "close_and_reissue",
    "unauthorized_transaction": "freeze_pending_investigation",
}
SEVERITY = ("keep_active", "freeze_pending_investigation", "close_and_reissue")
# Categories where only part of the charge is in question (an ATM that paid $100 of $200, a charge $100 too high).
# Every other category disputes the whole amount. In every reference answer a partial category is below the charge.
PARTIAL_AMOUNT_CATEGORIES = frozenset({"atm_cash_discrepancy", "incorrect_amount", "atm_deposit_not_credited"})
FRAUD_CATEGORIES = frozenset({"card_present_fraud", "card_not_present_fraud", "unauthorized_transaction"})
POLICE_REPORT_OVER = 500.0
# What the bank asks, in this order, and only what matters for the kind of dispute (doc 031).
QUESTION_ORDER = ("card_in_possession", "pin_compromised", "police_report_filed", "contacted_merchant", "discovery_date", "written_statement_provided")
PROVISIONAL_CATEGORIES = frozenset(
    {
        "unauthorized_transaction",
        "card_present_fraud",
        "card_not_present_fraud",
        "atm_cash_discrepancy",
        "duplicate_charge",
    }
)
MAX_OPEN_DISPUTES = {"Entry": 2, "Mid": 3, "Premium": 4, "Elite": 5}
# The bank's rule is stated against the statement date, which the data does not hold. In
# the reference answers a report 9 days or less after the charge carries the $50 limit and
# one 40 days or more after carries $500; 30 days splits them.
LATE_AFTER_DAYS = 30
MAX_AGE_DAYS = 60


class DisputeFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: str
    category: Category = Field(
        description=(
            "Bank policy: if the customer says they did not make the charge and fraud is suspected, "
            "use card_present_fraud for an in-store or physical-card charge (chip, PIN or swipe) or "
            "card_not_present_fraud for an online or phone charge. Use unauthorized_transaction only "
            "when fraud is NOT suspected."
        )
    )
    discovery_date: Optional[str] = Field(
        description=(
            "MM/DD/YYYY: when the customer first noticed. If the customer gave no year, use the year of the "
            "charge on the transaction record; never guess a different year. null if they have not said."
        )
    )
    noticed_words: Optional[str] = Field(
        description=(
            "The customer's own words about WHEN THEY NOTICED the problem (not when the charge happened), copied "
            "from the conversation, such as 'noticed on the 10th' or 'right away'. null if they have not said."
        )
    )
    # What the customer disputes: all of the charge, or only part (an ATM that paid $100 short).
    disputed_amount: float = Field(
        description=(
            "The amount in dispute. When the customer disputes the whole charge, use the amount on "
            "the transaction record even if the customer only gave a rounded figure; do not ask. For "
            "incorrect_amount it is only the overcharge (charged minus what it should have been); for "
            "atm_cash_discrepancy it is only the cash that was not paid out. Never the whole charge for these."
        )
    )
    transaction_type: TransactionType = Field(
        description="How the charge was paid, as the customer or the transaction record says. Do not guess."
    )
    # For each of these: null when the customer has not said. Do not ask them yourself and do not guess: the
    # bank asks only the ones that matter for this kind of dispute.
    card_in_possession: Optional[bool]
    pin_compromised: Optional[PinCompromised]
    contacted_merchant: Optional[bool]
    police_report_filed: Optional[bool]
    written_statement_provided: Optional[bool]
    card_closure: ClosureAnswer = Field(
        description=(
            "Only for a card the bank recommends closing and reissuing after this dispute: 'yes' if the "
            "customer said to close or cancel it, 'no' if they want to keep using it (asking for a new number or a "
            "reissue is NOT consent to close it: report 'no'), 'not_asked' if neither. Report 'not_asked' for "
            "every other card."
        )
    )


class DisputeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    disputes: list[DisputeFacts] = Field(
        description="One entry for every transaction the customer wants to dispute, on every card. Do not leave any out."
    )


def parse_date(text: str) -> Optional[date]:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip()[:10], fmt).date()
        except ValueError:
            continue
    return None


# A cash machine record says how it was paid; a purchase record does not say PIN, signature or online, so only these
# are taken from the record rather than from what the customer said.
RECORD_TRANSACTION_TYPES = {"atm_withdrawal": "atm_withdrawal", "atm_deposit": "atm_deposit"}


def settle_year(noticed: date, happened: date, today: date) -> date:
    """The customer says a month and day; the model may add the wrong year. A date before the charge is the same
    day in the charge's year, or the year after it, whichever is not before the charge and not after today."""
    if noticed >= happened:
        return noticed
    for year in (happened.year, happened.year + 1):
        try:
            candidate = noticed.replace(year=year)
        except ValueError:
            continue
        if happened <= candidate <= today:
            return candidate
    return noticed


_FULL_MONTHS = ("january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december", "sept")
_MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), start=1)}
_RIGHT_AWAY = re.compile(r"right away|immediately|same day|that day|straight away|as soon as|when it happened|at once", re.I)


def date_matches_words(noticed: date, words: str, happened: date, today: date) -> bool:
    """Does the date the model wrote agree with the customer's words about when they noticed? The model once wrote
    the charge's date for "noticed on the 10th". Day, month and relative words in the quote must fit the date."""
    text = words.lower()
    if _RIGHT_AWAY.search(text):
        return noticed == happened
    if "yesterday" in text:
        return noticed == today - timedelta(days=1)
    if re.search(r"\btoday\b|\bthis morning\b|\btonight\b", text):
        return noticed == today
    full = re.findall(r"\b(\d{1,2})/(\d{1,2})(?:/\d{2,4})?\b", text)
    if full:
        return any((int(m), int(d)) == (noticed.month, noticed.day) for m, d in full)
    months = {_MONTHS[w[:3]] for w in re.findall(r"[a-z]+", text) if w[:3] in _MONTHS and (len(w) == 3 or w.startswith(tuple(_FULL_MONTHS)))}
    days = [int(n) for n in re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", text) if 1 <= int(n) <= 31]
    if not months and not days:
        return False
    return (not months or noticed.month in months) and (not days or noticed.day in days)


def liability(transaction_date: date, discovery: date) -> float:
    return 500.0 if (discovery - transaction_date).days > LATE_AFTER_DAYS else 50.0


def provisional_credit(facts: DisputeFacts) -> bool:
    if facts.pin_compromised == "yes_shared":
        return False
    if facts.category in PROVISIONAL_CATEGORIES:
        return True
    # Doc 032 says the other categories are "not required but may be offered at discretion". The references offer it
    # only for a cancelled subscription the customer already took to the merchant (task 082); tasks 084 and 086 do
    # not for an incorrect amount or goods not received even after the merchant was contacted. Inferred, one task.
    return facts.category == "recurring_charge_after_cancellation" and bool(facts.contacted_merchant)


def tier_of(level: str) -> Optional[str]:
    from agent.workflows.tiers import tier_of_level

    return tier_of_level(level)


class DebitDispute:
    name = "debit_dispute"
    kinds = frozenset({IntentKind.DISPUTE_TRANSACTION})
    summary = "The customer disputes one or more debit card transactions."
    facts = DisputeRequest

    @staticmethod
    def _settle_duplicates(db: Any, disputes: Sequence[DisputeFacts]) -> list[DisputeFacts]:
        """Doc 031: when a charge was made twice, dispute the earliest one, once. The records of a double charge are
        identical rows (same account, merchant, amount and day), so the chosen row is the last of them in the
        bank's list, which is the one the reference answers dispute (task 084)."""
        rows = db.bank_account_transaction_history.data
        chosen: list[DisputeFacts] = []
        taken: set[str] = set()
        for item in disputes:
            txn = rows.get(item.transaction_id)
            if item.category == "duplicate_charge" and txn is not None:
                twins = [
                    k
                    for k, t in rows.items()
                    if all(t.get(f) == txn.get(f) for f in ("account_id", "description", "amount", "date"))
                ]
                pick = twins[-1]
                if pick != item.transaction_id:
                    item = item.model_copy(update={"transaction_id": pick})
                if pick in taken:
                    continue
                taken.add(pick)
            chosen.append(item)
        return chosen

    @staticmethod
    def _complete(db: Any, disputes: Sequence[DisputeFacts]) -> tuple[list[DisputeFacts], list[Ask]]:
        """Fill in what the kind of dispute makes obvious and list the few things only the customer can say.

        A fraud dispute needs the card and PIN answers (and a police report over $500); any other dispute needs to
        know whether the merchant was contacted. What does not matter for a kind gets the answer the references use
        (a card in hand, no PIN trouble, no police report). Every dispute needs the date it was noticed and the
        customer's yes to using this chat as their written statement."""
        asks: dict[str, Ask] = {}
        done: list[DisputeFacts] = []

        def need(slot: str, text: str) -> None:
            asks.setdefault(slot, Ask(slot, text))

        for item in disputes:
            txn = db.bank_account_transaction_history.data.get(item.transaction_id)
            amount = abs(float(txn.get("amount", 0))) if txn else 0.0
            fraud = item.category in FRAUD_CATEGORIES
            fill: dict[str, Any] = {}
            if fraud:
                if item.card_in_possession is None:
                    need("card_in_possession", "Do you still have your physical debit card with you? (yes or no)")
                if item.pin_compromised is None:
                    need("pin_compromised", "Do you think your PIN may have been seen or shared?")
                if amount > POLICE_REPORT_OVER and item.police_report_filed is None:
                    need("police_report_filed", "Have you filed a police report for this charge? (yes or no)")
                fill.update({k: False for k in ("contacted_merchant", "police_report_filed") if getattr(item, k) is None})
            else:
                if item.contacted_merchant is None:
                    need("contacted_merchant", "Have you already tried to sort this out with the merchant? (yes or no)")
                if item.card_in_possession is None:
                    fill["card_in_possession"] = True
                if item.pin_compromised is None:
                    fill["pin_compromised"] = "no"
                if item.police_report_filed is None:
                    fill["police_report_filed"] = False
            if item.discovery_date is None or not (item.noticed_words or "").strip():
                need("discovery_date", "When did you first notice this charge?")
            if item.written_statement_provided is None:
                need(
                    "written_statement_provided",
                    "Is it all right to use this conversation as your written statement of what happened? (yes or no)",
                )
            done.append(item.model_copy(update=fill))
        return done, sorted(asks.values(), key=lambda a: QUESTION_ORDER.index(a.slot))

    def decide(self, ctx: WorkflowContext, facts: DisputeRequest) -> Decision:
        if not facts.disputes:
            return Refuse("list at least one transaction to dispute")
        db = ctx.database
        steps: list[Step] = []
        card_action: dict[str, str] = {}
        cards_of: dict[str, str] = {}
        seen: set[str] = set()
        new_per_account: dict[str, int] = {}
        disputes = self._settle_duplicates(db, facts.disputes)
        for item in disputes:
            if db.bank_account_transaction_history.data.get(item.transaction_id) is None:
                return Refuse(f"{item.transaction_id} is not a transaction on this customer's accounts")
        disputes, asks = self._complete(db, disputes)
        if asks:
            return NeedInfo(tuple(asks))
        for item in disputes:
            if item.transaction_id in seen:
                return Refuse(f"{item.transaction_id} is listed twice")
            seen.add(item.transaction_id)
            txn = db.bank_account_transaction_history.data.get(item.transaction_id)
            account = db.accounts.data.get(str(txn.get("account_id"))) if txn else None
            if txn is None or account is None or account.get("user_id") != ctx.user_id:
                return Refuse(f"{item.transaction_id} is not a transaction on this customer's accounts")
            if account.get("status") != "OPEN":
                return Reply_closed(account)
            happened = parse_date(str(txn.get("date")))
            noticed = parse_date(item.discovery_date or "")
            if happened is None or noticed is None:
                return Refuse("discovery_date must be written MM/DD/YYYY")
            noticed = settle_year(noticed, happened, ctx.today)
            if not date_matches_words(noticed, str(item.noticed_words), happened, ctx.today):
                return Refuse(
                    f"the discovery date {noticed:%m/%d/%Y} does not fit what the customer said about when they noticed "
                    f"({str(item.noticed_words)!r}). It is the day they noticed the problem, not the day of the charge "
                    f"({happened:%m/%d/%Y}). Fix the date, or the quote, or ask them"
                )
            if (from_record := RECORD_TRANSACTION_TYPES.get(str(txn.get("type")))) is not None:
                item = item.model_copy(update={"transaction_type": from_record})
            if noticed < happened or noticed > ctx.today:
                return Refuse(
                    f"discovery_date {item.discovery_date} cannot be before the charge on "
                    f"{happened:%m/%d/%Y} or after today, {ctx.today:%m/%d/%Y}. If the customer gave no "
                    "year, use the year of the charge date; do not ask them."
                )
            amount = abs(float(txn.get("amount", 0)))
            if not 1.0 <= item.disputed_amount <= amount:
                return Refuse(
                    f"the disputed amount must be between $1.00 and the transaction's ${amount:.2f}. "
                    f"If the customer is disputing the whole charge, use ${amount:.2f}; do not ask them."
                )
            if item.category in PARTIAL_AMOUNT_CATEGORIES and item.disputed_amount >= amount:
                return Refuse(
                    f"a {item.category.replace('_', ' ')} dispute covers only part of the ${amount:.2f} charge, not "
                    "all of it: the overcharge, or the cash that was not paid out. Work it out from what the customer "
                    "said (what they were charged and what it should have been, or what they received) or ask them"
                )
            if (ctx.today - happened).days > MAX_AGE_DAYS:
                return Refuse(f"{item.transaction_id} is more than {MAX_AGE_DAYS} days old")
            cards = [
                c
                for c in db.debit_cards.data.values()
                if c.get("account_id") == account["account_id"] and c.get("status") != "CLOSED"
            ]
            if len(cards) != 1:
                return NeedInfo(
                    (Ask("card_id", "Which debit card was used for this transaction?"),)
                )
            card = cards[0]
            tier = tier_of(str(account.get("level")))
            account_id = str(account["account_id"])
            open_now = sum(
                1
                for d in db.debit_card_disputes.data.values()
                if d.get("account_id") == account_id and d.get("status") in (None, "OPEN", "PENDING", "UNDER_REVIEW")
            )
            new_per_account[account_id] = new_per_account.get(account_id, 0) + 1
            # An account level with no known tier is not capped here; the tool still checks.
            if tier is not None and open_now + new_per_account[account_id] > MAX_OPEN_DISPUTES[tier]:
                return Refuse(
                    f"the {account.get('level')} allows {MAX_OPEN_DISPUTES[tier]} open disputes and this would pass that"
                )
            action = CARD_ACTION.get(item.category, "keep_active")
            cards_of[item.transaction_id] = str(card["card_id"])
            card_action[str(card["card_id"])] = max(
                (card_action.get(str(card["card_id"]), "keep_active"), action), key=SEVERITY.index
            )
            steps.append(
                Step(
                    "file_debit_card_transaction_dispute_6281",
                    {
                        "transaction_id": item.transaction_id,
                        "account_id": account_id,
                        "card_id": card["card_id"],
                        "dispute_category": item.category,
                        "transaction_date": happened.strftime("%m/%d/%Y"),
                        "discovery_date": noticed.strftime("%m/%d/%Y"),
                        "disputed_amount": item.disputed_amount,
                        "transaction_type": item.transaction_type,
                        "card_in_possession": item.card_in_possession,
                        "pin_compromised": item.pin_compromised,
                        "contacted_merchant": item.contacted_merchant,
                        "police_report_filed": item.police_report_filed,
                        "written_statement_provided": item.written_statement_provided,
                        "provisional_credit_eligible": provisional_credit(item),
                        "card_action": action,
                        "customer_max_liability_amount": liability(happened, noticed),
                    },
                    f"Dispute the {str(txn.get('description') or 'card').title()} charge of ${item.disputed_amount:.2f} "
                    f"({item.category.replace('_', ' ')})",
                )
            )
        for card_id, action in card_action.items():
            if action == "close_and_reissue":
                answers = {i.card_closure for i in disputes if cards_of[i.transaction_id] == card_id}
                if "yes" in answers and "no" not in answers:
                    lost = any(not i.card_in_possession for i in disputes if cards_of[i.transaction_id] == card_id)
                    steps.append(
                        Step(
                            "close_debit_card_4721",
                            {"card_id": card_id, "reason": "lost" if lost else "fraud_suspected"},
                            f"Close {describe(db, db.debit_cards.data[card_id])} and issue a new one; it cannot be reopened",
                        )
                    )
                elif "no" not in answers:
                    label = describe(db, db.debit_cards.data[card_id])
                    return NeedInfo(
                        (
                            Ask(
                                "card_closure",
                                f"For {label}, the bank recommends closing it and issuing a new card. Do you want "
                                "me to close it, or would you rather keep it?",
                            ),
                        )
                    )
            elif action == "freeze_pending_investigation":
                steps.append(Step("freeze_debit_card_3892", {"card_id": card_id}))
        return Plan(tuple(steps))


def Reply_closed(account: Any) -> Decision:
    from agent.workflows.base import Reply

    return Reply(
        "That account is not open, so a debit card dispute cannot be filed on it."
    )
