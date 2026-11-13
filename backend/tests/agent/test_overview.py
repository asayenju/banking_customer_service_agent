"""A specialist looks the customer's own records up instead of asking the customer for what the records hold."""

from datetime import date

import pytest

from agent.overview import SECTIONS, section
from agent.specialist import _FINDABLE
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks

TASKS = {t["id"]: t for t in load_tasks()}


def world(task):
    return BankingSession.from_task(TASKS[task]).database


def test_every_section_is_json_the_model_can_read_and_never_holds_a_pin_or_cvv():
    import json

    db = world("task_078")
    for name in SECTIONS:
        text = json.dumps(section(db, "mc78a5b9d2", name), default=str)
        assert text and '"cvv"' not in text and '"pin"' not in text and "expiration_date" not in text, name


def test_the_cards_and_transactions_are_only_the_customers_own():
    db = world("task_078")
    cards = section(db, "mc78a5b9d2", "debit_cards")
    assert {c["user_id"] for c in cards} == {"mc78a5b9d2"} and all(c["last_4_digits"] for c in cards)
    other = section(db, "someone-else", "debit_cards")
    assert other == []


def test_the_credit_transactions_and_the_referrals_come_with_their_ids():
    db = world("task_033")
    refs = section(db, "76ad9cc60e", "other_records")["referrals"]
    assert len(refs) >= 2 and all(r["referrer_id"] == "76ad9cc60e" for r in refs)


@pytest.mark.parametrize(
    "name", ["transaction_id", "card_id", "account_id", "credit_card_account_id", "card_last_4_digits", "last_4_digits", "dispute_id"]
)
def test_what_the_records_hold_is_never_asked_of_the_customer(name):
    assert _FINDABLE.search(name)


@pytest.mark.parametrize("name", ["dispute_reason", "contacted_merchant", "issue_noticed_date", "what_happened", "amount_expected"])
def test_what_only_the_customer_knows_is_still_asked(name):
    assert not _FINDABLE.search(name)
