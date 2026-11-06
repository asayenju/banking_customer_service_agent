"""Slice 1 of the fees specialist: reading transactions, places, cards and levels from the records.

Real task data (starting databases of tasks 072, 073, 097) so the readers meet what the bank actually
stores. Another customer's account is never read, and an ending that cannot tell home from abroad is
unknown, never guessed.
"""

from datetime import date

from agent import fee_records as R
from agent.bank_accounts import accounts_of
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks

TASKS = {t["id"]: t for t in load_tasks()}


def world(task: str) -> BankingSession:
    return BankingSession.from_task(TASKS[task])


def test_transactions_are_one_accounts_oldest_first_and_only_posted() -> None:
    db = world("task_072").database
    rows = R.transactions_of(db, "chk_lj82d4f1a9")
    assert rows and all(t.account_id == "chk_lj82d4f1a9" for t in rows)
    assert rows == sorted(rows, key=lambda t: (t.when or date.min, t.transaction_id))
    assert {t.kind for t in rows} >= {"atm_withdrawal", "atm_fee", "fee_rebate"}
    assert all(t.when is not None for t in rows)


def test_another_customers_account_has_no_transactions_under_a_wrong_id() -> None:
    db = world("task_072").database
    assert R.transactions_of(db, "chk_does_not_exist") == []


def test_places_decide_home_or_abroad() -> None:
    assert R.is_foreign("ATM WITHDRAWAL - CHASE BANK #2847 CHICAGO IL") is False
    assert R.is_foreign("ATM WITHDRAWAL - BARCLAYS BANK LONDON UK") is True
    assert R.is_foreign("ATM WITHDRAWAL - SHINHAN BANK SEOUL KR") is True
    assert R.is_foreign("ATM WITHDRAWAL - HSBC HONG KONG HK") is True
    assert R.is_foreign("ATM WITHDRAWAL - SANTANDER MADRID ES") is True
    assert R.is_foreign("ATM WITHDRAWAL - BANAMEX MEXICO CITY MX") is True


def test_overlapping_endings_are_settled_by_the_city_or_left_unknown() -> None:
    assert R.is_foreign("ATM WITHDRAWAL - DEUTSCHE BANK FRANKFURT DE") is True
    assert R.is_foreign("ATM WITHDRAWAL - DEUTSCHE BANK BERLIN DE") is True
    assert R.is_foreign("ATM WITHDRAWAL - CHASE #4521 LOS ANGELES CA") is False  # California: a branch number
    assert R.is_foreign("ATM WITHDRAWAL - SOME BANK TORONTO CA") is True
    assert R.is_foreign("ATM WITHDRAWAL - SOME BANK DOVER DE") is None  # unknown: hand off, never guess
    assert R.is_foreign("ATM WITHDRAWAL - NO PLACE") is None
    assert R.is_foreign("") is None


def test_the_banks_own_atm_is_recognised() -> None:
    assert R.is_rho_atm("ATM WITHDRAWAL - RHO-BANK #1234 LOS ANGELES CA")
    assert not R.is_rho_atm("ATM WITHDRAWAL - CHASE BANK #4521 LOS ANGELES CA")


def test_every_withdrawal_in_the_checking_tasks_has_a_decided_place() -> None:
    for task in ("task_072", "task_073", "task_074"):
        db = world(task).database
        for account in accounts_of(db, _user(task)):
            if account.is_checking:
                for t in R.transactions_of(db, account.account_id):
                    if t.kind == "atm_withdrawal":
                        assert R.is_foreign(t.description) is not None, (task, t.description)


def _user(task: str) -> str:
    return next(
        a["arguments"]["user_id"] for a in TASKS[task]["evaluation_criteria"]["actions"] if a["name"] == "log_verification"
    )


def test_card_types_become_table_keys() -> None:
    assert R.card_key("EcoCard") == "EcoCard"
    assert R.card_key("Crypto-Cash Back") == "Crypto-Cash Back Card"
    assert R.card_key("crypto-cash back card") == "Crypto-Cash Back Card"
    assert R.card_key("Business Platinum Rewards Card") is None  # a business card has no savings bonus table
    assert R.card_key(None) is None


def test_cards_held_in_task_097_and_none_in_task_093() -> None:
    known, unknown = R.cards_held(world("task_097").database, _user("task_097"))
    assert sorted(known) == sorted(
        ["Silver Rewards Card", "EcoCard", "Bronze Rewards Card", "Crypto-Cash Back Card", "Diamond Elite Card"]
    )
    assert unknown == []
    assert R.cards_held(world("task_093").database, _user("task_093")) == ([], [])


def test_levels_become_table_keys_or_none() -> None:
    assert R.level_key("silver account", R.SAVINGS_BASE) == "Silver Account"
    assert R.has_savings_table("Gold Plus Account") and not R.has_savings_table("Emerald Saver Account")
    assert R.has_fee_schedule("Blue Account") and not R.has_fee_schedule("Gold Years Account")


def test_boost_pairings() -> None:
    assert R.boost_for("Green Account", "Gold Account") == 0.75
    assert R.boost_for("Purple Account", "Gold Account") == 0.1
    assert R.boost_for("Blue Account", "Gold Account") == 0.0  # a pairing the documents do not list: no boost
    assert R.boost_for("Gold Years Account", "Gold Plus Account") == 0.5
    assert R.boost_for("Mystery Account", "Gold Account") is None
    assert R.boost_for("Green Account", "Mystery Account") is None


def test_savings_accounts_are_read_in_either_layout_with_their_balance() -> None:
    db = world("task_097").database
    savings = [a for a in accounts_of(db, _user("task_097")) if a.product.value == "savings"]
    assert {a.name: a.balance for a in savings} == {
        "Silver Account": 100000.0, "Platinum Account": 70000.0, "Diamond Elite Account": 120000.0,
        "Silver Plus Account": 18000.0,
    }
