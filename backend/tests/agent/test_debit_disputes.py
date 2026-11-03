import json
from datetime import date

import pytest

from agent.specialist import ValidationContext
from agent.specialists import debit
from agent.specialists.debit import DEBIT_SPECIALIST, PARTIAL_AMOUNT_CATEGORIES
from agent.tier1 import Tier1Report, gold_steps, invalid_gold, render, run_tier1
from banking_tools import BankingSession, Fidelity
from banking_tools.manifest import build_manifest, load_tasks
from banking_tools.session import _base_db
from banking_tools.task_state import apply_initial_state
from knowledge.index import Document, KnowledgeIndex

TODAY = date(2025, 11, 14)
INDEX = KnowledgeIndex({"doc_bank_accounts_x": Document("doc_bank_accounts_x", "t", "x")})
TASKS = {t["id"]: t for t in load_tasks()}
ENTRIES = {e.task_id: e for e in build_manifest(list(TASKS.values()))}
DISPUTE = "file_debit_card_transaction_dispute_6281"


def v(db, user="u"):
    return ValidationContext(db, user, TODAY)


def world():
    """A tiny database: one account, one card, one withdrawal of $200."""
    db = _base_db().model_copy(deep=True)
    account, card = "chk_x", "dbc_x"
    db.accounts.data[account] = {"account_id": account, "user_id": "u", "class": "checking", "status": "OPEN"}
    db.debit_cards.data[card] = {"card_id": card, "account_id": account, "user_id": "u", "status": "ACTIVE", "last_4_digits": "1111"}
    db.bank_account_transaction_history.data["btxn_x"] = {"transaction_id": "btxn_x", "account_id": account, "date": "11/09/2025", "amount": "-200.0", "type": "atm_withdrawal"}
    return db


GOOD = {
    "transaction_id": "btxn_x", "account_id": "chk_x", "card_id": "dbc_x", "user_id": "u",
    "dispute_category": "card_present_fraud", "transaction_date": "11/09/2025", "discovery_date": "11/10/2025",
    "disputed_amount": 200.0, "transaction_type": "pin_purchase", "card_in_possession": True, "pin_compromised": "no",
    "contacted_merchant": False, "police_report_filed": False, "written_statement_provided": True,
    "provisional_credit_eligible": True, "customer_max_liability_amount": 50.0, "card_action": "keep_active",
}


def check(**change):
    return debit.dispute(v(world()), {**GOOD, **change})


def test_a_grounded_dispute_is_accepted():
    assert check() is None


@pytest.mark.parametrize(
    "change, part",
    [
        ({"transaction_id": "btxn_made_up"}, "does not exist"),
        ({"account_id": "chk_other"}, "not on that account"),
        ({"card_id": "dbc_made_up"}, "card does not exist"),
        ({"disputed_amount": 200.01}, "more than the transaction"),
        ({"disputed_amount": 0.0}, "more than zero"),
        ({"disputed_amount": -5.0}, "more than zero"),
        ({"disputed_amount": 150.0}, "whole amount"),
        ({"dispute_category": "duplicate_charge", "disputed_amount": 50.0}, "whole amount"),
        ({"transaction_date": "11/08/2025"}, "not the date on the transaction"),
        ({"transaction_date": "2025-11-09"}, "MM/DD/YYYY"),
        ({"discovery_date": "11/08/2025"}, "before it happened"),
        ({"discovery_date": "11/15/2025"}, "after today"),
        ({"provisional_credit_eligible": True, "written_statement_provided": False}, "written statement"),
    ],
)
def test_an_ungrounded_dispute_is_refused_with_a_reason(change, part):
    assert part in (check(**change) or "")


@pytest.mark.parametrize("category", sorted(PARTIAL_AMOUNT_CATEGORIES))
def test_only_the_partial_categories_may_dispute_part_of_a_charge(category):
    assert check(dispute_category=category, disputed_amount=100.0) is None


def test_a_card_from_another_account_cannot_be_used():
    db = world()
    db.debit_cards.data["dbc_y"] = {"card_id": "dbc_y", "account_id": "chk_y", "user_id": "u", "status": "ACTIVE"}
    assert "not linked to that account" in debit.dispute(v(db), {**GOOD, "card_id": "dbc_y"})


def test_a_timestamped_transaction_date_matches_by_its_day():
    db = world()
    db.bank_account_transaction_history.data["btxn_x"]["date"] = "11/09/2025 02:02:15"
    assert debit.dispute(v(db), GOOD) is None


def test_a_deposit_is_disputed_by_its_size_not_its_sign():
    db = world()
    db.bank_account_transaction_history.data["btxn_x"]["amount"] = "400.0"
    assert debit.dispute(v(db), {**GOOD, "dispute_category": "atm_deposit_not_credited", "disputed_amount": 100.0}) is None


# ------------------------------------------------------------------ against all the bank's own examples

def gold_disputes():
    for task in TASKS.values():
        for step in gold_steps(task):
            if step.name == DISPUTE:
                yield task, step


def test_every_gold_dispute_passes_the_validator_on_the_database_as_it_stood_when_it_was_filed():
    """The rules above are not invented: the bank's 23 reference disputes all satisfy them."""
    checked = 0
    for task in TASKS.values():
        steps = gold_steps(task)
        if not any(s.name == DISPUTE for s in steps):
            continue
        session = BankingSession.from_task(task, fidelity=Fidelity.TAU2)
        for step in steps:
            if step.kind == "write" and step.name == DISPUTE:
                problem = debit.dispute(ValidationContext(session.database, str(step.args["user_id"]), TODAY), step.args)
                assert problem is None, (task["id"], step.args["transaction_id"], problem)
                checked += 1
            if step.kind in ("write", "read", "verify") and invalid_gold([step]) is None:
                session.call(step.name, step.args)  # so the next dispute sees the state this one left
    assert checked >= 20


# ------------------------------------------------------------------ tier 1: the dispute tasks

@pytest.mark.parametrize("task_id", ["task_082", "task_084", "task_085", "task_086"])
def test_the_dispute_segments_of_these_tasks_end_in_the_database_tau2_gets(task_id):
    result = run_tier1(TASKS[task_id], DEBIT_SPECIALIST, index=INDEX)
    assert result.error is None, result.error
    assert result.strict_equal and result.semantic_equal, result.mismatches
    assert any(seg.startswith("DISPUTE_TRANSACTION") for seg in result.segments), result.segments


def test_a_dispute_that_does_not_match_the_transaction_stops_the_tier_one_run():
    task = json.loads(json.dumps(TASKS["task_084"]))
    for action in task["evaluation_criteria"]["actions"]:
        if action["name"] == "call_discoverable_agent_tool" and action["arguments"]["agent_tool_name"] == DISPUTE:
            args = json.loads(action["arguments"]["arguments"])
            args["disputed_amount"] = 99999.0
            action["arguments"]["arguments"] = json.dumps(args)
            break
    result = run_tier1(task, DEBIT_SPECIALIST, index=INDEX)
    assert result.error and "more than the transaction" in result.error and not result.passed


# ------------------------------------------------------------------ the whole pilot, offline

PILOT = [t for t in TASKS.values() if ENTRIES[t["id"]].pilot]


def test_the_pilot_is_the_thirteen_wholly_debit_tasks():
    assert len(PILOT) == 13


def test_task_083_is_excluded_because_its_own_gold_breaks_the_banks_tool_contract():
    result = run_tier1(TASKS["task_083"], DEBIT_SPECIALIST, index=INDEX)
    assert result.excluded and "customer_max_liability_amount" in result.excluded
    assert not result.passed and result.error is None
    assert "customer_max_liability_amount" in (ENTRIES["task_083"].excluded_reason or "")


def test_every_other_pilot_task_ends_in_taus_exact_database_hash():
    report = Tier1Report(tuple(run_tier1(t, DEBIT_SPECIALIST, index=INDEX) for t in PILOT))
    text = render(report)
    usable = [r for r in report.results if not r.excluded]
    assert len(usable) == 12 and "1 excluded" in text
    failures = [(r.task_id, r.error or r.mismatches) for r in usable if not (r.passed and r.strict_equal)]
    assert failures == [], text
