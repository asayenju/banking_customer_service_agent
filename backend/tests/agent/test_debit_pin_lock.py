"""Doc 041's fraud risk score, worked out from the records. The score is checked against the cards of tasks 090 to 092."""

from datetime import date

import pytest

from agent.workflows.debit_pin_lock import home_place, level_of, score_card
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks

TASKS = {t["id"]: t for t in load_tasks()}
TODAY = date(2025, 11, 14)


def card_score(task, suffix, **kw):
    db = BankingSession.from_task(TASKS[task]).database
    card = next(c for c in db.debit_cards.data.values() if c["card_id"].endswith(suffix) and c.get("pin_locked"))
    return score_card(db, card, TODAY, **kw)


@pytest.mark.parametrize(
    ("total", "level"),
    [(0, "LOW"), (4, "LOW"), (5, "MEDIUM"), (7, "MEDIUM"), (8, "HIGH"), (10, "HIGH"), (11, "VERY_HIGH"), (14, "VERY_HIGH"), (15, "CRITICAL")],
)
def test_the_thresholds_are_the_documents(total, level):
    assert level_of(total) == level


def test_the_home_city_comes_from_the_address_on_file():
    assert home_place("2847 Pine Street, Unit 12C, Seattle, WA 98101") == ("SEATTLE", "WA")
    assert home_place("no address") == ("", "")


def test_a_routine_decline_at_a_home_shop_scores_low():
    s = card_score("task_091", "_green")
    assert s.level == "LOW" and s.flags["A1"] == 0


def test_a_2am_decline_in_another_state_is_a_single_high_risk_flag():
    s = card_score("task_091", "_evergreen")
    assert s.flags["A1"] == 2 and s.flags["B1"] == 3 and s.flags["C1"] == 2 and s.escalation == "single_flag"


def test_three_prior_locks_is_the_automatic_pin_reset_flag():
    s = card_score("task_091", "_blue")
    assert s.flags["D1"] == 3 and s.escalation == "single_flag"


def test_a_country_mismatch_is_critical_and_a_security_hold_escalates():
    assert card_score("task_092", "_blue").flags["A1"] == 3
    assert card_score("task_092", "_lightblue").escalation == "security_hold"


def test_an_answer_that_clears_the_location_removes_its_flags_and_the_score_is_recalculated():
    base = card_score("task_091", "_lightblue")
    cleared = card_score("task_091", "_lightblue", removed=frozenset({"location"}))
    assert cleared.total == base.total - base.flags["A1"] - base.flags["A2"]
