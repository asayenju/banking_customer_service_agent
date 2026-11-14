"""The card resolver: the Carlos wallet (Evergreen 3847, Green 8165, Light Blue 5291, closed 2847) in ordinary words."""

import pytest

from agent.cards import owned_cards
from agent.entities import resolve_cards
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks

CARLOS = "cr89a2b3c4"
TASKS = {t["id"]: t for t in load_tasks()}
DB = BankingSession.from_task(TASKS["task_079"]).database
SHOWN = owned_cards(DB, CARLOS)
OWNED = owned_cards(DB, CARLOS, statuses=["ACTIVE", "FROZEN", "PENDING", "CLOSED"])


def fours(text, **kw):
    return [c["last_4_digits"] for c in resolve_cards(text, SHOWN, DB, owned=OWNED, **kw).cards]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("evergreen and green accounts", ["3847", "8165"]),
        ("3847 and 8165", ["3847", "8165"]),
        ("the card ending 3847 (active, on your Evergreen Account)", ["3847"]),
        ("my Light Blue card", ["5291"]),
        ("Green", ["8165"]),
        ("dbc_cr89a2b3c4_green", ["8165"]),
    ],
)
def test_cards_are_named_by_account_last_four_pasted_line_or_id(text, expected):
    assert sorted(fours(text)) == sorted(expected)


def test_ordinals_count_along_the_list_that_was_shown():
    # The list is in card id order: Evergreen (ev), Green, Light Blue (lb).
    order = [c["last_4_digits"] for c in SHOWN]
    assert order == ["3847", "8165", "5291"]
    assert fours("the first two") == order[:2]
    assert fours("the second one") == [order[1]]
    assert fours("the last one") == [order[-1]]


@pytest.mark.parametrize("text", ["all", "both of them", "every card", "those", "close them"])
def test_all_and_those_mean_every_card_on_offer(text):
    resolved = resolve_cards(text, SHOWN, DB, owned=OWNED)
    assert resolved.every and len(resolved.cards) == 3


def test_a_number_that_is_not_theirs_is_reported_and_never_swapped_for_another_card():
    resolved = resolve_cards("9999", SHOWN, DB, owned=OWNED)
    assert resolved.cards == () and resolved.unknown == ("9999",)
    mixed = resolve_cards("3847 and 9999", SHOWN, DB, owned=OWNED)
    assert [c["last_4_digits"] for c in mixed.cards] == ["3847"] and mixed.unknown == ("9999",)


def test_a_closed_card_named_is_reported_not_used():
    resolved = resolve_cards("2847", SHOWN, DB, owned=OWNED)
    assert resolved.cards == () and [c["last_4_digits"] for c in resolved.closed] == ["2847"]


def test_green_is_not_light_green_and_light_green_is_not_green():
    db = BankingSession.from_task(TASKS["task_078"]).database
    shown = owned_cards(db, "mc78a5b9d2")
    names = lambda t: [c["last_4_digits"] for c in resolve_cards(t, shown, db).cards]  # noqa: E731
    assert names("my green card") == ["7391"]
    assert names("the light green card") == ["8624"]


def test_empty_or_unrelated_words_name_nothing():
    for text in ("", "  ", "hmm", "not sure"):
        assert resolve_cards(text, SHOWN, DB, owned=OWNED).cards == ()


@pytest.mark.parametrize(("text", "expected"), [("2", "8165"), ("number 3", "5291"), ("#1", "3847")])
def test_a_number_picks_from_the_numbered_list_just_shown(text, expected):
    assert fours(text) == [expected]
    assert resolve_cards("7", SHOWN, DB, owned=OWNED).cards == ()
