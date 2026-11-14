"""Settling which debit cards a request means, before any model runs (the Carlos loop)."""

from agent.specialists.debit import prepare
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks
from orchestrator.executor import RouteOutcome
from orchestrator.types import Intent, IntentKind as K

CARLOS = "cr89a2b3c4"
DB = BankingSession.from_task({t["id"]: t for t in load_tasks()}["task_079"]).database
EV, GREEN, LB = "dbc_cr89a2b3c4_ev", "dbc_cr89a2b3c4_green", "dbc_cr89a2b3c4_lb"


def run(kind, card):
    return prepare(Intent(id="x", kind=kind, summary="s", slots={"card_id": card} if card else {}), CARLOS, DB)


def test_two_cards_named_by_account_are_both_closed_in_one_request():
    out = run(K.CLOSE_CARD, "evergreen and green accounts")
    assert not isinstance(out, RouteOutcome)
    assert out.slots["card_ids"] == f"{EV}, {GREEN}" and "card_id" not in out.slots


def test_two_cards_named_by_last_four_digits():
    out = run(K.CLOSE_CARD, "3847 and 8165")
    assert out.slots["card_ids"] == f"{EV}, {GREEN}"


def test_the_pasted_list_line_picks_that_card():
    assert run(K.CLOSE_CARD, "the card ending 3847 (active, on your Evergreen Account)").slots["card_id"] == EV


def test_those_and_all_mean_every_card_for_a_close():
    out = run(K.CLOSE_CARD, "those")
    assert out.slots["card_ids"].count("dbc_") == 3


def test_a_card_that_is_not_theirs_is_said_and_never_swapped():
    out = run(K.CLOSE_CARD, "3847 and 9999")
    assert isinstance(out, RouteOutcome)
    text = out.questions[0].text
    assert "9999" in text and text.count("Which card do you mean?") == 1


def test_a_closed_card_is_reported():
    out = run(K.CLOSE_CARD, "2847")
    assert isinstance(out, RouteOutcome) and "already closed" in out.questions[0].text


def test_other_kinds_still_take_one_card_only():
    out = run(K.CHANGE_OR_RESET_PIN, "3847 and 8165")
    assert isinstance(out, RouteOutcome) and "Which card do you mean?" in out.questions[0].text
    assert run(K.CHANGE_OR_RESET_PIN, "8165").slots["card_id"] == GREEN


def cancel(**slots):
    return prepare(Intent(id="x", kind=K.CLOSE_CARD, summary="cancel some debit cards", slots=slots), CARLOS, DB)


def test_cancel_with_no_reason_asks_freeze_or_close_once():
    out = cancel(card_id="3847")
    assert isinstance(out, RouteOutcome) and out.questions[0].slot == "cancel_means"
    assert "freeze it for now" in out.questions[0].text


def test_closing_for_good_goes_on_without_asking_again():
    out = cancel(card_id="3847", cancel_means="close it for good")
    assert not isinstance(out, RouteOutcome) and out.slots["card_id"] == EV


def test_choosing_freeze_closes_nothing():
    out = cancel(card_id="3847", cancel_means="just freeze it for now")
    assert isinstance(out, RouteOutcome) and "not closed anything" in out.text


def test_a_clear_close_or_a_lost_card_is_not_asked():
    for said in ("close my card", "my card was stolen, cancel it"):
        out = prepare(Intent(id="x", kind=K.CLOSE_CARD, summary=said, slots={"card_id": "3847"}), CARLOS, DB)
        assert not isinstance(out, RouteOutcome)


def test_a_freeze_answered_with_a_number_from_the_list_freezes_that_card():
    from agent.recipes.direct import DirectRunner

    import asyncio
    from orchestrator.context import RouteContext, SessionRequest, scoped

    shared = BankingSession.from_task({t["id"]: t for t in load_tasks()}["task_079"])
    runner = DirectRunner(shared.database)
    intent = Intent(id="x", kind=K.FREEZE_CARD, summary="freeze a card", slots={"card_id": "2"})
    ctx = RouteContext(intent_id="x", verified_user_id=CARLOS, session=scoped(shared)(SessionRequest("x")), mode="execute")
    out = asyncio.run(runner.run(intent, ctx))
    assert "8165" in out.text and shared.database.debit_cards.data[GREEN]["status"] == "FROZEN"
