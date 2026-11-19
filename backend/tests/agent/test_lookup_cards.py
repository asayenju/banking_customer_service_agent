"""Asking about cards lists credit cards too, so a customer with only credit cards is never told they have none."""

import asyncio

from agent.recipes.direct import DirectRunner
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.types import Intent, IntentKind as K

TASK = next(t for t in load_tasks() if t["id"] == "task_026")  # Amara Okonkwo: Business Silver and Silver Rewards cards


def test_a_card_lookup_names_the_credit_cards_when_there_is_no_debit_card():
    session = BankingSession.from_task(TASK)
    db = session.database
    user = next(a["user_id"] for a in db.credit_card_accounts.data.values())
    ctx = RouteContext(intent_id="x", verified_user_id=user, session=scoped(session)(SessionRequest("x")), mode="plan")
    intent = Intent(id="x", kind=K.LOOKUP_INFO, summary="check my business silver rewards card")
    outcome = asyncio.run(DirectRunner(db).run(intent, ctx))
    assert "no cards on file" not in outcome.text
    assert "Silver" in outcome.text


def test_asking_about_a_card_type_the_customer_does_not_hold_says_so_and_shows_what_they_have():
    session = BankingSession.from_task(next(t for t in load_tasks() if t["id"] == "task_005"))  # Kenji: a Silver Rewards Card only
    db = session.database
    user = "6680a37184"
    ctx = RouteContext(intent_id="x", verified_user_id=user, session=scoped(session)(SessionRequest("x")), mode="plan")
    ask = lambda said: asyncio.run(DirectRunner(db).run(Intent(id="x", kind=K.ACCOUNT_BALANCE_OR_HISTORY, summary=said), ctx)).text
    gold = ask("check the balance on my Gold Rewards card")
    assert gold.startswith("I don't see a Gold Rewards Card on your profile.") and "Silver Rewards Card" in gold
    assert "I don't see" not in ask("check the balance on my Silver Rewards card"), "a card they hold is shown, not reported missing"
    assert "I don't see" not in ask("show my balance"), "no card named, no complaint"
