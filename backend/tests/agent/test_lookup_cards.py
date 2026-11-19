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
