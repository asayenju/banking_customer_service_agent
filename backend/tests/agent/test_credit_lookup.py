"""The credit specialist searches the customer's records before it asks them anything about a charge."""

import asyncio
import json
from datetime import date

import pytest

from agent.credit_lookup import find_transactions, list_cards, parse_when
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist
from agent.specialists.credit import CREDIT_SPECIALIST
from agent.toolbox import SecretVault
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex
from orchestrator.context import RouteContext, SessionRequest
from orchestrator.executor import OutcomeStatus
from orchestrator.context import scoped
from orchestrator.types import Intent, IntentKind as K

TODAY = date(2025, 11, 14)
FATIMA = "890389b165"
INDEX = KnowledgeIndex({"doc_credit_cards_x": Document("doc_credit_cards_x", "Internal: credit disputes", "dispute a credit card charge")})


def world():
    return BankingSession.from_task({t["id"]: t for t in load_tasks()}["task_031"])


def find(db, **said):
    base = dict(merchant="", amount="", on_or_after="", on_or_before="", transaction_id="")
    return find_transactions(db, FATIMA, TODAY, **{**base, **said})


@pytest.mark.parametrize(
    "text", ["11/07/25", "11/07/2025", "November 7th of 2025", "nov 7, 2025", "2025-11-07", "november 7th 2025"]
)
def test_a_date_is_understood_however_it_is_written(text):
    assert parse_when(text) == date(2025, 11, 7)


def test_a_date_with_no_year_takes_the_banks_current_year():
    assert parse_when("November 7th", TODAY) == date(2025, 11, 7) and parse_when("November 7th") is None


def test_the_marriott_charge_is_found_by_the_merchant_alone_by_a_rough_amount_and_by_its_id():
    db = world().database
    assert [r["transaction_id"] for r in find(db, merchant="marriott")["matches"]] == ["txn_adea68821a1d"]
    assert [r["transaction_id"] for r in find(db, merchant="Marriott hotel", amount="around 160")["matches"]] == ["txn_adea68821a1d"]
    assert find(db, transaction_id="txn_adea68821a1d")["matches"][0]["amount"] == "$167.34"


def test_a_date_filter_in_any_spelling_finds_it_and_a_wrong_one_does_not():
    db = world().database
    assert find(db, merchant="Marriott", on_or_after="11/01/25", on_or_before="November 14th 2025")["matches"]
    wrong = find(db, merchant="Marriott", on_or_after="01/01/2025", on_or_before="02/01/2025")
    assert wrong["matches"] == [] and wrong["merchant_only"][0]["transaction_id"] == "txn_adea68821a1d", "it says what fits the merchant"


def test_another_customers_charge_is_never_returned():
    db = world().database
    assert find(db, transaction_id="txn_a7b8c9d0e1f2")["matches"] == [], "the Gold card's Marriott charge is somebody else's"
    assert all(r["transaction_id"] != "txn_a7b8c9d0e1f2" for r in find(db, merchant="marriott")["matches"])


def test_the_customers_cards_are_listed():
    assert any(c["card_type"] == "Silver Rewards Card" for c in list_cards(world().database, FATIMA))


# ---------------------------------------------------------------- the rule: look before you ask

def call(name, **arguments):
    return ToolCall(id=f"c-{name}-{len(arguments)}", name=name, arguments_json=json.dumps(arguments))


def turn(*calls):
    return AssistantTurn(tool_calls=tuple(calls), usage=Usage(input_tokens=10))


def run(script, kind=K.DISPUTE_TRANSACTION):
    shared = world()
    llm = FakeToolLLM(script)
    spec = Specialist(CREDIT_SPECIALIST, llm=llm, index=INDEX, database=shared.database, vault=SecretVault())
    intent = Intent(id="x", kind=kind, summary="dispute a marriott charge", slots={})
    ctx = RouteContext(intent_id="x", verified_user_id=FATIMA, session=scoped(shared)(SessionRequest("x")), mode="plan")
    return asyncio.run(spec.run(intent, ctx)), llm


def test_the_search_tools_are_offered_to_the_credit_specialist():
    _, llm = run([turn(call("ask_customer", details=["x"]))])
    assert {"find_credit_transactions", "list_credit_cards", "credit_dispute_history", "look_up_customer_records"} <= set(llm.calls[0]["tools"])


def test_asking_the_customer_before_any_search_is_refused_and_the_model_is_told_to_search():
    outcome, llm = run(
        [
            turn(call("ask_customer", details=["transaction_date"], question="What date was it?")),
            turn(call("find_credit_transactions", merchant="Marriott", amount="", on_or_after="", on_or_before="", transaction_id="")),
            turn(call("ask_customer", details=["dispute_reason"], question="What went wrong?")),
        ]
    )
    told = llm.calls[1]["messages"][-1]["content"]
    assert "look it up first" in told and "find_credit_transactions" in told
    assert outcome.status is OutcomeStatus.NEEDS_INFO and outcome.questions[0].slot == "dispute_reason"


def test_any_one_search_tool_unlocks_asking_and_the_rule_is_for_disputes_only():
    outcome, _ = run([turn(call("list_credit_cards")), turn(call("ask_customer", details=["dispute_reason"], question="Why?"))])
    assert outcome.status is OutcomeStatus.NEEDS_INFO
    close, _ = run([turn(call("ask_customer", details=["reason"], question="Why close it?"))], kind=K.CLOSE_CARD)
    assert close.status is OutcomeStatus.NEEDS_INFO, "closing a card is not held to the rule"


def test_the_credit_cards_last_four_digits_are_the_customers_to_give_because_the_records_do_not_hold_them():
    outcome, _ = run(
        [
            turn(call("list_credit_cards")),
            turn(call("ask_customer", details=["card_last_4_digits"], question="What are the last four digits on the card?")),
        ]
    )
    assert outcome.status is OutcomeStatus.NEEDS_INFO and outcome.questions[0].slot == "card_last_4_digits"


def test_an_id_the_records_hold_is_still_never_asked_of_a_credit_customer():
    outcome, llm = run(
        [
            turn(call("list_credit_cards")),
            turn(call("ask_customer", details=["transaction_id"], question="What is the transaction id?")),
            turn(call("ask_customer", details=["dispute_reason"], question="What went wrong?")),
        ]
    )
    assert "look it up" in llm.calls[2]["messages"][-1]["content"] and outcome.questions[0].slot == "dispute_reason"
