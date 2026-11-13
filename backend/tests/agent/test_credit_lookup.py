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


# ---------------------------------------------------------------- the ways a model or a customer might ask for the same charge

MARRIOTT = "txn_adea68821a1d"


def ids(found):
    return [r["transaction_id"] for r in found["matches"]]


@pytest.mark.parametrize(
    "merchant",
    ["Marriott", "marriott", "MARRIOTT HOTELS", "Marriott Hotels", "Marriott hotel", "a Marriott hotel", "marriott hotels charge", "Marriot", "mariot hotel", "Mariott", "marrriott"],
)
def test_the_merchant_is_found_however_it_is_spelled_or_misspelled(merchant):
    assert MARRIOTT in ids(find(world().database, merchant=merchant))


@pytest.mark.parametrize("merchant", ["Hilton", "Amazon", "zzz", "", "ab"])
def test_a_merchant_that_is_not_there_does_not_match_the_marriott_charge(merchant):
    found = ids(find(world().database, merchant=merchant)) if merchant else []
    assert MARRIOTT not in found


@pytest.mark.parametrize("amount", ["167.34", "$167.34", "around 167", "about $160", "160", "170", "around 160 dollars", "167"])
def test_the_amount_is_found_exactly_or_roughly(amount):
    assert MARRIOTT in ids(find(world().database, merchant="marriott", amount=amount))


@pytest.mark.parametrize("amount", ["167.00", "$300", "around 50"])
def test_a_wrong_amount_does_not_match(amount):
    assert MARRIOTT not in ids(find(world().database, merchant="marriott", amount=amount))


@pytest.mark.parametrize(
    ("after", "before"),
    [
        ("11/07/25", "11/07/25"),
        ("11/07/2025", "11/07/2025"),
        ("November 7th of 2025", "November 7th of 2025"),
        ("Nov 7", "Nov 7"),
        ("11/01/2025", ""),
        ("", "11/14/2025"),
        ("2025-11-01", "2025-11-14"),
        ("October 31, 2025", "November 8, 2025"),
    ],
)
def test_the_date_filters_find_it_on_the_day_in_a_range_and_with_one_side_open(after, before):
    assert MARRIOTT in ids(find(world().database, merchant="marriott", on_or_after=after, on_or_before=before))


@pytest.mark.parametrize(("after", "before"), [("11/08/2025", ""), ("", "11/06/2025"), ("11/01/2024", "11/30/2024")])
def test_a_date_range_that_excludes_the_day_does_not_find_it(after, before):
    assert MARRIOTT not in ids(find(world().database, merchant="marriott", on_or_after=after, on_or_before=before))


def test_a_date_the_search_cannot_read_is_ignored_not_treated_as_no_match():
    assert MARRIOTT in ids(find(world().database, merchant="marriott", on_or_after="last two weeks", on_or_before="recently"))


def test_with_every_filter_wrong_the_merchant_only_rows_are_still_offered():
    found = find(world().database, merchant="marriott", amount="$9999", on_or_after="11/08/2025")
    assert found["matches"] == [] and MARRIOTT in [r["transaction_id"] for r in found["merchant_only"]]


def test_the_search_works_through_the_tool_the_model_calls_and_returns_json():
    from agent.credit_lookup import FIND, credit_lookups

    tool = next(t for t in credit_lookups(world().database, FATIMA, TODAY) if t.schema.name == FIND)
    text = tool.run({"merchant": "mariot hotel", "amount": "", "on_or_after": "11/07/25", "on_or_before": "11/07/25", "transaction_id": ""})
    assert MARRIOTT in text and json.loads(text)["matches"][0]["amount"] == "$167.34"
    assert set(tool.schema.parameters["required"]) == set(tool.schema.parameters["properties"]), "strict: every argument is given"


def test_a_customer_with_no_such_charge_gets_an_empty_answer_with_a_tip():
    found = find_transactions(world().database, "someone-without-cards", TODAY, "marriott", "", "", "", "")
    assert found["matches"] == [] and "tip" in found


def test_each_merchant_the_customer_has_is_found_by_its_own_name_and_by_no_other_merchants_rows():
    """Spelling tolerance must not make 'Facebook Ads' also return the Google Ads charges."""
    db = world().database
    names = sorted({t["merchant_name"] for t in db.credit_card_transaction_history.data.values() if t["user_id"] == FATIMA})
    assert len(names) > 30
    for name in names:
        got = {r["merchant"] for r in find(db, merchant=name)["matches"]}
        assert got == {name}, (name, got)


@pytest.mark.parametrize(("asked", "merchant", "fits"), [
    ("marriott", "Marriott Hotels", True), ("mariot hotel", "Marriott Hotels", True), ("a hotel", "Marriott Hotels", True),
    ("Whole Foods", "Whole Foods Market", True), ("hilton", "Marriott Hotels", False), ("google ads", "Facebook Ads", False),
    ("united", "Southwest Airlines", False), ("", "anything", True),
])
def test_merchant_matching_directly(asked, merchant, fits):
    from agent.credit_lookup import merchant_fits

    assert merchant_fits(asked, merchant) is fits
