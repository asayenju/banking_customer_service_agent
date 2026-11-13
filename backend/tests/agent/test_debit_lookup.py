"""A specialist finds a debit card's transactions itself instead of asking the customer for ids."""

import asyncio
import json
from datetime import date

import pytest

from agent.debit_lookup import FIND, NAMES, debit_lookups, find_transactions, list_cards
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist
from agent.specialists.debit import DEBIT_SPECIALIST
from agent.toolbox import SecretVault
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.executor import OutcomeStatus
from orchestrator.types import Intent, IntentKind as K

TODAY = date(2025, 11, 14)
MEI, VALERIA = "mc78a5b9d2", "a7c2f85d91"
INDEX = KnowledgeIndex({"doc_x": Document("doc_x", "Internal: debit disputes", "dispute a debit card charge")})
TASKS = {t["id"]: t for t in load_tasks()}


def world(task="task_078"):
    return BankingSession.from_task(TASKS[task])


def find(task, user, **said):
    base = dict(card="", merchant="", amount="", on_or_after="", on_or_before="", transaction_id="")
    return find_transactions(world(task).database, user, TODAY, **{**base, **said})


def ids(found):
    return [r["transaction_id"] for r in found["matches"]]


# ---------------------------------------------------------------- naming the card

@pytest.mark.parametrize("card", ["dbc_mc78a5b9d2_green", "7391", "Green", "my Green Account card", "chk_mc78a5b9d2_2"])
def test_a_card_is_found_by_its_id_its_last_four_digits_or_its_account_however_it_is_named(card):
    found = find("task_078", MEI, card=card)
    assert found["matches"] and {r["account"] for r in found["matches"]} == {"Green Account"}
    assert {r["card_last_4"] for r in found["matches"]} == {"7391"}


def test_a_closed_cards_last_four_digits_lead_to_the_same_account_as_its_replacement():
    old = find("task_078", MEI, card="2193")
    assert {r["account"] for r in old["matches"]} == {"Light Blue Account"}


def test_with_no_card_named_every_account_is_searched():
    everything = find("task_078", MEI)
    assert everything["searched"] == "every account" and len({r["account"] for r in everything["matches"]}) == 3


def test_a_card_that_is_not_the_customers_says_so_and_points_to_the_list():
    found = find("task_078", MEI, card="9999")
    assert found["matches"] == [] and "list_debit_cards" in found["tip"]
    assert find("task_078", MEI, card="dbc_f4a72c9e55")["matches"] == [], "another customer's card id"


def test_another_customers_transactions_never_come_back():
    assert find("task_084", MEI, merchant="bistro")["matches"] == []
    assert ids(find("task_084", VALERIA, merchant="bistro")) == ["btxn_d7f2a98c1b34", "btxn_e8c3b09d2c45"]


# ---------------------------------------------------------------- what was bought

@pytest.mark.parametrize("merchant", ["Cheesecake Factory", "cheesecake", "the cheesecake factory charge", "cheesecake factry"])
def test_the_merchant_is_found_however_it_is_spelled(merchant):
    assert ids(find("task_078", MEI, merchant=merchant)) == ["btxn_c9cbc7bd9454"]


@pytest.mark.parametrize("amount", ["87.34", "$87.34", "around 87", "about 90"])
def test_the_amount_is_found_exactly_or_roughly(amount):
    assert "btxn_c9cbc7bd9454" in ids(find("task_078", MEI, amount=amount))


def test_a_wrong_amount_or_a_wrong_card_does_not_match():
    assert "btxn_c9cbc7bd9454" not in ids(find("task_078", MEI, amount="$300", merchant="cheesecake"))
    assert find("task_078", MEI, card="Light Blue", merchant="cheesecake")["matches"] == [], "it was on the Green account"


@pytest.mark.parametrize(
    ("after", "before"),
    [("11/06/25", "11/06/25"), ("11/06/2025", "11/06/2025"), ("November 6th 2025", "November 6th 2025"), ("11/01/2025", ""), ("", "11/14/2025")],
)
def test_the_date_filters_find_it_in_any_spelling(after, before):
    assert "btxn_c9cbc7bd9454" in ids(find("task_078", MEI, merchant="cheesecake", on_or_after=after, on_or_before=before))


def test_a_date_range_that_excludes_the_day_finds_nothing_and_the_merchant_only_rows_are_offered():
    found = find("task_078", MEI, merchant="cheesecake", on_or_after="11/08/2025")
    assert found["matches"] == [] and [r["transaction_id"] for r in found["merchant_only"]] == ["btxn_c9cbc7bd9454"]


def test_a_transaction_id_is_looked_up_directly_and_only_among_the_customers_own():
    assert ids(find("task_078", MEI, transaction_id="btxn_c9cbc7bd9454")) == ["btxn_c9cbc7bd9454"]
    assert find("task_078", MEI, transaction_id="btxn_d7f2a98c1b34")["matches"] == []


def test_pending_charges_show_their_status_and_the_card_they_were_on():
    pending = find("task_078", MEI, merchant="amazon")["matches"][0]
    assert pending["status"] == "pending" and pending["account"] == "Light Green Account" and pending["card_last_4"] == "8624"


def test_a_date_with_a_time_on_it_is_still_read():
    db = world("task_078").database
    db.bank_account_transaction_history.data["btxn_c9cbc7bd9454"]["date"] = "11/06/2025 14:22:10"
    found = find_transactions(db, MEI, TODAY, "", "cheesecake", "", "11/06/2025", "11/06/2025", "")
    assert ids(found) == ["btxn_c9cbc7bd9454"]


# ---------------------------------------------------------------- the card list and the tool as the model calls it

def test_the_cards_are_listed_with_last_four_digits_status_and_account_for_this_customer_only():
    cards = list_cards(world("task_078").database, MEI)
    assert {(c["last_4_digits"], c["status"], c["account"]) for c in cards} >= {("4582", "ACTIVE", "Light Blue Account"), ("2193", "CLOSED", "Light Blue Account")}
    assert list_cards(world("task_078").database, "nobody") == []


def test_the_search_tool_takes_every_argument_and_returns_json():
    tool = next(t for t in debit_lookups(world("task_078").database, MEI, TODAY) if t.schema.name == FIND)
    assert set(tool.schema.parameters["required"]) == set(tool.schema.parameters["properties"]) == {
        "card", "merchant", "amount", "on_or_after", "on_or_before", "transaction_id",
    }
    text = tool.run({"card": "7391", "merchant": "cheesecake", "amount": "", "on_or_after": "", "on_or_before": "", "transaction_id": ""})
    assert json.loads(text)["matches"][0]["transaction_id"] == "btxn_c9cbc7bd9454"


# ---------------------------------------------------------------- the rule: look before you ask

def call(name, **arguments):
    return ToolCall(id=f"c-{name}-{len(arguments)}", name=name, arguments_json=json.dumps(arguments))


def turn(*calls):
    return AssistantTurn(tool_calls=tuple(calls), usage=Usage(input_tokens=10))


def run(script, kind=K.DISPUTE_TRANSACTION):
    shared = world("task_078")
    llm = FakeToolLLM(script)
    spec = Specialist(DEBIT_SPECIALIST, llm=llm, index=INDEX, database=shared.database, vault=SecretVault())
    intent = Intent(id="x", kind=kind, summary="dispute the cheesecake charge", slots={})
    ctx = RouteContext(intent_id="x", verified_user_id=MEI, session=scoped(shared)(SessionRequest("x")), mode="plan")
    return asyncio.run(spec.run(intent, ctx)), llm


def test_the_debit_specialist_is_offered_the_search_tools():
    _, llm = run([turn(call("ask_customer", details=["x"]))])
    assert NAMES <= set(llm.calls[0]["tools"])


def test_asking_a_debit_customer_about_a_charge_before_any_search_is_refused():
    outcome, llm = run(
        [
            turn(call("ask_customer", details=["dispute_reason"], question="What happened?")),
            turn(call("find_debit_transactions", card="", merchant="cheesecake", amount="", on_or_after="", on_or_before="", transaction_id="")),
            turn(call("ask_customer", details=["dispute_reason"], question="What went wrong with it?")),
        ]
    )
    assert "look it up first" in llm.calls[1]["messages"][-1]["content"]
    assert outcome.status is OutcomeStatus.NEEDS_INFO and outcome.questions[0].slot == "dispute_reason"
