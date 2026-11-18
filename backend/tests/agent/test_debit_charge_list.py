"""A debit dispute that does not say which charge goes card first, then that card's charges as taps, built in code like the
credit card history: no model is involved until the customer's own words point at a charge."""

import asyncio
import json

from agent.assembly import build_conversation
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist
from agent.specialists.debit import DEBIT_SPECIALIST
from agent.toolbox import SecretVault
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.decomposer import Interpretation, ScriptedDecomposer
from orchestrator.executor import OutcomeStatus
from orchestrator.scoring import IntentScore
from orchestrator.types import Decomposition, Intent, IntentKind as K

VALERIA = "a7c2f85d91"  # task 084: two checking accounts with a debit card each, Evergreen is one
INDEX = KnowledgeIndex({"doc_x": Document("doc_x", "Internal: debit disputes", "dispute a debit card charge")})
TASK = next(t for t in load_tasks() if t["id"] == "task_084")


def call(name, **arguments):
    return ToolCall(id=f"c-{name}", name=name, arguments_json=json.dumps(arguments))


def turn(*calls):
    return AssistantTurn(tool_calls=tuple(calls), usage=Usage(input_tokens=10))


SEARCH = call("find_debit_transactions", card="", merchant="zzz", amount="", on_or_after="", on_or_before="", transaction_id="")


def run(slots, script=()):
    shared = BankingSession.from_task(TASK)
    llm = FakeToolLLM(list(script))  # an empty script fails loudly if a model is called at all
    spec = Specialist(DEBIT_SPECIALIST, llm=llm, index=INDEX, database=shared.database, vault=SecretVault())
    intent = Intent(id="x", kind=K.DISPUTE_TRANSACTION, summary="dispute a charge", slots=slots)
    ctx = RouteContext(intent_id="x", verified_user_id=VALERIA, session=scoped(shared)(SessionRequest("x")), mode="plan")
    return asyncio.run(spec.run(intent, ctx)), llm


def test_a_dispute_that_names_no_card_asks_which_card_first_without_a_model():
    outcome, llm = run({})
    assert outcome.status is OutcomeStatus.NEEDS_INFO and not llm.calls
    text = outcome.questions[0].text
    assert text.startswith("Which card do you mean?") and text.count("\n- ") == 2, "one tap line per debit card"


def test_naming_the_card_lists_that_cards_charges_with_a_dispute_button_each_without_a_model():
    outcome, llm = run({"card_id": "Evergreen"})
    assert outcome.status is OutcomeStatus.DONE and outcome.choices and not llm.calls
    assert all(c.id.startswith("dispute:") and c.action == "Dispute" and c.kind is K.DISPUTE_TRANSACTION for c in outcome.choices)
    assert all(c.slots["transaction_id"] and c.slots["account_id"] and c.slots["amount"].startswith(("$", "-$")) for c in outcome.choices)
    lines = {line.removeprefix("- ") for line in outcome.text.splitlines()}
    assert all(c.label in lines for c in outcome.choices), "every button belongs to a line the customer can read"
    assert len({c.slots["account_id"] for c in outcome.choices}) == 1, "one card's charges, never a mix"


def test_the_buttons_are_the_customers_own_charges():
    outcome, _ = run({"card_id": "Evergreen"})
    db = BankingSession.from_task(TASK).database
    owned = {a["account_id"] for a in db.accounts.data.values() if a.get("user_id") == VALERIA}
    assert all(db.bank_account_transaction_history.data[c.slots["transaction_id"]]["account_id"] in owned for c in outcome.choices)


def test_a_charge_the_customer_describes_goes_to_the_model_to_search_for():
    outcome, llm = run({"merchant": "Bella's Bistro"}, [turn(SEARCH), turn(call("ask_customer", details=["dispute_reason"], question="What happened?"))])
    assert llm.calls and outcome.status is OutcomeStatus.NEEDS_INFO and not outcome.choices


def test_when_the_search_found_nothing_and_the_model_asks_which_charge_the_card_question_or_list_comes_instead():
    outcome, _ = run({"merchant": "zzz"}, [turn(SEARCH), turn(call("ask_customer", details=["transaction"], question="Which charge?"))])
    assert outcome.status is OutcomeStatus.NEEDS_INFO and outcome.questions[0].text.startswith("Which card do you mean?")
    named, _ = run({"merchant": "zzz", "card_id": "Evergreen"}, [turn(SEARCH), turn(call("ask_customer", details=["transaction"], question="Which charge?"))])
    assert named.status is OutcomeStatus.DONE and named.choices


def test_asking_for_something_other_than_the_charge_stays_a_question():
    outcome, _ = run({"merchant": "zzz", "card_id": "Evergreen"}, [turn(SEARCH), turn(call("ask_customer", details=["dispute_reason"], question="Why?"))])
    assert outcome.status is OutcomeStatus.NEEDS_INFO and not outcome.choices


class Scores:
    async def score(self, message, decomposition):
        return [IntentScore.flat(i.id, 0.99) for i in decomposition.intents]


def test_pressing_one_makes_a_dispute_of_that_exact_charge_and_sends_it_to_the_debit_specialist():
    history = Interpretation(
        decomposition=Decomposition(
            intents=(Intent(id="h", kind=K.ACCOUNT_BALANCE_OR_HISTORY, summary="show my recent transactions", slots={"account_id": "Evergreen"}),)
        ),
        claims={"phone_number": "312-555-7294", "email": "valeria.montes@finmail.com"},
    )
    conv = build_conversation(
        decomposer=ScriptedDecomposer([history]), scorer=Scores(), llm=FakeToolLLM([]), index=INDEX, session=BankingSession.from_task(TASK)
    )
    first = asyncio.run(conv.orchestrator.handle_turn("m1", "show me my recent transactions"))
    assert first.reply.choices, "a checking account's history offers Dispute buttons"
    chosen = first.reply.choices[0]
    second = asyncio.run(conv.orchestrator.handle_turn("m2", f"Dispute: {chosen.label}", chosen.id))
    events = [(e.kind.value, e.detail) for e in second.trace.events]
    assert ("decided", "button") in events and ("scored", "button") in events, "made from the offer, not read by a model"
    assert ("routed", "sub_agent") in events
