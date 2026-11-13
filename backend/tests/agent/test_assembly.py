"""Every specialist is connected to the orchestrator: one wiring, used by tests and by the app."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import main
from agent.assembly import SPECIALISTS, DatabaseDirectory, build_conversation
from agent.dispatcher import BY_RECORD, FIXED, Name, sub_agent_kinds, unmapped_kinds
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex
from orchestrator.decomposer import Interpretation, ScriptedDecomposer
from orchestrator.scoring import IntentScore
from orchestrator.types import Decomposition, Intent, IntentKind as K

TASK = next(t for t in load_tasks() if t["id"] == "task_078")
INDEX = KnowledgeIndex({"doc_x": Document("doc_x", "Internal: x", "lost card protocol")})
USER = "mc78a5b9d2"
LB, GREEN, LG = "dbc_mc78a5b9d2_lb", "dbc_mc78a5b9d2_green", "dbc_mc78a5b9d2_lg"
CLAIMS = {"email": "meiling.chen@outlook.com", "phone_number": "619-555-0284"}


class Scores:
    async def score(self, message, decomposition):
        return [IntentScore.flat(i.id, 0.99) for i in decomposition.intents]


def conversation(script_llm=(), script_decomposer=()):
    return build_conversation(
        decomposer=ScriptedDecomposer(list(script_decomposer)),
        scorer=Scores(),
        llm=FakeToolLLM(list(script_llm)),
        index=INDEX,
        session=BankingSession.from_task(TASK),
    )


def test_every_specialist_name_has_a_configuration_and_a_running_specialist():
    assert set(SPECIALISTS) == set(Name)
    conv = conversation()
    assert set(conv.specialists) == set(Name)
    assert all(s.config is SPECIALISTS[n] for n, s in conv.specialists.items())


def test_every_sub_agent_kind_reaches_a_specialist_that_handles_it():
    assert unmapped_kinds() == frozenset()
    wallet = next(t for t in load_tasks() if t["id"] == "task_080")  # one customer with debit and credit cards
    conv = build_conversation(
        decomposer=ScriptedDecomposer([]), scorer=Scores(), llm=FakeToolLLM([]), index=INDEX,
        session=BankingSession.from_task(wallet),
    )
    for kind in sub_agent_kinds():
        if kind is K.CREDIT_LIMIT_REQUEST:
            slots = {"credit_card_account_id": "cc_tm92c4d7e8_gold"}
        else:
            slots = {"card_id": "dbc_tm92c4d7e8_blue"} if kind in BY_RECORD else {}
        intent = Intent(id="i", kind=kind, summary="s", slots=slots)
        chosen = conv.dispatcher.specialist_name(intent, "tm92c4d7e8")
        assert isinstance(chosen, Name), (kind, chosen)
        assert kind in conv.specialists[chosen].config.kinds, (kind, chosen)
    assert FIXED[K.REPORT_FRAUD_OR_LOSS] is Name.FRAUD and FIXED[K.FEE_OR_INTEREST_ADJUSTMENT] is Name.FEES
    assert FIXED[K.OPEN_ACCOUNT] is Name.ACCOUNTS and FIXED[K.GENERAL_QUESTION] is Name.GENERAL


def test_the_directory_finds_a_customer_by_email_phone_or_id_and_nobody_else():
    db = BankingSession.from_task(TASK).database
    directory = DatabaseDirectory(db)
    find = lambda claims: asyncio.run(directory.find(claims))  # noqa: E731
    assert find({"email": "Meiling.Chen@outlook.com"}).user_id == USER
    assert find({"phone_number": "(619) 555-0284"}).user_id == USER
    assert find({"user_id": USER.upper()}).user_id == USER
    assert find({"email": "nobody@example.com"}) is None and find({"date_of_birth": "07/18/1991"}) is None


def wallet_form(cards):
    return {
        "reason": "lost", "loss_is_definite": True, "credit_answer": "none_were_in_the_wallet",
        "credit_cards": [], "credit_disputes": [],
        "debit_cards": [{"card_id": c, "accepts_refund_to_checking": True, "replacement": None} for c in cards],
    }


def test_a_lost_wallet_goes_through_the_orchestrator_the_fraud_specialist_and_the_direct_freeze():
    report = Intent(id="w", kind=K.REPORT_FRAUD_OR_LOSS, summary="lost wallet", slots={"card_ids": "Light Blue and Green"})
    form = AssistantTurn(tool_calls=(ToolCall(id="c", name="start_wallet_loss", arguments_json=json.dumps(wallet_form([LB, GREEN]))),), usage=Usage(input_tokens=1))
    conv = conversation(
        [form],
        [
            Interpretation(decomposition=Decomposition(intents=(report,)), claims=CLAIMS),
            Interpretation(decomposition=Decomposition(intents=()), confirmed=("w",)),
        ],
    )
    first = asyncio.run(conv.orchestrator.handle_turn("m1", "I lost my wallet"))
    status = lambda c: conv.session.database.debit_cards.data[c]["status"]  # noqa: E731
    assert (status(LB), status(GREEN)) == ("FROZEN", "FROZEN") and status(LG) == "ACTIVE" and "Here is exactly what I will do" in first.reply.text
    asyncio.run(conv.orchestrator.handle_turn("m2", "yes"))
    assert (status(LB), status(GREEN), status(LG)) == ("CLOSED", "CLOSED", "ACTIVE")


def test_a_debit_request_and_a_fee_request_in_one_conversation_each_reach_their_own_specialist():
    pin = Intent(id="a", kind=K.CARD_CONTROLS, summary="card controls", slots={"card_id": GREEN})
    fee = Intent(id="b", kind=K.FEE_OR_INTEREST_ADJUSTMENT, summary="an atm fee", slots={})
    ask = lambda name: AssistantTurn(tool_calls=(ToolCall(id=name, name="ask_customer", arguments_json=json.dumps({"details": [name]})),), usage=Usage(input_tokens=1))  # noqa: E731
    conv = conversation(
        [ask("debit_detail"), ask("fee_detail")],
        [Interpretation(decomposition=Decomposition(intents=(pin, fee)), claims=CLAIMS)],
    )
    result = asyncio.run(conv.orchestrator.handle_turn("m1", "two things"))
    asked = {d.intent_id: [q.slot for q in d.questions] for d in result.decisions}
    assert asked["a"] == ["debit_detail"] and asked["b"] == ["fee_detail"], asked
    assert not any("not implemented" in (o.error or "") for o in result.outcomes)


def test_a_pin_typed_in_a_message_never_reaches_the_model():
    pin_value = "4821"
    intent = Intent(id="p", kind=K.CHANGE_OR_RESET_PIN, summary="new pin", slots={"card_id": GREEN, "new_pin": "{{secret:pin}}"})
    llm = FakeToolLLM([AssistantTurn(tool_calls=(ToolCall(id="x", name="ask_customer", arguments_json=json.dumps({"details": ["last_4_digits"]})),), usage=Usage(input_tokens=1))])
    conv = build_conversation(
        decomposer=ScriptedDecomposer([Interpretation(decomposition=Decomposition(intents=(intent,)), claims=CLAIMS)]),
        scorer=Scores(), llm=llm, index=INDEX, session=BankingSession.from_task(TASK),
    )
    asyncio.run(conv.orchestrator.handle_turn("m1", f"set my new pin to {pin_value}"))
    assert pin_value not in json.dumps(llm.calls, default=str)


# ---------------------------------------------------------------- the HTTP face

def test_the_api_runs_a_conversation_through_the_wired_orchestrator_and_remembers_it():
    intent = Intent(id="g", kind=K.UNCLEAR_OR_GREETING, summary="hello")
    built = []

    def factory():
        conv = conversation([], [])
        built.append(conv)
        return conv

    main.app.state.factory = factory
    main.app.state.conversations = {}
    client = TestClient(main.app)
    first = client.post("/conversations/c1/messages", json={"message_id": "m1", "message": "hello"})
    assert first.status_code == 200 and "help" in first.json()["reply"].lower() and not first.json()["escalated"]
    again = client.post("/conversations/c1/messages", json={"message_id": "m1", "message": "hello"})
    assert again.json() == first.json(), "a message id handled once is not handled twice"
    client.post("/conversations/c2/messages", json={"message_id": "m1", "message": "hello"})
    assert len(built) == 2, "one conversation per id, created on its first message"
    assert intent


def test_the_demo_page_is_served_and_a_scenario_picks_the_sample_customer():
    asked = []

    def scenario_factory(task_id):
        asked.append(task_id)
        return conversation([], [])

    main.app.state.scenario_factory = scenario_factory
    main.app.state.conversations = {}
    client = TestClient(main.app)
    page = client.get("/demo")
    assert page.status_code == 200 and "Rho-Bank assistant" in page.text
    ok = client.post("/conversations/s1/messages", json={"message_id": "m1", "message": "hello", "scenario": "wallet"})
    assert ok.status_code == 200 and asked == ["task_078"]
    client.post("/conversations/s1/messages", json={"message_id": "m2", "message": "hello again", "scenario": "dispute"})
    assert asked == ["task_078"], "the scenario is read only on the first message"
    bad = client.post("/conversations/s2/messages", json={"message_id": "m1", "message": "hello", "scenario": "nope"})
    assert bad.status_code == 422
