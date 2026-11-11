"""Slice 2 of the fraud composer: the cards a customer names are frozen first, then one yes covers the rest."""

import asyncio
import json

from agent.containment import WalletContainment, named_cards
from agent.dispatcher import Name, RecordKinds, SubAgentDispatcher
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.recipes.direct import DirectRunner
from agent.specialist import Specialist
from agent.specialists.fraud import FRAUD_SPECIALIST
from agent.toolbox import SecretVault
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex
from orchestrator.auth import CustomerRecord, ScriptedDirectory
from orchestrator.context import scoped
from orchestrator.decomposer import Interpretation, ScriptedDecomposer
from orchestrator.orchestrator import Orchestrator
from orchestrator.routes import Route
from orchestrator.scoring import IntentScore
from orchestrator.types import Decomposition, Intent, IntentKind as K

TASK = next(t for t in load_tasks() if t["id"] == "task_078")
USER = "mc78a5b9d2"
LB, GREEN, LG = "dbc_mc78a5b9d2_lb", "dbc_mc78a5b9d2_green", "dbc_mc78a5b9d2_lg"
INDEX = KnowledgeIndex({"doc_x": Document("doc_x", "Internal: x", "lost card protocol")})
RECORD = CustomerRecord(
    user_id=USER, date_of_birth="07/18/1991", email="meiling.chen@outlook.com",
    phone_number="619-555-0284", address="3245 Sunset Boulevard, San Diego, CA 92103",
)
CLAIMS = {"email": RECORD.email, "phone_number": RECORD.phone_number}


class Scores:
    async def score(self, message, decomposition):
        return [IntentScore.flat(i.id, 0.99) for i in decomposition.intents]


def wallet_form(cards, **over):
    form = {
        "reason": "lost", "loss_is_definite": True, "credit_answer": "none_were_in_the_wallet",
        "credit_cards": [], "credit_disputes": [],
        "debit_cards": [{"card_id": c, "accepts_refund_to_checking": True, "replacement": None} for c in cards],
    }
    form.update(over)
    return form


def form_turn(form):
    return AssistantTurn(tool_calls=(ToolCall(id="c1", name="start_wallet_loss", arguments_json=json.dumps(form)),), usage=Usage(input_tokens=1))


def status(shared, card):
    return shared.database.debit_cards.data[card]["status"]


def world(script, intents, *, containment=True, claims=CLAIMS, later=()):
    shared = BankingSession.from_task(TASK)
    spec = Specialist(FRAUD_SPECIALIST, llm=FakeToolLLM(script), index=INDEX, database=shared.database, vault=SecretVault())
    dispatcher = SubAgentDispatcher({Name.FRAUD: spec}, RecordKinds(shared.database))
    decomposer = ScriptedDecomposer(
        [Interpretation(decomposition=Decomposition(intents=tuple(intents)), claims=claims), *later]
    )
    orch = Orchestrator(
        decomposer=decomposer, scorer=Scores(), directory=ScriptedDirectory(RECORD),
        runners={Route.DIRECT: DirectRunner(shared.database), Route.SUB_AGENT: dispatcher},
        session_for=scoped(shared),
        containment=WalletContainment(shared.database) if containment else None,
    )
    return orch, shared


def report(slots):
    return Intent(id="w", kind=K.REPORT_FRAUD_OR_LOSS, summary="I lost my wallet", slots=slots)


CONFIRM = Interpretation(decomposition=Decomposition(intents=()), confirmed=("w",))
NAMED = {"card_ids": "Light Blue, Green and Light Green"}


def test_named_cards_are_frozen_first_then_one_plan_waits_for_the_yes():
    orch, shared = world([form_turn(wallet_form([LB, GREEN, LG]))], [report(NAMED)], later=[CONFIRM])
    first = asyncio.run(orch.handle_turn("m1", "I lost my wallet with three debit cards"))
    assert {status(shared, c) for c in (LB, GREEN, LG)} == {"FROZEN"}, "contained before the yes"
    assert "is now frozen" in first.reply.text and "Here is exactly what I will do" in first.reply.text
    assert [e.tool for e in shared.audit_log] == ["freeze_debit_card_3892"] * 3, "a freeze is the only thing done before the yes"
    second = asyncio.run(orch.handle_turn("m2", "yes"))
    assert (status(shared, LB), status(shared, GREEN), status(shared, LG)) == ("CLOSED", "CLOSED", "FROZEN")
    assert [o.status.value for o in second.outcomes] == ["done"]


def test_without_containment_nothing_changes_until_the_yes():
    orch, shared = world([form_turn(wallet_form([LB, GREEN, LG]))], [report(NAMED)], containment=False)
    asyncio.run(orch.handle_turn("m1", "I lost my wallet"))
    assert {status(shared, c) for c in (LB, GREEN, LG)} == {"ACTIVE"}


def test_no_card_named_freezes_nothing_and_asks_which():
    orch, shared = world([form_turn(wallet_form([]))], [report({})])
    first = asyncio.run(orch.handle_turn("m1", "I lost my wallet"))
    assert {status(shared, c) for c in (LB, GREEN, LG)} == {"ACTIVE"} and shared.audit_log == ()
    assert "Which cards were in the wallet?" in first.reply.text and "dbc_" not in first.reply.text


def test_an_unverified_caller_gets_nothing_frozen():
    wrong = {"email": "someone@else.com", "phone_number": "000-000-0000"}
    orch, shared = world([form_turn(wallet_form([LB]))], [report(NAMED)], claims=wrong)
    first = asyncio.run(orch.handle_turn("m1", "I lost my wallet"))
    assert {status(shared, c) for c in (LB, GREEN, LG)} == {"ACTIVE"} and shared.audit_log == ()
    assert "frozen" not in first.reply.text


def test_a_possible_loss_freezes_and_closes_nothing():
    orch, shared = world([form_turn(wallet_form([LB, GREEN], loss_is_definite=False))], [report({"card_ids": "Light Blue and Green"})])
    first = asyncio.run(orch.handle_turn("m1", "I think I might have lost my wallet"))
    assert (status(shared, LB), status(shared, GREEN), status(shared, LG)) == ("FROZEN", "FROZEN", "ACTIVE")
    assert "Nothing is closed" in first.reply.text and "Here is exactly what I will do" not in first.reply.text


def test_a_card_found_afterwards_is_unfrozen_on_its_own():
    unfreeze = Intent(id="u", kind=K.UNFREEZE_CARD, summary="found it", slots={"card_id": GREEN})
    orch, shared = world(
        [form_turn(wallet_form([LB, LG, GREEN]))], [report(NAMED)],
        later=[Interpretation(decomposition=Decomposition(intents=(unfreeze,)))],
    )
    asyncio.run(orch.handle_turn("m1", "I lost my wallet"))
    asyncio.run(orch.handle_turn("m2", "wait, I found my Green card"))
    assert status(shared, GREEN) == "ACTIVE" and status(shared, LB) == "FROZEN"


# ---------------------------------------------------------------- which cards the words name

def test_cards_are_named_by_id_last_four_or_account_name_and_only_the_customers_own():
    shared = BankingSession.from_task(TASK)
    db = shared.database
    db.debit_cards.data["dbc_stranger"] = {"card_id": "dbc_stranger", "account_id": "chk_x", "user_id": "someone_else", "status": "ACTIVE", "last_4_digits": "9999"}
    ids = lambda slots: [c["card_id"] for c in named_cards(db, USER, slots)]  # noqa: E731
    assert ids({"card_id": LB}) == [LB]
    assert ids({"card_id": "7391"}) == [GREEN], "the last four digits"
    assert ids({"cards": "my Green account card, and the Light Green"}) == [GREEN, LG], "an account name must match exactly one"
    assert ids({"card_id": "dbc_stranger"}) == [] and ids({"card_id": "9999"}) == [], "another customer's card never matches"
    assert ids({"cards": "the Blue one"}) == [], "a name that matches nothing is ignored"


def test_only_an_active_card_is_frozen_and_the_freeze_intents_carry_ids_not_words():
    shared = BankingSession.from_task(TASK)
    db = shared.database
    db.debit_cards.data[GREEN]["status"] = "FROZEN"
    db.debit_cards.data[LB]["status"] = "CLOSED"
    intents = WalletContainment(db).freezes(report({"card_ids": "Light Blue, Green and Light Green"}), USER)
    assert [(i.kind, i.slots) for i in intents] == [(K.FREEZE_CARD, {"card_id": LG})]
    assert WalletContainment(db).freezes(Intent(id="x", kind=K.FREEZE_CARD, summary="s", slots={"card_id": LG}), USER) == ()
