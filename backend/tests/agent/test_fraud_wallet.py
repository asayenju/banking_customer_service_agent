"""Slices 1, 3 and 4 of the fraud composer: the wallet workflow's rules, and that it is wired in."""

import asyncio
import json
from datetime import date

from agent.dispatcher import FIXED, Name, RecordKinds, SubAgentDispatcher, unmapped_kinds
from agent.fraud_report import FRAUD_TASKS, fraud_task_ids, run_fraud_report
from agent.guard import allowed_writes
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist
from agent.specialists.fraud import FRAUD_SPECIALIST, VALIDATORS
from agent.tier1 import gold_steps
from agent.toolbox import SecretVault
from agent.workflows.base import HandOff, NeedInfo, Plan, Refuse, Reply, WorkflowContext
from agent.workflows.fraud_wallet import WalletFacts, WalletLoss
from agent.workflows.gold_facts import facts_from_gold
from banking_tools import BankingSession
from banking_tools.catalog import get_registry
from banking_tools.manifest import load_tasks
from banking_tools.session import _base_db
from banking_tools.task_state import apply_initial_state
from banking_tools.types import ToolKind
from knowledge.index import Document, KnowledgeIndex
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.executor import OutcomeStatus
from orchestrator.types import Intent, IntentKind as K

TODAY = date(2025, 11, 14)
TASKS = {t["id"]: t for t in load_tasks()}
INDEX = KnowledgeIndex({"doc_x": Document("doc_x", "Internal: x", "lost card protocol")})
NOBODY = {"credit_answer": "none_were_in_the_wallet", "credit_cards": [], "credit_disputes": []}


def db_of(task: str):
    return apply_initial_state(_base_db().model_copy(deep=True), TASKS[task])


def card(card_id: str, replacement=None, accepts=True):
    return {"card_id": card_id, "accepts_refund_to_checking": accepts, "replacement": replacement}


def facts(**over):
    base = {"reason": "lost", "loss_is_definite": True, "debit_cards": [], **NOBODY}
    base.update(over)
    return WalletFacts.model_validate(base)


def decide(task: str, user: str, db=None, **over):
    return WalletLoss().decide(WorkflowContext(db or db_of(task), user, TODAY), facts(**over))


def tools(plan: Plan):
    return [(s.tool, s.args.get("card_id") or s.args.get("account_id") or s.args.get("credit_card_account_id")) for s in plan.steps]


# ---------------------------------------------------------------- wiring (slice 1)

def test_the_composer_owns_the_fraud_kind_and_every_kind_still_has_an_owner():
    assert unmapped_kinds() == frozenset()
    assert FIXED[K.REPORT_FRAUD_OR_LOSS] is Name.FRAUD
    assert FRAUD_SPECIALIST.kinds == frozenset({K.REPORT_FRAUD_OR_LOSS})
    assert [w.name for w in FRAUD_SPECIALIST.workflows] == ["wallet_loss"]


def test_every_write_the_guard_lets_the_kind_make_has_a_validator():
    registry = get_registry()
    writes = {t for t in allowed_writes(K.REPORT_FRAUD_OR_LOSS) if registry[t].kind is ToolKind.WRITE}
    assert writes <= set(VALIDATORS), sorted(writes - set(VALIDATORS))


def test_the_dispatcher_hands_the_kind_to_the_composer_and_a_missing_one_fails_plainly():
    shared = BankingSession.from_task(TASKS["task_077"])
    llm = FakeToolLLM([AssistantTurn(tool_calls=(ToolCall(id="c1", name="finish", arguments_json=json.dumps({"message": "I can only guide you from here. Would you like me to connect you with a person?"})),), usage=Usage(input_tokens=1))])
    specialist = Specialist(FRAUD_SPECIALIST, llm=llm, index=INDEX, database=shared.database, vault=SecretVault())
    it = Intent(id="i1", kind=K.REPORT_FRAUD_OR_LOSS, summary="lost my wallet", slots={})

    def ctx() -> RouteContext:
        session = scoped(shared)(SessionRequest(it.id, mode="plan", approved=None))
        return RouteContext(intent_id=it.id, verified_user_id="lj82d4f1a9", session=session, mode="plan", approved=None)

    assert asyncio.run(SubAgentDispatcher({Name.FRAUD: specialist}, RecordKinds(shared.database)).run(it, ctx())).status is OutcomeStatus.DONE
    assert asyncio.run(SubAgentDispatcher({}, RecordKinds(shared.database)).run(it, ctx())).status is OutcomeStatus.FAILED


# ---------------------------------------------------------------- the debit half (slice 3)

def test_nothing_is_closed_until_the_loss_is_definite_and_the_credit_card_is_still_asked_about():
    asked = decide("task_077", "lj82d4f1a9", loss_is_definite=False, credit_answer="not_asked_yet",
                   debit_cards=[card("dbc_lj82d4f1a9_bluest")])
    assert isinstance(asked, NeedInfo) and [a.slot for a in asked.questions] == ["credit_cards_in_the_wallet"]
    waiting = decide("task_077", "lj82d4f1a9", loss_is_definite=False, debit_cards=[card("dbc_lj82d4f1a9_bluest")])
    assert isinstance(waiting, Reply) and "Nothing is closed" in waiting.text


def test_a_wallet_with_no_named_card_asks_which_and_lists_the_customers_own_cards():
    asked = decide("task_077", "lj82d4f1a9")
    assert isinstance(asked, NeedInfo) and asked.questions[0].slot == "cards_in_the_wallet"
    assert "Silver Rewards" in asked.questions[0].text and "dbc_" not in asked.questions[0].text


def test_a_card_with_pending_purchases_is_held_frozen_and_the_rest_are_closed():
    """Task 078: Light Green has two pending purchases, so it is frozen and left open; the others close."""
    plan = decide("task_078", "mc78a5b9d2", debit_cards=[card("dbc_mc78a5b9d2_lb"), card("dbc_mc78a5b9d2_green"), card("dbc_mc78a5b9d2_lg")])
    assert isinstance(plan, Plan)
    made = tools(plan)
    assert ("close_debit_card_4721", "dbc_mc78a5b9d2_lb") in made and ("close_debit_card_4721", "dbc_mc78a5b9d2_green") in made
    assert ("close_debit_card_4721", "dbc_mc78a5b9d2_lg") not in made
    assert ("freeze_debit_card_3892", "dbc_mc78a5b9d2_lg") in made, "an active card with pending purchases is frozen, not left live"
    assert "stays frozen" in plan.steps[0].note and "pending" in plan.steps[0].note


def test_a_pending_refund_waits_for_the_customers_accept_and_then_the_card_closes():
    cards = ["dbc_cr89a2b3c4_green"]
    asked = decide("task_079", "cr89a2b3c4", debit_cards=[card(c, accepts=False) for c in cards])
    assert isinstance(asked, NeedInfo) and asked.questions[0].slot == "accepts_refund_to_checking"
    plan = decide("task_079", "cr89a2b3c4", debit_cards=[card(c, accepts=True) for c in cards])
    assert isinstance(plan, Plan) and ("close_debit_card_4721", "dbc_cr89a2b3c4_green") in tools(plan)


def test_an_entry_account_is_closed_but_not_reordered_the_same_day():
    """The 48 hour wait (doc 029): Light Blue is Entry, so its card closes and the order is left for later."""
    order = {"delivery_option": "STANDARD", "card_design": "CLASSIC", "shipping_address": None, "accepts_excess_fee": True}
    plan = decide("task_079", "cr89a2b3c4", debit_cards=[card("dbc_cr89a2b3c4_lb", replacement=order)])
    assert isinstance(plan, Plan)
    assert [t for t, _ in tools(plan)] == ["close_debit_card_4721"]
    assert "48 hours" in plan.steps[0].note and "11/16/2025" in plan.steps[0].note


def test_a_level_with_no_tier_cannot_be_reordered_and_hands_over_rather_than_guessing():
    order = {"delivery_option": "STANDARD", "card_design": "CLASSIC", "shipping_address": None, "accepts_excess_fee": False}
    result = decide("task_077", "lj82d4f1a9", debit_cards=[card("dbc_538bfb9cba", replacement=order)])
    assert isinstance(result, HandOff)


def test_another_customers_card_and_a_listed_twice_card_are_refused():
    other = decide("task_077", "lj82d4f1a9", debit_cards=[card("dbc_mc78a5b9d2_lb")])
    assert isinstance(other, Refuse)
    twice = decide("task_077", "lj82d4f1a9", debit_cards=[card("dbc_lj82d4f1a9_bluest"), card("dbc_lj82d4f1a9_bluest")])
    assert isinstance(twice, Refuse)


def test_an_already_closed_card_is_noted_not_closed_again():
    db = db_of("task_078")
    db.debit_cards.data["dbc_mc78a5b9d2_lb_old1"]  # exists and is closed in the task data
    result = decide("task_078", "mc78a5b9d2", db=db, debit_cards=[card("dbc_mc78a5b9d2_lb_old1")])
    assert isinstance(result, Reply) and "already closed" in result.text


# ---------------------------------------------------------------- the credit half and one preview (slice 4)

def gold_facts(task: str):
    db = db_of(task)
    writes = [s for s in gold_steps(TASKS[task]) if s.kind == "write" and s.name not in {"freeze_debit_card_3892", "activate_debit_card_8292"}]
    db_frozen = db.model_copy(deep=True)
    for s in gold_steps(TASKS[task]):
        if s.name == "freeze_debit_card_3892":
            db_frozen.debit_cards.data[s.args["card_id"]]["status"] = "FROZEN"
        if s.name == "unfreeze_debit_card_3893" and s.args["card_id"] not in {w.args["card_id"] for w in writes if w.name == "close_debit_card_4721"}:
            db_frozen.debit_cards.data[s.args["card_id"]]["status"] = "ACTIVE"
    return db_frozen, facts_from_gold("wallet_loss", writes, db_frozen)


def test_one_plan_orders_the_steps_closes_then_orders_then_credit_replacements_then_disputes():
    db, form = gold_facts("task_080")
    plan = WalletLoss().decide(WorkflowContext(db, "tm92c4d7e8", TODAY), WalletFacts.model_validate(form))
    assert isinstance(plan, Plan)
    names = [s.tool for s in plan.steps]
    first = {n: names.index(n) for n in set(names)}
    assert first["unfreeze_debit_card_3893"] < first["close_debit_card_4721"] < first["order_debit_card_5739"]
    assert first["order_debit_card_5739"] < first["order_replacement_credit_card_7291"] < first["file_credit_card_transaction_dispute_4829"]
    assert names.count("order_replacement_credit_card_7291") == 2 and names.count("file_credit_card_transaction_dispute_4829") == 3
    assert all(s.note for s in plan.steps), "every step has a customer-facing line written by code"


def test_a_credit_card_that_is_already_replaced_is_skipped_with_a_reason_and_the_rest_goes_ahead():
    db, form = gold_facts("task_080")
    db.credit_card_accounts.data["cc_tm92c4d7e8_gold"]["status"] = "CLOSED"
    plan = WalletLoss().decide(WorkflowContext(db, "tm92c4d7e8", TODAY), WalletFacts.model_validate(form))
    assert isinstance(plan, Plan)
    assert [a for t, a in tools(plan) if t == "order_replacement_credit_card_7291"] == ["cc_tm92c4d7e8_eco"]
    assert "Not included" in plan.steps[0].note and "already closed" in plan.steps[0].note


def test_credit_answers_that_contradict_each_other_are_refused_or_asked():
    db, form = gold_facts("task_080")
    liar = WalletFacts.model_validate({**form, "credit_answer": "none_were_in_the_wallet"})
    assert isinstance(WalletLoss().decide(WorkflowContext(db, "tm92c4d7e8", TODAY), liar), Refuse)
    none_listed = WalletFacts.model_validate({**form, "credit_answer": "some_were_in_the_wallet", "credit_cards": [], "credit_disputes": []})
    asked = WalletLoss().decide(WorkflowContext(db, "tm92c4d7e8", TODAY), none_listed)
    assert isinstance(asked, NeedInfo) and "which_credit_cards" in [a.slot for a in asked.questions]


def test_after_the_plan_the_customer_hears_what_is_still_frozen_and_that_activation_is_next():
    db = db_of("task_078")
    for c in db.debit_cards.data.values():
        if c["user_id"] == "mc78a5b9d2" and c["status"] == "ACTIVE":
            c["status"] = "FROZEN"
    plan = WalletLoss().decide(WorkflowContext(db, "mc78a5b9d2", TODAY), facts(debit_cards=[card("dbc_mc78a5b9d2_lb"), card("dbc_mc78a5b9d2_lg")]))
    assert isinstance(plan, Plan)
    # Play the closing on the copy, as execution would, then ask for the closing note.
    for s in plan.steps:
        if s.tool == "close_debit_card_4721":
            db.debit_cards.data[s.args["card_id"]]["status"] = "CLOSED"
        if s.tool == "unfreeze_debit_card_3893":
            db.debit_cards.data[s.args["card_id"]]["status"] = "ACTIVE"
    text = WalletLoss().after(WorkflowContext(db, "mc78a5b9d2", TODAY), plan.steps)
    assert "Still frozen" in text and "no new card yet" in text


# ---------------------------------------------------------------- the replay (slices 3 and 4 evidence)

def test_the_five_wallet_tasks_replay_as_strict_passes_with_no_write_outside_a_plan():
    assert tuple(fraud_task_ids()) == FRAUD_TASKS
    report = run_fraud_report(INDEX)
    assert [r.status.value for r in report.rows] == ["PASS"] * 5, [(r.task_id, r.note) for r in report.rows]
    assert all(r.strict for r in report.rows) and report.bypasses == 0 and report.clean
