import asyncio
import json

import pytest

from agent.dispatcher import Name, RecordKinds, SubAgentDispatcher
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist, ValidationContext
from agent.specialists import debit
from dataclasses import replace

from agent.specialists.debit import DEBIT_SPECIALIST as WITH_WORKFLOWS, prepare

# These tests are about the validators and the guard, which the free-form plan path still uses.
DEBIT_SPECIALIST = replace(WITH_WORKFLOWS, workflows=())
from agent.toolbox import SecretVault
from banking_tools import BankingSession, Fidelity
from datetime import date
from knowledge.index import Document, KnowledgeIndex
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.types import Intent, IntentKind as K

ME = "lj82d4f1a9"
CARD, ACCOUNT = "dbc_lj82d4f1a9_bluest", "chk_lj82d4f1a9"
TWO = "538bfb9cba"
ONE = "125"  # owns exactly one debit card
MANY = "kj93a7b2e1"
MANY_CARDS = ["dbc_kj93a7b2e1_blue", "dbc_kj93a7b2e1_green", "dbc_kj93a7b2e1_lightgreen"]
INDEX = KnowledgeIndex({"doc_bank_accounts_x": Document("doc_bank_accounts_x", "Internal: x", "debit card replacement procedure")})


def fresh():
    return BankingSession(fidelity=Fidelity.TAU2)


def v(shared, user=ME):
    return ValidationContext(shared.database, user, date(2025, 11, 14))


def card(shared, card_id=CARD, **changes):
    shared.database.debit_cards.data[card_id].update(changes)


def call(name, **arguments):
    return ToolCall(id=f"c-{name}", name=name, arguments_json=json.dumps(arguments))


def turn(*calls):
    return AssistantTurn(tool_calls=tuple(calls), usage=Usage(input_tokens=50))


def plan(*steps):
    return call("propose_plan", steps=[{"tool": t, "arguments_json": json.dumps(a), "description": "x"} for t, a in steps])


def specialist(script, shared):
    return Specialist(DEBIT_SPECIALIST, llm=FakeToolLLM(script), index=INDEX, database=shared.database, vault=SecretVault())


def ctx(shared, intent, user=ME, mode="plan", approved=None):
    session = scoped(shared)(SessionRequest(intent.id, mode=mode, approved=approved))
    return RouteContext(intent_id=intent.id, verified_user_id=user, session=session, mode=mode, approved=approved)


def run(spec, intent, context):
    return asyncio.run(spec.run(intent, context))


# ------------------------------------------------------------------ status validators

@pytest.mark.parametrize(
    "validator, good, bad",
    [
        (debit.freeze, "ACTIVE", ["FROZEN", "CLOSED", "PENDING"]),
        (debit.unfreeze, "FROZEN", ["ACTIVE", "CLOSED"]),
    ],
)
def test_freeze_and_unfreeze_accept_only_the_right_status(validator, good, bad):
    for status, ok in [(good, True)] + [(b, False) for b in bad]:
        shared = fresh()
        card(shared, status=status)
        assert (validator(v(shared), {"card_id": CARD}) is None) is ok, status


def test_unfreezing_is_refused_when_the_account_behind_the_card_is_closed():
    shared = fresh()
    card(shared, status="FROZEN")
    shared.database.accounts.data[ACCOUNT]["status"] = "CLOSED"
    assert "not open" in debit.unfreeze(v(shared), {"card_id": CARD})


@pytest.mark.parametrize("status, ok", [("ACTIVE", True), ("PENDING", True), ("FROZEN", False), ("CLOSED", False)])
def test_closing_needs_an_active_or_pending_card_and_a_frozen_one_must_be_unfrozen_first(status, ok):
    shared = fresh()
    card(shared, status=status)
    result = debit.close(v(shared), {"card_id": CARD, "reason": "lost"})
    assert (result is None) is ok
    if status == "FROZEN":
        assert "unfrozen before" in result


@pytest.mark.parametrize("reason, ok", [("lost", True), ("STOLEN", True), ("fraud_suspected", True), ("because", False), ("", False)])
def test_closing_needs_a_reason_the_bank_recognises(reason, ok):
    assert (debit.close(v(fresh()), {"card_id": CARD, "reason": reason}) is None) is ok


def test_an_unknown_card_is_refused_by_every_lifecycle_validator():
    shared = fresh()
    for check in (debit.freeze, debit.unfreeze, debit.close, debit.activate("activate_debit_card_8291")):
        assert "does not exist" in check(v(shared), {"card_id": "dbc_made_up", "reason": "lost"})


# ------------------------------------------------------------------ activation variants

@pytest.mark.parametrize(
    "reason, tool",
    [
        ("new_account", "activate_debit_card_8291"),
        ("first_card", "activate_debit_card_8291"),
        ("lost", "activate_debit_card_8292"),
        ("stolen", "activate_debit_card_8292"),
        ("fraud", "activate_debit_card_8292"),
        ("expired", "activate_debit_card_8293"),
        ("damaged", "activate_debit_card_8293"),
        ("upgrade", "activate_debit_card_8293"),
        ("bank_reissue", "activate_debit_card_8293"),
    ],
)
def test_the_activation_tool_must_fit_the_cards_issue_reason(reason, tool):
    shared = fresh()
    card(shared, status="PENDING", issue_reason=reason)
    for name in debit.ACTIVATION_REASONS:
        result = debit.activate(name)(v(shared), {"card_id": CARD})
        assert (result is None) is (name == tool), (reason, name)
        if name != tool:
            assert tool in result, "the refusal names the right tool"


def test_only_a_card_waiting_to_be_activated_can_be_activated():
    shared = fresh()
    assert "waiting to be activated" in debit.activate("activate_debit_card_8291")(v(shared), {"card_id": CARD})


# ------------------------------------------------------------------ ordering and fees

ORDER = {"account_id": ACCOUNT, "user_id": ME, "delivery_option": "STANDARD", "delivery_fee": 0.0, "card_design": "CLASSIC", "design_fee": 0.0, "shipping_address": "1 Main St", "excess_replacement_fee": None}


def closed(shared):
    card(shared, status="CLOSED")
    return shared


def test_ordering_is_fine_once_the_old_card_is_closed_and_not_before():
    assert "close it first" in debit.order(v(fresh()), ORDER)
    assert debit.order(v(closed(fresh())), ORDER) is None


@pytest.mark.parametrize(
    "change, ok",
    [
        ({}, True),
        ({"delivery_option": "EXPEDITED", "delivery_fee": 15.0}, True),
        ({"delivery_option": "EXPEDITED", "delivery_fee": 0.0}, True),
        ({"delivery_option": "RUSH", "delivery_fee": 35.0}, True),
        ({"card_design": "PREMIUM", "design_fee": 10.0}, True),
        ({"card_design": "CUSTOM", "design_fee": 15.0}, True),
        ({"excess_replacement_fee": 25.0}, True),
        ({"delivery_fee": 15.0}, False),
        ({"delivery_option": "RUSH", "delivery_fee": 15.0}, False),
        ({"delivery_option": "EXPEDITED", "delivery_fee": 50.0}, False),
        ({"card_design": "CLASSIC", "design_fee": 10.0}, False),
        ({"card_design": "PREMIUM", "design_fee": 99.0}, False),
        ({"excess_replacement_fee": 7.0}, False),
        ({"delivery_option": "TELEPORT"}, False),
        ({"card_design": "GOLD"}, False),
    ],
)
def test_only_published_fee_values_for_the_chosen_option_are_accepted(change, ok):
    assert (debit.order(v(closed(fresh())), {**ORDER, **change}) is None) is ok


def test_a_card_can_only_be_ordered_for_an_open_checking_account():
    shared = closed(fresh())
    shared.database.accounts.data[ACCOUNT]["status"] = "CLOSED"
    assert "not open" in debit.order(v(shared), ORDER)
    assert "does not exist" in debit.order(v(shared), {**ORDER, "account_id": "nope"})


# ------------------------------------------------------------------ plans through the specialist

def replacing(**slots):
    return Intent(id="x", kind=K.ORDER_REPLACEMENT_CARD, summary="replace my card", slots=slots or {"card_id": CARD})


def test_an_unfreeze_close_order_plan_is_accepted_in_that_order_and_refused_in_any_other():
    shared = fresh()
    card(shared, status="FROZEN")
    steps = [
        ("unfreeze_debit_card_3893", {"card_id": CARD}),
        ("close_debit_card_4721", {"card_id": CARD, "reason": "lost"}),
        ("order_debit_card_5739", ORDER),
    ]
    good = run(specialist([turn(plan(*steps))], shared), replacing(), ctx(shared, replacing()))
    assert good.status is OutcomeStatus.NEEDS_CONFIRMATION and [s.tool for s in good.proposal.steps] == [t for t, _ in steps]
    for bad in ([steps[1], steps[0], steps[2]], steps[1:], [steps[0], steps[2], steps[1]]):
        outcome = run(specialist([turn(plan(*bad))], shared), replacing(), ctx(shared, replacing()))
        assert outcome.status is OutcomeStatus.FAILED, [t for t, _ in bad]
    assert shared.audit_log == ()


def test_a_frozen_card_cannot_be_closed_without_unfreezing_it_first():
    shared = fresh()
    card(shared, status="FROZEN")
    intent = Intent(id="x", kind=K.CLOSE_CARD, summary="close it", slots={"card_id": CARD})
    outcome = run(specialist([turn(plan(("close_debit_card_4721", {"card_id": CARD, "reason": "lost"})))], shared), intent, ctx(shared, intent))
    assert outcome.status is OutcomeStatus.FAILED and "unfrozen before" in outcome.error


def test_the_whole_approved_plan_runs_and_leaves_the_world_as_the_customer_agreed():
    shared = fresh()
    card(shared, status="FROZEN")
    steps = [("unfreeze_debit_card_3893", {"card_id": CARD}), ("close_debit_card_4721", {"card_id": CARD, "reason": "lost"}), ("order_debit_card_5739", ORDER)]
    spec = specialist([turn(plan(*steps))], shared)
    proposal = run(spec, replacing(), ctx(shared, replacing())).proposal
    outcome = run(spec, replacing(), ctx(shared, replacing(), mode="execute", approved=proposal))
    assert outcome.status is OutcomeStatus.DONE
    assert shared.database.debit_cards.data[CARD]["status"] == "CLOSED"
    assert any(c["status"] == "PENDING" for c in shared.database.debit_cards.data.values() if c["account_id"] == ACCOUNT and c["card_id"] != CARD)


# ------------------------------------------------------------------ which card? (multi-card ambiguity)

def closing(**slots):
    return Intent(id="x", kind=K.CLOSE_CARD, summary="close my card", slots=slots)


def test_a_customer_with_several_cards_who_names_none_is_asked_before_any_model_runs():
    shared = fresh()
    llm = FakeToolLLM([])
    spec = Specialist(DEBIT_SPECIALIST, llm=llm, index=INDEX, database=shared.database, vault=SecretVault())
    outcome = run(spec, closing(), ctx(shared, closing(), user=MANY))
    assert outcome.status is OutcomeStatus.NEEDS_INFO and llm.calls == [], "no model was called"
    (question,) = outcome.questions
    assert question.slot == "card_id" and question.text.startswith("Which card do you mean?")
    assert question.text.count("the card ending") == 3
    for card_id in MANY_CARDS:
        assert card_id not in question.text, "ids are never shown"
    assert "4111" not in question.text


def test_a_customer_with_one_card_is_not_asked():
    shared = fresh()
    spec = specialist([turn(call("finish", message="ok"))], shared)
    assert run(spec, closing(), ctx(shared, closing(), user=ONE)).status is OutcomeStatus.DONE


def test_a_customer_with_two_cards_is_asked_too():
    shared = fresh()
    outcome = run(specialist([], shared), closing(), ctx(shared, closing()))
    assert outcome.status is OutcomeStatus.NEEDS_INFO and outcome.questions[0].text.count("the card ending") == 2


def test_naming_an_account_or_a_transaction_is_enough_for_the_model_to_find_the_card():
    shared = fresh()
    spec = specialist([turn(call("finish", message="ok"))], shared)
    outcome = run(spec, closing(account_id="chk_kj93a7b2e1_1"), ctx(shared, closing(), user=MANY))
    assert outcome.status is OutcomeStatus.DONE


def test_closed_cards_are_not_offered():
    shared = fresh()
    card(shared, "dbc_kj93a7b2e1_blue", status="CLOSED")
    outcome = prepare(closing(), MANY, shared.database)
    assert isinstance(outcome, RouteOutcome) and outcome.questions[0].text.count("the card ending") == 2


def test_answering_with_the_last_four_digits_picks_the_card_in_code():
    shared = fresh()
    last4 = shared.database.debit_cards.data["dbc_kj93a7b2e1_green"]["last_4_digits"]
    working = prepare(closing(card_id=last4), MANY, shared.database)
    assert isinstance(working, Intent) and working.slots["card_id"] == "dbc_kj93a7b2e1_green"


def test_a_last_four_that_matches_nothing_or_two_cards_asks_again_with_the_list():
    shared = fresh()
    assert isinstance(prepare(closing(card_id="0000"), MANY, shared.database), RouteOutcome)
    a, b = MANY_CARDS[:2]
    shared.database.debit_cards.data[a]["last_4_digits"] = shared.database.debit_cards.data[b]["last_4_digits"] = "1111"
    outcome = prepare(closing(card_id="1111"), MANY, shared.database)
    assert isinstance(outcome, RouteOutcome) and outcome.questions[0].text.count("the card ending") == 2


def test_someone_elses_card_is_never_offered_or_accepted():
    shared = fresh()
    theirs = "dbc_kj93a7b2e1_blue"
    outcome = prepare(closing(card_id=theirs), ONE, shared.database)
    assert isinstance(outcome, RouteOutcome), "an id that is not the customer's is treated as not found: ask again"
    text = outcome.questions[0].text
    assert text.count("the card ending") == 1 and theirs not in text
    assert shared.database.debit_cards.data[theirs]["last_4_digits"] not in text
    # and if a plan names it anyway, the guard refuses it
    spec = specialist([turn(plan(("close_debit_card_4721", {"card_id": theirs, "reason": "lost"})))], shared)
    intent = closing(account_id="chk_lj82d4f1a9")  # names a record, so no question first
    assert run(spec, intent, ctx(shared, intent, user=ONE)).status is OutcomeStatus.FAILED


def test_kinds_that_do_not_need_a_specific_card_are_left_alone():
    shared = fresh()
    intent = Intent(id="x", kind=K.ORDER_REPLACEMENT_CARD, summary="new card", slots={})
    assert prepare(intent, MANY, shared.database) is intent


def test_the_plan_stays_bound_to_the_intent_as_the_orchestrator_holds_it_not_the_resolved_copy():
    from orchestrator.decision import fingerprint

    shared = fresh()
    last4 = shared.database.debit_cards.data["dbc_kj93a7b2e1_green"]["last_4_digits"]
    intent = closing(card_id=last4)
    spec = specialist([turn(plan(("close_debit_card_4721", {"card_id": "dbc_kj93a7b2e1_green", "reason": "lost"})))], shared)
    outcome = run(spec, intent, ctx(shared, intent, user=MANY))
    assert outcome.status is OutcomeStatus.NEEDS_CONFIRMATION
    assert outcome.proposal.intent_fp == fingerprint(intent), "a yes would otherwise look stale"


def test_the_dispatcher_also_understands_a_last_four_answer():
    shared = fresh()
    last4 = shared.database.debit_cards.data["dbc_kj93a7b2e1_green"]["last_4_digits"]
    d = SubAgentDispatcher({}, RecordKinds(shared.database))
    assert d.specialist_name(closing(card_id=last4), MANY) is Name.DEBIT


def test_the_debit_specialist_covers_exactly_the_kinds_the_dispatcher_sends_it():
    from agent.dispatcher import BY_RECORD, FIXED

    sent = {k for k, n in FIXED.items() if n is Name.DEBIT} | set(BY_RECORD)
    assert DEBIT_SPECIALIST.kinds <= sent | {K.DISPUTE_TRANSACTION}
    assert {K.CLOSE_CARD, K.ORDER_REPLACEMENT_CARD, K.ACTIVATE_CARD, K.CHANGE_OR_RESET_PIN, K.CARD_CONTROLS} <= DEBIT_SPECIALIST.kinds
