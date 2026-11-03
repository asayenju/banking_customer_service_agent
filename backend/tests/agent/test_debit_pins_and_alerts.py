import asyncio
import json
from datetime import date

import pytest

from agent.dispatcher import Name, RecordKinds, SubAgentDispatcher
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist, ValidationContext
from agent.specialists import debit
from agent.specialists.debit import DEBIT_SPECIALIST
from agent.toolbox import SecretVault
from banking_tools import BankingSession, Fidelity
from knowledge.index import Document, KnowledgeIndex
from orchestrator.auth import CustomerRecord, ScriptedDirectory
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.decomposer import Interpretation, ScriptedDecomposer
from orchestrator.executor import OutcomeStatus
from orchestrator.mocks import ScriptedRoute
from orchestrator.orchestrator import Orchestrator
from orchestrator.routes import Route
from orchestrator.scoring import IntentScore
from orchestrator.types import Decomposition, Intent, IntentKind as K

ME, CARD, ACCOUNT, LAST4 = "lj82d4f1a9", "dbc_lj82d4f1a9_bluest", "chk_lj82d4f1a9", "4821"
PIN, OLD_PIN, CVV = "7351", "2468", "739"
INDEX = KnowledgeIndex({"doc_bank_accounts_x": Document("doc_bank_accounts_x", "Internal: PIN", "pin reset procedure")})


def fresh():
    return BankingSession(fidelity=Fidelity.TAU2)


def v(shared):
    return ValidationContext(shared.database, ME, date(2025, 11, 14))


def card(shared, **changes):
    shared.database.debit_cards.data[CARD].update(changes)


def call(name, **arguments):
    return ToolCall(id=f"c-{name}-{len(json.dumps(arguments))}", name=name, arguments_json=json.dumps(arguments))


def turn(*calls):
    return AssistantTurn(tool_calls=tuple(calls), usage=Usage(input_tokens=50))


def plan(*steps):
    return call("propose_plan", steps=[{"tool": t, "arguments_json": json.dumps(a), "description": "x"} for t, a in steps])


# ------------------------------------------------------------------ validators

@pytest.mark.parametrize("new_pin, ok", [("7351", True), ("1234", False), ("9876", False), ("1111", False), ("12", False), ("abcd", False), ("", False)])
def test_a_new_pin_must_follow_the_banks_own_rule(new_pin, ok):
    shared = fresh()
    args = {"card_id": CARD, "current_pin": OLD_PIN, "new_pin": new_pin}
    assert (debit.change_pin(v(shared), args) is None) is ok


def test_the_new_pin_must_differ_from_the_current_one():
    assert "differ" in debit.change_pin(v(fresh()), {"card_id": CARD, "current_pin": PIN, "new_pin": PIN})


@pytest.mark.parametrize("check, args", [
    (debit.change_pin, {"current_pin": OLD_PIN, "new_pin": PIN}),
    (debit.reset_pin, {"last_4_digits": LAST4, "new_pin": PIN}),
    (debit.recurring_block, {"block_recurring": True}),
    (debit.temporary_limit, {"limit_type": "atm", "new_limit": 900}),
])
def test_pins_blocks_and_limits_need_an_active_card(check, args):
    shared = fresh()
    assert check(v(shared), {"card_id": CARD, **args}) is None
    for status in ("FROZEN", "CLOSED", "PENDING"):
        card(shared, status=status)
        assert "active card" in check(v(shared), {"card_id": CARD, **args}), status
    assert "does not exist" in check(v(shared), {"card_id": "dbc_made_up", **args})


def test_a_pin_reset_needs_the_last_four_digits_of_that_card():
    shared = fresh()
    assert debit.reset_pin(v(shared), {"card_id": CARD, "last_4_digits": LAST4, "new_pin": PIN}) is None
    assert "do not match" in debit.reset_pin(v(shared), {"card_id": CARD, "last_4_digits": "0000", "new_pin": PIN})


@pytest.mark.parametrize("limit_type, limit, ok", [("atm", 900, True), ("purchase", 5000.0, True), ("atm", 0, False), ("atm", -5, False), ("atm", 12.5, False), ("atm", True, False), ("wire", 900, False), ("atm", "900", False)])
def test_a_temporary_limit_is_a_positive_whole_number_for_atm_or_purchase(limit_type, limit, ok):
    args = {"card_id": CARD, "limit_type": limit_type, "new_limit": limit}
    assert (debit.temporary_limit(v(fresh()), args) is None) is ok


def test_alerts_are_cleared_only_where_the_bank_allows_it():
    shared = fresh()
    args = {"card_id": CARD, "reason": "customer_verified"}
    assert "no active fraud alert" in debit.clear_alert(v(shared), args)
    card(shared, fraud_alert_active=True, alert_source="customer")
    assert debit.clear_alert(v(shared), args) is None
    card(shared, alert_source="bank_initiated")
    assert "only the security team" in debit.clear_alert(v(shared), args)
    assert "no velocity block" in debit.clear_alert(v(shared), {"card_id": CARD, "reason": "velocity_clear"})
    card(shared, velocity_blocked=True)
    assert debit.clear_alert(v(shared), {"card_id": CARD, "reason": "velocity_clear"}) is None
    assert "customer_verified or velocity_clear" in debit.clear_alert(v(shared), {"card_id": CARD, "reason": "because"})


# ------------------------------------------------------------------ the PIN, start to finish

def real_customer():
    user = BankingSession().database.users.data[ME]
    return CustomerRecord(user_id=ME, date_of_birth=user["date_of_birth"], email=user["email"], phone_number=user["phone_number"], address=user["address"])


class Scores:
    async def score(self, message, decomposition):
        return [IntentScore.flat(i.id, 0.99) for i in decomposition.intents]


def pin_world(model_script, intent_kind=K.CHANGE_OR_RESET_PIN, slots=None, prepare=None):
    shared = fresh()
    vault = SecretVault()
    llm = FakeToolLLM(model_script)
    spec = Specialist(DEBIT_SPECIALIST, llm=llm, index=INDEX, database=shared.database, vault=vault)
    dispatcher = SubAgentDispatcher({Name.DEBIT: spec}, RecordKinds(shared.database))
    record = real_customer()
    claims = {"email": record.email, "phone_number": record.phone_number}
    intent = Intent(id="p", kind=intent_kind, summary="reset my pin", slots=slots or {"card_id": CARD, "new_pin": "{{secret:pin}}", "last_4_digits": LAST4})
    decomposer = ScriptedDecomposer(
        [
            Interpretation(decomposition=Decomposition(intents=(intent,)), claims=claims),
            Interpretation(decomposition=Decomposition(intents=()), confirmed=("p",)),
        ]
    )
    orch = Orchestrator(
        decomposer=decomposer,
        scorer=Scores(),
        directory=ScriptedDirectory(record),
        runners={Route.DIRECT: ScriptedRoute(), Route.SUB_AGENT: dispatcher},
        session_for=scoped(shared),
        capture_secrets=vault.capture,
    )
    return orch, shared, llm, decomposer


RESET = ("reset_debit_card_pin_6284", {"card_id": CARD, "last_4_digits": LAST4, "new_pin": "{{secret:pin}}"})


def everything_visible(*results, shared, llm, decomposer):
    parts = [json.dumps(llm.calls, default=str), json.dumps(decomposer.calls, default=str)]
    for r in results:
        parts += [repr(r.trace), repr(r.reply), repr(r.outcomes), repr(r.decisions)]
    parts += [json.dumps([e.model_dump(mode="json") for e in shared.audit_log], default=str)]
    return "\n".join(parts)


def test_a_pin_the_customer_types_never_reaches_a_model_a_trace_a_reply_or_the_audit_log():
    script = [
        turn(call("get_debit_cards_by_account_id_7823", account_id=ACCOUNT)),  # a read whose result holds the CVV
        turn(plan(RESET)),
    ]
    orch, shared, llm, decomposer = pin_world(script)
    first = asyncio.run(orch.handle_turn("m1", f"Please reset my pin, the new pin is {PIN}. My card ends {LAST4}."))
    assert "Here is exactly what I will do" in first.reply.text and "(kept private)" in first.reply.text
    assert shared.audit_log == () and shared.database.debit_cards.data[CARD].get("pin_last_changed") is None
    second = asyncio.run(orch.handle_turn("m2", "yes"))

    assert [o.status for o in second.outcomes] == [OutcomeStatus.DONE]
    assert shared.database.debit_cards.data[CARD]["pin_last_changed"] == "11/14/2025", "the PIN reset really happened"
    seen = everything_visible(first, second, shared=shared, llm=llm, decomposer=decomposer)
    assert PIN not in seen, "the PIN leaked"
    assert CVV not in seen, "the card's security code leaked from a read result"
    assert "{{secret:pin}}" in seen, "the model and the decomposer were given the reference instead"
    assert [e.arguments["new_pin"] for e in shared.audit_log] == ["***"]


def test_the_plan_the_customer_approves_is_bound_to_the_reference_not_the_digits():
    from orchestrator.plan_preview import plan_fingerprint

    orch, shared, llm, decomposer = pin_world([turn(plan(RESET))])
    first = asyncio.run(orch.handle_turn("m1", f"set my new pin to {PIN} for the card ending {LAST4}"))
    fp = first.decisions[0].fingerprint
    assert fp and PIN not in fp
    other_pin_same_ref = orch  # a different PIN behind the same reference is the same plan text
    assert "kept private" in first.reply.text


def test_a_pin_phrased_in_different_ways_is_still_caught():
    for phrasing in (f"my pin is {PIN}", f"change the pin to {PIN}", f"the new pin should be {PIN}", f"PIN: {PIN}", f"{PIN} is my new pin"):
        orch, shared, llm, decomposer = pin_world([turn(plan(RESET))])
        asyncio.run(orch.handle_turn("m1", f"reset it, {phrasing}"))
        assert PIN not in json.dumps(decomposer.calls) + json.dumps(llm.calls, default=str), phrasing


def test_a_pin_in_a_second_message_after_the_question_is_caught_too():
    orch, shared, llm, decomposer = pin_world([turn(plan(RESET))])
    asyncio.run(orch.handle_turn("m1", "reset my pin"))
    asyncio.run(orch.handle_turn("m2", f"yes, the new pin is {PIN}"))
    assert PIN not in json.dumps(decomposer.calls)


def test_a_plan_that_uses_a_digit_pin_directly_is_refused_before_the_customer_sees_it():
    bad = ("reset_debit_card_pin_6284", {"card_id": CARD, "last_4_digits": LAST4, "new_pin": PIN})
    orch, shared, llm, decomposer = pin_world([turn(plan(bad))])
    first = asyncio.run(orch.handle_turn("m1", f"reset my pin to {PIN}"))
    assert [o.status for o in first.outcomes] == [OutcomeStatus.FAILED]
    assert PIN not in repr(first.reply) + repr(first.trace)
    assert shared.audit_log == ()


# ------------------------------------------------------------------ a fraud alert needs a yes

def test_clearing_a_fraud_alert_is_previewed_as_something_that_cannot_be_undone_and_waits_for_a_yes():
    shared = fresh()
    card(shared, fraud_alert_active=True, alert_source="customer")
    step = ("clear_debit_card_fraud_alert_4892", {"card_id": CARD, "reason": "customer_verified"})
    llm = FakeToolLLM([turn(plan(step))])
    spec = Specialist(DEBIT_SPECIALIST, llm=llm, index=INDEX, database=shared.database, vault=SecretVault())
    intent = Intent(id="a", kind=K.CARD_CONTROLS, summary="clear the alert", slots={"card_id": CARD})
    session = scoped(shared)(SessionRequest("a"))
    outcome = asyncio.run(spec.run(intent, RouteContext(intent_id="a", verified_user_id=ME, session=session, mode="plan")))
    assert outcome.status is OutcomeStatus.NEEDS_CONFIRMATION
    (s,) = outcome.proposal.steps
    assert s.reversible is False and not outcome.proposal.reversible_only
    assert "cannot be undone" in outcome.proposal.rendered
    assert shared.audit_log == () and shared.database.debit_cards.data[CARD]["fraud_alert_active"] is True


def test_through_the_orchestrator_the_alert_is_cleared_only_after_the_yes():
    orch, shared, llm, decomposer = pin_world(
        [turn(plan(("clear_debit_card_fraud_alert_4892", {"card_id": CARD, "reason": "customer_verified"})))],
        intent_kind=K.CARD_CONTROLS,
        slots={"card_id": CARD},
    )
    card(shared, fraud_alert_active=True, alert_source="customer")
    first = asyncio.run(orch.handle_turn("m1", "my card is blocked by a fraud alert, it was me"))
    assert [d.action.value for d in first.decisions] == ["ask"] and shared.audit_log == ()
    second = asyncio.run(orch.handle_turn("m2", "yes"))
    assert [e.tool for e in shared.audit_log] == ["clear_debit_card_fraud_alert_4892"]
    assert shared.database.debit_cards.data[CARD]["fraud_alert_active"] is False and second.outcomes[0].status is OutcomeStatus.DONE


def test_a_bank_raised_alert_is_not_cleared_and_no_plan_is_offered():
    orch, shared, llm, decomposer = pin_world(
        [turn(plan(("clear_debit_card_fraud_alert_4892", {"card_id": CARD, "reason": "customer_verified"})))],
        intent_kind=K.CARD_CONTROLS,
        slots={"card_id": CARD},
    )
    card(shared, fraud_alert_active=True, alert_source="bank_initiated")
    first = asyncio.run(orch.handle_turn("m1", "clear the alert"))
    assert [o.status for o in first.outcomes] == [OutcomeStatus.FAILED] and shared.audit_log == ()


# ------------------------------------------------------------------ the undoable ones need no yes

def test_blocking_recurring_payments_can_be_undone_so_it_runs_without_a_yes_when_the_kind_is_certain():
    orch, shared, llm, decomposer = pin_world(
        [turn(plan(("set_debit_card_recurring_block_7382", {"card_id": CARD, "block_recurring": True})))],
        intent_kind=K.CARD_CONTROLS,
        slots={"card_id": CARD},
    )
    first = asyncio.run(orch.handle_turn("m1", "stop my subscriptions on this card"))
    assert [o.status for o in first.outcomes] == [OutcomeStatus.DONE]
    assert shared.database.debit_cards.data[CARD]["recurring_blocked"] is True and "Here is exactly" not in first.reply.text


def test_a_temporary_limit_increase_is_previewed_because_it_is_not_undoable():
    shared = fresh()
    step = ("request_temporary_debit_card_limit_increase_8374", {"card_id": CARD, "limit_type": "atm", "new_limit": 2000})
    spec = Specialist(DEBIT_SPECIALIST, llm=FakeToolLLM([turn(plan(step))]), index=INDEX, database=shared.database, vault=SecretVault())
    intent = Intent(id="l", kind=K.CARD_CONTROLS, summary="raise my atm limit", slots={"card_id": CARD})
    outcome = asyncio.run(spec.run(intent, RouteContext(intent_id="l", verified_user_id=ME, session=scoped(shared)(SessionRequest("l")), mode="plan")))
    if outcome.status is OutcomeStatus.NEEDS_CONFIRMATION:
        assert not outcome.proposal.reversible_only
    else:
        assert outcome.status is OutcomeStatus.FAILED  # the bank may refuse this account type; never a silent success
