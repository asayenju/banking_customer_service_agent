import asyncio
import json

import pytest

from agent.dispatcher import (
    BY_RECORD,
    FIXED,
    Name,
    RecordKinds,
    SubAgentDispatcher,
    sub_agent_kinds,
    unmapped_kinds,
)
from agent.guard import allowed_writes
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.loop import Budget
from agent.specialist import Specialist, SpecialistConfig, ValidationContext, extract_field, system_prompt
from agent.toolbox import SecretVault
from banking_tools import BankingSession, Fidelity
from knowledge.index import Document, KnowledgeIndex
from knowledge.scopes import DEBIT
from orchestrator.auth import CustomerRecord, ScriptedDirectory
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.decomposer import Interpretation, ScriptedDecomposer
from orchestrator.executor import OutcomeStatus
from orchestrator.mocks import ScriptedRoute
from orchestrator.orchestrator import Orchestrator
from orchestrator.routes import ROUTES, Route
from orchestrator.scoring import IntentScore
from orchestrator.types import Decomposition, Intent, IntentKind as K

ME = "lj82d4f1a9"
MY_CARD, MY_ACCOUNT, MY_CREDIT = "dbc_lj82d4f1a9_bluest", "chk_lj82d4f1a9", "cc_lj82d4f1a9_silver"
OTHER_CARD, OTHER_ACCOUNT, OTHER_CREDIT = "dbc_kj93a7b2e1_blue", "chk_kj93a7b2e1_1", "cc_76ad9cc60e_gold"
INDEX = KnowledgeIndex(
    {
        "doc_bank_accounts_x_001": Document("doc_bank_accounts_x_001", "Internal: Closing Debit Cards", "To close a debit card use close_debit_card_4721 after confirming the reason."),
    }
)
CONFIG = SpecialistConfig(
    name="debit",
    kinds=frozenset({K.CLOSE_CARD, K.ORDER_REPLACEMENT_CARD, K.CARD_CONTROLS}),
    instructions="You handle debit card requests.",
    scope=DEBIT,
)


def call(name, **arguments):
    return ToolCall(id=f"c-{name}", name=name, arguments_json=json.dumps(arguments))


def turn(*calls):
    return AssistantTurn(tool_calls=tuple(calls), usage=Usage(input_tokens=50))


def plan_call(*steps):
    return call("propose_plan", steps=[{"tool": t, "arguments_json": json.dumps(a), "description": "x"} for t, a in steps])


def closing(**slots):
    return Intent(id="x", kind=K.CLOSE_CARD, summary="close my debit card", slots=slots or {"card_id": MY_CARD})


def make(script, config=CONFIG, vault=None):
    shared = BankingSession(fidelity=Fidelity.TAU2)
    llm = FakeToolLLM(script)
    return Specialist(config, llm=llm, index=INDEX, database=shared.database, vault=vault or SecretVault()), llm, shared


def context(shared, intent, *, mode="plan", approved=None, user=ME):
    session = scoped(shared)(SessionRequest(intent.id, mode=mode, approved=approved))
    return RouteContext(intent_id=intent.id, verified_user_id=user, session=session, mode=mode, approved=approved)


def run(specialist, intent, ctx):
    return asyncio.run(specialist.run(intent, ctx))


CLOSE_ARGS = {"card_id": MY_CARD, "reason": "lost"}


# ------------------------------------------------------------------ plan mode

def test_a_proposed_plan_is_rebuilt_by_code_from_the_catalog_not_taken_from_the_model():
    spec, _, shared = make([turn(plan_call(("close_debit_card_4721", CLOSE_ARGS)))])
    intent = closing()
    outcome = run(spec, intent, context(shared, intent))
    assert outcome.status is OutcomeStatus.NEEDS_CONFIRMATION
    (step,) = outcome.proposal.steps
    assert step.tool == "close_debit_card_4721" and step.reversible is False
    assert step.description != "x" and "debit card" in step.description.lower(), "the model's description was ignored"
    assert step.args["card_id"] == MY_CARD
    assert "Here is exactly what I will do" in outcome.proposal.rendered
    assert shared.audit_log == (), "planning changed nothing"


ORDER_ARGS = {"account_id": MY_ACCOUNT, "user_id": "someone_else", "delivery_option": "STANDARD", "delivery_fee": 0.0, "card_design": "CLASSIC", "design_fee": 0.0, "shipping_address": "1 Main St", "excess_replacement_fee": None}
REPLACE = [("close_debit_card_4721", CLOSE_ARGS), ("order_debit_card_5739", ORDER_ARGS)]


def replacing(**slots):
    return Intent(id="x", kind=K.ORDER_REPLACEMENT_CARD, summary="new card", slots=slots or {"account_id": MY_ACCOUNT})


def test_the_verified_customer_is_put_into_a_step_and_the_models_choice_of_customer_is_ignored():
    spec, _, shared = make([turn(plan_call(*REPLACE))])
    intent = replacing()
    outcome = run(spec, intent, context(shared, intent))
    assert outcome.status is OutcomeStatus.NEEDS_CONFIRMATION
    assert [s.args["user_id"] for s in outcome.proposal.steps if "user_id" in s.args] == [ME]


# ------------------------------------------------------------------ the dry run

def test_a_plan_is_judged_step_by_step_against_the_state_the_earlier_steps_leave():
    """Ordering a card is impossible while one is active, but fine after closing it."""
    spec, _, shared = make([turn(plan_call(*REPLACE)), turn(plan_call(REPLACE[1]))])
    intent = replacing()
    assert run(spec, intent, context(shared, intent)).status is OutcomeStatus.NEEDS_CONFIRMATION
    alone = run(spec, intent, context(shared, intent))
    assert alone.status is OutcomeStatus.FAILED and "step 1 would not go through" in alone.error
    assert "active debit card" in alone.error


def test_the_order_of_the_steps_matters_to_the_dry_run():
    spec, _, shared = make([turn(plan_call(REPLACE[1], REPLACE[0]))])
    intent = replacing()
    outcome = run(spec, intent, context(shared, intent))
    assert outcome.status is OutcomeStatus.FAILED and "step 1" in outcome.error


def test_a_dry_run_changes_nothing_real_not_the_database_the_audit_log_or_the_hash():
    spec, _, shared = make([turn(plan_call(*REPLACE))])
    start = shared.db_hash()
    intent = replacing()
    assert run(spec, intent, context(shared, intent)).status is OutcomeStatus.NEEDS_CONFIRMATION
    assert shared.db_hash() == start and shared.audit_log == ()
    assert shared.database.debit_cards.data[MY_CARD]["status"] == "ACTIVE"


def test_a_pin_in_a_step_works_in_the_dry_run_and_never_appears_in_the_error_text():
    vault = SecretVault()
    vault.store("pin", "7351")
    cfg = SpecialistConfig(name="debit", kinds=frozenset({K.CHANGE_OR_RESET_PIN}), instructions="pins", scope=DEBIT)
    bad = {"card_id": MY_CARD, "last_4_digits": "0000", "new_pin": "{{secret:pin}}"}
    spec, _, shared = make([turn(plan_call(("reset_debit_card_pin_6284", bad)))], cfg, vault)
    intent = Intent(id="x", kind=K.CHANGE_OR_RESET_PIN, summary="reset my pin", slots={"card_id": MY_CARD})
    outcome = run(spec, intent, context(shared, intent))
    assert outcome.status is OutcomeStatus.FAILED and "7351" not in outcome.error


@pytest.mark.parametrize(
    "steps, part",
    [
        ([("freeze_debit_card_3892", {"card_id": MY_CARD})], "may not use"),
        ([("transfer_funds_between_bank_accounts_7291", {})], "may not use"),
        ([("no_such_tool_0000", {})], "may not use"),
        ([("close_debit_card_4721", {"card_id": MY_CARD})], "did not fit"),
        ([("close_debit_card_4721", {"card_id": OTHER_CARD, "reason": "lost"})], "refused"),
        ([("close_debit_card_4721", {"card_id": "dbc_made_up", "reason": "lost"})], "refused"),
    ],
)
def test_an_unusable_plan_is_refused_before_the_customer_ever_sees_it(steps, part):
    spec, _, shared = make([turn(plan_call(*steps))])
    intent = closing()
    outcome = run(spec, intent, context(shared, intent))
    assert outcome.status is OutcomeStatus.FAILED and part in outcome.error
    assert OTHER_CARD not in outcome.error


def test_malformed_plan_arguments_are_refused():
    for bad in ([], "not a list", [{"tool": "close_debit_card_4721"}], [{"tool": "close_debit_card_4721", "arguments_json": "{nope", "description": "x"}]):
        spec, _, shared = make([turn(call("propose_plan", steps=bad))])
        intent = closing()
        assert run(spec, intent, context(shared, intent)).status is OutcomeStatus.FAILED


def test_a_raw_pin_in_a_plan_is_refused_and_a_reference_is_kept_private():
    vault = SecretVault()
    vault.store("pin", "7351")
    pin_args = {"card_id": MY_CARD, "last_4_digits": "4821", "new_pin": "7351"}
    cfg = SpecialistConfig(name="debit", kinds=frozenset({K.CHANGE_OR_RESET_PIN}), instructions="pins", scope=DEBIT)
    intent = Intent(id="x", kind=K.CHANGE_OR_RESET_PIN, summary="reset my pin", slots={"card_id": MY_CARD})
    spec, _, shared = make([turn(plan_call(("reset_debit_card_pin_6284", pin_args)))], cfg, vault)
    assert run(spec, intent, context(shared, intent)).status is OutcomeStatus.FAILED
    ok_args = {**pin_args, "new_pin": "{{secret:pin}}"}
    spec, _, shared = make([turn(plan_call(("reset_debit_card_pin_6284", ok_args)))], cfg, vault)
    outcome = run(spec, intent, context(shared, intent))
    assert outcome.status is OutcomeStatus.NEEDS_CONFIRMATION
    assert "7351" not in outcome.proposal.rendered and "(kept private)" in outcome.proposal.rendered


def test_a_validator_can_refuse_a_step_with_a_reason_written_in_code():
    def not_before_noon(v: ValidationContext, args):
        return "the card has a pending dispute" if args["card_id"] == MY_CARD else None

    cfg = SpecialistConfig(name="debit", kinds=CONFIG.kinds, instructions="x", scope=DEBIT, validators={"close_debit_card_4721": not_before_noon})
    spec, _, shared = make([turn(plan_call(("close_debit_card_4721", CLOSE_ARGS)))], cfg)
    intent = closing()
    outcome = run(spec, intent, context(shared, intent))
    assert outcome.status is OutcomeStatus.FAILED and "pending dispute" in outcome.error


def test_asking_the_customer_returns_only_the_names_of_the_details():
    spec, _, shared = make([turn(call("ask_customer", details=["card_id", "reason", "Not Valid!", "4821"]))])
    intent = closing(card_id="")
    outcome = run(spec, intent, context(shared, intent))
    assert outcome.status is OutcomeStatus.NEEDS_INFO
    assert [q.slot for q in outcome.questions] == ["card_id", "reason"]


def test_a_question_with_nothing_usable_in_it_fails_rather_than_asking_nothing():
    spec, _, shared = make([turn(call("ask_customer", details=["4821", ""]))])
    intent = closing()
    assert run(spec, intent, context(shared, intent)).status is OutcomeStatus.FAILED


def test_finishing_without_a_plan_is_a_done_answer_with_secrets_scrubbed():
    vault = SecretVault()
    vault.store("pin", "7351")
    spec, _, shared = make([turn(call("finish", message="Policy says no. Your code 7351 is fine."))], vault=vault)
    intent = closing()
    outcome = run(spec, intent, context(shared, intent))
    assert outcome.status is OutcomeStatus.DONE and "7351" not in outcome.text and "Policy says no" in outcome.text


def test_escalating_is_a_failure_that_offers_a_human_with_a_fixed_reason():
    spec, _, shared = make([turn(call("escalate", reason="customer card 4111111111111111 lost"))])
    intent = closing()
    outcome = run(spec, intent, context(shared, intent))
    assert outcome.status is OutcomeStatus.FAILED and "4111" not in outcome.error


def test_running_out_of_budget_is_a_failure_with_a_code_reason():
    cfg = SpecialistConfig(name="debit", kinds=CONFIG.kinds, instructions="x", scope=DEBIT, budget=Budget(max_steps=2))
    spec, _, shared = make([turn(call("get_current_time"))] * 5, cfg)
    intent = closing()
    outcome = run(spec, intent, context(shared, intent))
    assert outcome.status is OutcomeStatus.FAILED and "steps_exhausted" in outcome.error


def test_the_model_can_read_search_and_is_offered_no_write_tools_in_plan_mode():
    spec, llm, shared = make([turn(call("search_policy", query="close debit card")), turn(call("finish", message="ok"))])
    intent = closing()
    run(spec, intent, context(shared, intent))
    offered = set(llm.calls[0]["tools"])
    assert "search_policy" in offered and {"finish", "ask_customer", "escalate", "propose_plan"} <= offered
    assert not (offered & allowed_writes(K.CLOSE_CARD)), "writes are described, not offered"
    searched = llm.calls[1]["messages"][-1]["content"]
    assert "<notes>" in searched and "close_debit_card_4721" not in searched, "the note's tool name is hidden"


def test_the_prompt_lists_the_tools_a_plan_may_use_with_their_arguments_and_whether_they_can_be_undone():
    prompt = system_prompt(CONFIG, K.CLOSE_CARD)
    assert "close_debit_card_4721" in prompt and "cannot be undone" in prompt
    assert "freeze_debit_card_3892" not in prompt or "can be undone" in prompt
    assert "never changes anything" in system_prompt(CONFIG, K.GENERAL_QUESTION)


def test_the_customers_words_reach_the_model_with_secrets_already_swapped_for_references():
    vault = SecretVault()
    intent = Intent(id="x", kind=K.CLOSE_CARD, summary="my pin is 7351 close the card", slots={"card_id": MY_CARD})
    spec, llm, shared = make([turn(call("finish", message="ok"))], vault=vault)
    run(spec, intent, context(shared, intent))
    sent = json.dumps(llm.calls)
    assert "7351" not in sent and "{{secret:pin}}" in sent


def test_without_a_session_a_verified_customer_or_the_right_kind_it_fails_closed():
    spec, _, shared = make([])
    intent = closing()
    assert run(spec, intent, RouteContext(intent_id="x", verified_user_id=ME)).status is OutcomeStatus.FAILED
    assert run(spec, intent, context(shared, intent, user=None)).status is OutcomeStatus.FAILED
    wrong = Intent(id="x", kind=K.TRANSFER_FUNDS, summary="s")
    assert run(spec, wrong, context(shared, wrong)).status is OutcomeStatus.FAILED


# ------------------------------------------------------------------ execute mode

def approved_plan(spec, shared, *steps):
    intent = closing()
    s2, _, _ = make([turn(plan_call(*steps))])
    s2._db = shared.database
    return run(s2, intent, context(shared, intent)).proposal


def test_execute_mode_runs_the_approved_steps_in_code_without_calling_a_model():
    spec, llm, shared = make([])
    plan = approved_plan(spec, shared, ("close_debit_card_4721", CLOSE_ARGS))
    intent = closing()
    outcome = run(spec, intent, context(shared, intent, mode="execute", approved=plan))
    assert outcome.status is OutcomeStatus.DONE and outcome.text.startswith("Done:")
    assert llm.calls == [], "no model was called"
    assert [e.tool for e in shared.audit_log] == ["close_debit_card_4721"]


def test_execute_mode_without_an_approved_plan_does_nothing():
    spec, _, shared = make([])
    intent = closing()
    assert run(spec, intent, context(shared, intent, mode="execute")).status is OutcomeStatus.FAILED
    assert shared.audit_log == ()


def test_a_step_that_no_longer_passes_the_validator_stops_the_run_before_any_change():
    state = {"strict": False}

    def validator(v, args):
        return "the card now has a pending dispute" if state["strict"] else None

    cfg = SpecialistConfig(name="debit", kinds=CONFIG.kinds, instructions="x", scope=DEBIT, validators={"close_debit_card_4721": validator})
    spec, _, shared = make([turn(plan_call(("close_debit_card_4721", CLOSE_ARGS)))], cfg)
    intent = closing()
    plan = run(spec, intent, context(shared, intent)).proposal
    state["strict"] = True  # things changed between the preview and the yes
    outcome = run(spec, intent, context(shared, intent, mode="execute", approved=plan))
    assert outcome.status is OutcomeStatus.FAILED and "pending dispute" in outcome.error
    assert shared.audit_log == ()


def test_if_the_world_changed_after_the_preview_the_run_stops_at_that_step_and_goes_no_further():
    spec, _, shared = make([turn(plan_call(*REPLACE))])
    intent = replacing()
    plan = run(spec, intent, context(shared, intent)).proposal
    # between the preview and the yes the card was closed some other way
    shared.database.debit_cards.data[MY_CARD]["status"] = "CLOSED"
    outcome = run(spec, intent, context(shared, intent, mode="execute", approved=plan))
    assert outcome.status is OutcomeStatus.FAILED and "step 1 did not go through" in outcome.error
    assert [e.tool for e in shared.audit_log] == ["close_debit_card_4721"] and shared.audit_log[0].ok is False


def test_a_multi_step_plan_runs_every_step_in_order_when_nothing_changed():
    spec, _, shared = make([turn(plan_call(*REPLACE))])
    intent = replacing()
    plan = run(spec, intent, context(shared, intent)).proposal
    outcome = run(spec, intent, context(shared, intent, mode="execute", approved=plan))
    assert outcome.status is OutcomeStatus.DONE
    assert [e.tool for e in shared.audit_log] == ["close_debit_card_4721", "order_debit_card_5739"]
    assert all(e.ok for e in shared.audit_log)


def test_a_reference_to_an_earlier_results_field_is_filled_in_from_its_output():
    assert extract_field("Card ID: dbc_9\nStatus: CLOSED", "card_id") == "dbc_9"
    assert extract_field("New Account Id : abc123", "new_account_id") == "abc123"
    assert extract_field("nothing here", "card_id") is None
    spec, _, shared = make([])
    assert spec._resolve({"a": "{{step_1.card_id}}", "b": 5}, {1: "Card ID: dbc_9"}) == {"a": "dbc_9", "b": 5}
    assert spec._resolve({"a": "{{step_2.card_id}}"}, {1: "Card ID: dbc_9"}) is None


# ------------------------------------------------------------------ the dispatcher

def test_every_sub_agent_kind_has_an_owner_and_a_new_kind_fails_this_test():
    assert sub_agent_kinds() == {k for k, r in ROUTES.items() if r is Route.SUB_AGENT}
    assert unmapped_kinds() == frozenset(), f"decide who handles: {sorted(unmapped_kinds())}"
    assert set(FIXED) & BY_RECORD == set()


def test_the_dispatcher_refuses_a_kind_nobody_owns(monkeypatch):
    from agent import dispatcher

    monkeypatch.setitem(dispatcher.ROUTES, K.OTHER, Route.SUB_AGENT)
    assert K.OTHER in dispatcher.unmapped_kinds()


def dispatcher_for(shared, **specialists):
    return SubAgentDispatcher(specialists, RecordKinds(shared.database))


def ctx_for(shared, intent, user=ME):
    return context(shared, intent, user=user)


@pytest.mark.parametrize(
    "kind, slots, expected",
    [
        (K.DISPUTE_TRANSACTION, {"card_id": MY_CARD}, Name.DEBIT),
        (K.DISPUTE_TRANSACTION, {"account_id": MY_ACCOUNT}, Name.DEBIT),
        (K.DISPUTE_TRANSACTION, {"credit_card_account_id": MY_CREDIT}, Name.CREDIT),
        (K.DISPUTE_TRANSACTION, {"account_id": MY_CREDIT}, Name.CREDIT),
        (K.CLOSE_CARD, {"card_id": MY_CARD}, Name.DEBIT),
        (K.CLOSE_CARD, {"credit_card_account_id": MY_CREDIT}, Name.CREDIT),
        (K.ORDER_REPLACEMENT_CARD, {"card_id": MY_CARD}, Name.DEBIT),
        (K.ORDER_REPLACEMENT_CARD, {"credit_card_account_id": MY_CREDIT}, Name.CREDIT),
    ],
)
def test_debit_or_credit_is_decided_from_the_record_the_customer_named(kind, slots, expected):
    shared = BankingSession()
    d = dispatcher_for(shared)
    assert d.specialist_name(Intent(id="x", kind=kind, summary="s", slots=slots), ME) is expected


def test_a_transaction_is_traced_to_its_product():
    shared = BankingSession()
    db = shared.database
    bank_txn = next(k for k, v in db.bank_account_transaction_history.data.items() if v["account_id"] == MY_ACCOUNT)
    d = dispatcher_for(shared)
    assert d.specialist_name(Intent(id="x", kind=K.DISPUTE_TRANSACTION, summary="s", slots={"transaction_id": bank_txn}), ME) is Name.DEBIT
    credit_txn = next(k for k, v in db.credit_card_transaction_history.data.items() if v["user_id"] == "01f21c9970")
    assert d.specialist_name(Intent(id="x", kind=K.DISPUTE_TRANSACTION, summary="s", slots={"transaction_id": credit_txn}), "01f21c9970") is Name.CREDIT


@pytest.mark.parametrize(
    "slots",
    [
        {},
        {"card_id": ""},
        {"card_id": "dbc_made_up"},
        {"card_id": OTHER_CARD},
        {"credit_card_account_id": OTHER_CREDIT},
        {"account_id": OTHER_ACCOUNT},
        {"card_id": MY_CARD, "credit_card_account_id": MY_CREDIT},
    ],
)
def test_nothing_named_unknown_someone_elses_or_both_products_means_asking_not_guessing(slots):
    shared = BankingSession(fidelity=Fidelity.TAU2)
    d = dispatcher_for(shared)
    intent = Intent(id="x", kind=K.CLOSE_CARD, summary="close it", slots=slots)
    assert d.specialist_name(intent, ME) == ["card_or_account_id"]
    outcome = asyncio.run(d.run(intent, ctx_for(shared, intent)))
    assert outcome.status is OutcomeStatus.NEEDS_INFO and [q.slot for q in outcome.questions] == ["card_or_account_id"]


def test_another_customers_id_is_indistinguishable_from_one_that_does_not_exist():
    shared = BankingSession()
    d = dispatcher_for(shared)
    mine = lambda c: d.specialist_name(Intent(id="x", kind=K.CLOSE_CARD, summary="s", slots={"card_id": c}), ME)
    assert mine(OTHER_CARD) == mine("dbc_made_up") == ["card_or_account_id"]


def test_a_credit_limit_request_with_no_record_named_is_a_credit_one_but_a_named_debit_card_is_not():
    shared = BankingSession()
    d = dispatcher_for(shared)
    assert d.specialist_name(Intent(id="x", kind=K.CREDIT_LIMIT_REQUEST, summary="s"), ME) is Name.CREDIT
    assert d.specialist_name(Intent(id="x", kind=K.CREDIT_LIMIT_REQUEST, summary="s", slots={"card_id": MY_CARD}), ME) is Name.DEBIT


def test_fixed_kinds_go_to_their_owner_without_looking_at_records():
    shared = BankingSession()
    d = dispatcher_for(shared)
    for kind, owner in FIXED.items():
        assert d.specialist_name(Intent(id="x", kind=kind, summary="s"), None) is owner


def test_a_kind_whose_specialist_is_not_built_yet_is_a_plain_failure_not_a_silent_gap():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    d = dispatcher_for(shared)  # no specialists at all
    intent = Intent(id="x", kind=K.TRANSFER_FUNDS, summary="move money")
    outcome = asyncio.run(d.run(intent, ctx_for(shared, intent)))
    assert outcome.status is OutcomeStatus.FAILED and "not implemented" in outcome.error


def test_the_dispatcher_hands_a_resolved_intent_to_the_matching_specialist():
    spec, llm, shared = make([turn(call("finish", message="routed"))])
    d = dispatcher_for(shared, **{Name.DEBIT.value: spec})
    d = SubAgentDispatcher({Name.DEBIT: spec}, RecordKinds(shared.database))
    intent = closing()
    outcome = asyncio.run(d.run(intent, ctx_for(shared, intent)))
    assert outcome.status is OutcomeStatus.DONE and outcome.text == "routed" and len(llm.calls) == 1
    assert d.proposes_plans is True


# ------------------------------------------------------------------ through the real orchestrator

def real_customer():
    user = BankingSession().database.users.data[ME]
    return CustomerRecord(user_id=ME, date_of_birth=user["date_of_birth"], email=user["email"], phone_number=user["phone_number"], address=user["address"])


class Scores:
    async def score(self, message, decomposition):
        return [IntentScore.flat(i.id, 0.99) for i in decomposition.intents]


def test_preview_then_yes_then_the_change_through_the_real_orchestrator():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    spec, llm, _ = make([turn(plan_call(("close_debit_card_4721", CLOSE_ARGS)))])
    spec._db = shared.database
    dispatcher = SubAgentDispatcher({Name.DEBIT: spec}, RecordKinds(shared.database))
    record = real_customer()
    claims = {"email": record.email, "phone_number": record.phone_number}
    intent = closing()
    orch = Orchestrator(
        decomposer=ScriptedDecomposer(
            [
                Interpretation(decomposition=Decomposition(intents=(intent,)), claims=claims),
                Interpretation(decomposition=Decomposition(intents=()), confirmed=("x",)),
            ]
        ),
        scorer=Scores(),
        directory=ScriptedDirectory(record),
        runners={Route.DIRECT: ScriptedRoute(), Route.SUB_AGENT: dispatcher},
        session_for=scoped(shared),
    )
    first = asyncio.run(orch.handle_turn("m1", "close my debit card, it was lost"))
    assert "Here is exactly what I will do" in first.reply.text and "cannot be undone" in first.reply.text
    assert shared.audit_log == () and len(llm.calls) == 1

    second = asyncio.run(orch.handle_turn("m2", "yes"))
    assert [o.status for o in second.outcomes] == [OutcomeStatus.DONE]
    assert [e.tool for e in shared.audit_log] == ["close_debit_card_4721"]
    assert shared.database.debit_cards.data[MY_CARD]["status"] == "CLOSED"
    assert len(llm.calls) == 1, "the yes ran the plan in code; the model was not called again"
