import asyncio
import json

import pytest

from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.loop import Budget, StopReason, run_agent
from agent.specialist import Specialist, SpecialistConfig, _clean_details
from agent.specialists.debit import DEBIT_SPECIALIST
from agent.specialists.general import GENERAL_ANSWERER
from agent.toolbox import ScopedToolBox, SecretVault
from banking_tools import BankingSession, Fidelity
from knowledge.index import Document, KnowledgeIndex
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.executor import OutcomeStatus
from orchestrator.types import Intent, IntentKind as K

ME, CARD, ACCOUNT = "lj82d4f1a9", "dbc_lj82d4f1a9_bluest", "chk_lj82d4f1a9"
BLUE = "doc_checking_accounts_blue_account_001"
INDEX = KnowledgeIndex({BLUE: Document(BLUE, "Blue Account", "The Blue Account has a $10 monthly fee.")})


def call(name, **arguments):
    return ToolCall(id=f"c-{name}-{len(json.dumps(arguments))}-{id(arguments) % 997}", name=name, arguments_json=json.dumps(arguments))


def turn(*calls):
    return AssistantTurn(tool_calls=tuple(calls), usage=Usage(input_tokens=40))


def plan(*steps):
    return call("propose_plan", steps=[{"tool": t, "arguments_json": json.dumps(a), "description": "x"} for t, a in steps])


# ------------------------------------------------------------------ the loop

def run(script, review, **kw):
    shared = BankingSession(fidelity=Fidelity.TAU2)
    llm = FakeToolLLM(script)
    box = ScopedToolBox(scoped(shared)(SessionRequest("i")), user_id=ME, allowed_tools=["get_current_time"])
    result = asyncio.run(run_agent(llm, box, system="s", request="r", review=review, **kw))
    return result, llm


def test_a_refused_ending_goes_back_to_the_model_with_the_reason_and_a_corrected_one_is_accepted():
    seen = []

    def review(name, arguments):
        seen.append(arguments["message"])
        return None if arguments["message"] == "good" else "say it as good"

    result, llm = run([turn(call("finish", message="bad")), turn(call("finish", message="good"))], review)
    assert result.finished and result.arguments == {"message": "good"} and seen == ["bad", "good"]
    assert result.errors == 1 and result.steps == 2
    told = llm.calls[1]["messages"][-1]
    assert told["role"] == "tool" and told["content"] == "Not accepted: say it as good. Correct it and call finish again."


def test_a_model_that_keeps_getting_it_wrong_is_still_stopped_by_the_error_budget():
    result, llm = run([turn(call("finish", message="bad"))] * 10, lambda n, a: "no", budget=Budget(max_errors=3))
    assert result.reason is StopReason.ERRORS and result.errors == 3 and len(llm.calls) == 3


def test_a_review_that_breaks_refuses_rather_than_crashing_or_accepting():
    def broken(name, arguments):
        raise RuntimeError("card 4111111111111111 down")

    result, llm = run([turn(call("finish", message="x")), turn(call("finish", message="y"))], broken, budget=Budget(max_errors=2))
    assert result.reason is StopReason.ERRORS and "4111" not in repr(llm.calls)
    assert "could not be checked" in llm.calls[1]["messages"][-1]["content"]


def test_without_a_review_the_first_ending_is_final_as_before():
    result, _ = run([turn(call("finish", message="x"))], None)
    assert result.finished and result.errors == 0


def test_a_review_is_only_asked_about_endings_not_about_ordinary_tool_calls():
    asked = []
    run([turn(call("get_current_time")), turn(call("finish", message="x"))], lambda n, a: asked.append(n))
    assert asked == ["finish"]


# ------------------------------------------------------------------ a plan the code refuses can be corrected

def make(script, config=DEBIT_SPECIALIST, index=INDEX):
    shared = BankingSession(fidelity=Fidelity.TAU2)
    llm = FakeToolLLM(script)
    return Specialist(config, llm=llm, index=index, database=shared.database, vault=SecretVault()), llm, shared


def ask(spec, shared, intent):
    ctx = RouteContext(intent_id=intent.id, verified_user_id=ME, session=scoped(shared)(SessionRequest(intent.id)), mode="plan")
    return asyncio.run(spec.run(intent, ctx))


def limit_intent():
    return Intent(id="x", kind=K.CARD_CONTROLS, summary="raise my atm limit", slots={"card_id": CARD})


def limit_step(amount):
    return ("request_temporary_debit_card_limit_increase_8374", {"card_id": CARD, "limit_type": "atm", "new_limit": amount})


def test_a_limit_below_the_current_one_is_refused_with_the_banks_reason_and_the_corrected_plan_is_accepted():
    spec, llm, shared = make([turn(plan(limit_step(450))), turn(plan(limit_step(2000)))])
    outcome = ask(spec, shared, limit_intent())
    assert outcome.status is OutcomeStatus.NEEDS_CONFIRMATION
    assert outcome.proposal.steps[0].args["new_limit"] == 2000
    told = llm.calls[1]["messages"][-1]["content"]
    assert told.startswith("Not accepted: step 1") and "not higher than the current limit" in told
    assert shared.audit_log == () and len(llm.calls) == 2


def test_the_made_up_close_reason_from_the_live_run_is_refused_then_fixed():
    spec, llm, shared = make(
        [
            turn(plan(("close_debit_card_4721", {"card_id": CARD, "reason": "Lost wallet"}))),
            turn(plan(("close_debit_card_4721", {"card_id": CARD, "reason": "lost"}))),
        ]
    )
    intent = Intent(id="x", kind=K.CLOSE_CARD, summary="close it", slots={"card_id": CARD})
    outcome = ask(spec, shared, intent)
    assert outcome.status is OutcomeStatus.NEEDS_CONFIRMATION
    assert "not a reason a card can be closed for" in llm.calls[1]["messages"][-1]["content"]


def test_a_plan_that_forgot_a_tool_it_may_not_use_is_told_so():
    spec, llm, shared = make([turn(plan(("freeze_debit_card_3892", {"card_id": CARD}))), turn(plan(("close_debit_card_4721", {"card_id": CARD, "reason": "lost"})))])
    intent = Intent(id="x", kind=K.CLOSE_CARD, summary="close it", slots={"card_id": CARD})
    assert ask(spec, shared, intent).status is OutcomeStatus.NEEDS_CONFIRMATION
    assert "may not use" in llm.calls[1]["messages"][-1]["content"]


def test_the_reason_given_to_the_model_never_contains_a_secret_or_another_customers_data():
    vault_pin = "7351"
    spec, llm, shared = make([turn(plan(("reset_debit_card_pin_6284", {"card_id": CARD, "last_4_digits": "0000", "new_pin": "{{secret:pin}}"})))])
    spec._vault.store("pin", vault_pin)
    intent = Intent(id="x", kind=K.CHANGE_OR_RESET_PIN, summary="reset my pin", slots={"card_id": CARD})
    ask(spec, shared, intent)
    assert vault_pin not in json.dumps(llm.calls)
    other = make([turn(plan(("close_debit_card_4721", {"card_id": "dbc_kj93a7b2e1_blue", "reason": "lost"})))])
    spec2, llm2, shared2 = other
    ask(spec2, shared2, Intent(id="x", kind=K.CLOSE_CARD, summary="close it", slots={"account_id": ACCOUNT}))
    told = "".join(m["content"] for c in llm2.calls for m in c["messages"] if m["role"] == "tool")
    assert "dbc_kj93a7b2e1" not in told and "record_belongs_to_another_customer" in told


def test_a_plan_the_model_cannot_correct_ends_in_the_specific_reason_not_a_vague_stop():
    spec, llm, shared = make([turn(plan(limit_step(450)))])  # nothing more scripted: the model then fails
    outcome = ask(spec, shared, limit_intent())
    assert outcome.status is OutcomeStatus.FAILED and "not higher than the current limit" in outcome.error


def test_the_rules_tell_the_model_it_will_be_told_why():
    from agent.specialist import system_prompt

    assert "you are told why" in system_prompt(DEBIT_SPECIALIST, K.CLOSE_CARD)


# ------------------------------------------------------------------ the answerer and its citations

def test_an_answer_without_a_citation_is_sent_back_and_the_cited_one_is_accepted():
    spec, llm, shared = make(
        [
            turn(call("search_policy", query="blue account monthly fee")),
            turn(call("finish", message="It costs $10 a month.")),
            turn(call("finish", message=f"It costs $10 a month [{BLUE}].")),
        ],
        config=GENERAL_ANSWERER,
    )
    intent = Intent(id="q", kind=K.GENERAL_QUESTION, summary="what does the blue account cost")
    ctx = RouteContext(intent_id="q", verified_user_id=None, session=scoped(shared)(SessionRequest("q")), mode="plan")
    outcome = asyncio.run(spec.run(intent, ctx))
    assert outcome.status is OutcomeStatus.DONE and BLUE in outcome.text
    assert "cite at least one document" in llm.calls[2]["messages"][-1]["content"]


# ------------------------------------------------------------------ asking the customer

@pytest.mark.parametrize(
    "given, expected",
    [
        (["Transaction ID", "Card ID"], ["transaction_id", "card_id"]),
        (["transaction_id", "Transaction ID", "transaction-id"], ["transaction_id"]),
        (["Date of birth!", "  amount  "], ["date_of_birth", "amount"]),
        (["4821", "pin_2589", "x", "", "   "], []),
        (["address line 2"], ["address_line_2"]),
        ([5, None, {"a": 1}], []),
        ("card_id", []),
        (None, []),
    ],
)
def test_the_names_the_model_gives_are_tidied_and_anything_that_could_be_a_number_is_dropped(given, expected):
    assert _clean_details(given) == expected


def test_a_question_with_untidy_names_now_reaches_the_customer():
    spec, _, shared = make([turn(call("ask_customer", details=["Transaction ID", "Transaction Date"]))])
    intent = Intent(id="x", kind=K.DISPUTE_TRANSACTION, summary="dispute", slots={"account_id": ACCOUNT})
    outcome = ask(spec, shared, intent)
    assert outcome.status is OutcomeStatus.NEEDS_INFO and [q.slot for q in outcome.questions] == ["transaction_id", "transaction_date"]


def test_a_question_with_no_usable_name_is_sent_back_to_be_rewritten():
    spec, llm, shared = make([turn(call("ask_customer", details=["4821"])), turn(call("ask_customer", details=["card_id"]))])
    intent = Intent(id="x", kind=K.DISPUTE_TRANSACTION, summary="dispute", slots={"account_id": ACCOUNT})
    outcome = ask(spec, shared, intent)
    assert outcome.status is OutcomeStatus.NEEDS_INFO and outcome.questions[0].slot == "card_id"
    assert "name the details you need" in llm.calls[1]["messages"][-1]["content"]
