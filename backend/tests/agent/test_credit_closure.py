"""Closing a credit card with retention: eligibility from the records, at most two offers asked as
questions, a close only in a plan the customer sees, and never after an accepted offer."""

import asyncio
import json
from datetime import date

import pytest

from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist
from agent.specialists.credit import CREDIT_SPECIALIST
from agent.tier1 import run_tier1
from agent.toolbox import SecretVault
from agent.workflows.base import HandOff, NeedInfo, Plan, Refuse, Reply, WorkflowContext
from agent.workflows.credit_closure import AFTER_CLOSING, ClosureFacts, CreditClosure
from banking_tools import BankingSession, Fidelity
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.executor import OutcomeStatus
from orchestrator.types import Intent, IntentKind as K

TODAY = date(2025, 11, 14)
INDEX = KnowledgeIndex({"doc_credit_cards_x": Document("doc_credit_cards_x", "Internal: x", "credit card procedure")})
TASKS = {t["id"]: t for t in load_tasks()}
ME = "lj82d4f1a9"  # Silver Rewards (mid tier), opened 05/20/2024: under two years
ACCT = "cc_lj82d4f1a9_silver"
CHECKING = "chk_lj82d4f1a9"


@pytest.mark.parametrize("task_id", ["task_043", "task_044", "task_045", "task_047", "task_049"])
def test_the_reference_answer_is_reproduced_through_the_closure_workflow(task_id):
    result = run_tier1(TASKS[task_id], CREDIT_SPECIALIST, index=INDEX)
    assert result.error is None and result.excluded is None, result
    assert result.passed, result.mismatches
    assert result.bypasses == 0 and result.segments


def test_task_048_differs_from_the_reference_only_in_the_crypto_card_credit_which_the_policy_sets_at_5_not_20():
    result = run_tier1(TASKS["task_048"], CREDIT_SPECIALIST, index=INDEX)
    assert result.error is None
    assert not result.passed
    assert len(result.mismatches) == 2 and all(m.startswith("credit_card_transaction_history/") for m in result.mismatches), result.mismatches


def test_task_049_gets_the_five_dollar_credit_and_closes_only_the_card_with_earlier_retention():
    result = run_tier1(TASKS["task_049"], CREDIT_SPECIALIST, index=INDEX)
    assert result.passed
    assert len(result.segments) == 2, "one request per card"


# ---------------------------------------------------------------- helpers

def world(**changes):
    s = BankingSession(fidelity=Fidelity.TAU2)
    s.database.credit_card_accounts.data[ACCT].update({"current_balance": "$0.00", **changes})
    return s


def facts(**changes):
    base = dict(credit_card_account_id=ACCT, reason=None, pay_from_checking_account_id=None, waiver_answer=None, credit_answer=None, keeps_card=False)
    return ClosureFacts(**{**base, **changes})


def decide(s, **changes):
    return CreditClosure().decide(WorkflowContext(s.database, ME, TODAY), facts(**changes))


def tools(plan):
    return [st.tool.rsplit("_", 1)[0] for st in plan.steps]


def long_standing(**changes):
    return world(date_of_account_open="11/10/2022", **changes)


def record(s, when, key="r1"):
    s.database.credit_card_closure_reasons.data[key] = {
        "record_id": key, "credit_card_account_id": ACCT, "user_id": ME, "closure_reason": "other", "logged_at": when, "status": "LOGGED",
    }


# ---------------------------------------------------------------- eligibility

def test_each_blocker_is_a_plain_reply_and_no_offer_is_made():
    s = world()
    s.database.transaction_disputes.data["d"] = {"dispute_id": "d", "user_id": ME, "transaction_id": "t", "status": "UNDER_REVIEW"}
    s.database.credit_card_transaction_history.data["t"] = {"transaction_id": "t", "user_id": ME, "credit_card_account_id": ACCT, "transaction_amount": "$5.00", "transaction_date": "10/01/2025"}
    assert isinstance(decide(s, reason="annual_fee"), Reply)
    s = world()
    s.database.credit_card_orders.data["o"] = {"order_id": "o", "credit_card_account_id": ACCT, "user_id": ME, "status": "SHIPPED", "order_date": "11/10/2025"}
    assert "replacement" in decide(s, reason="annual_fee").text
    assert "60 days" in decide(world(date_of_account_open="09/16/2025"), reason="annual_fee").text
    assert isinstance(decide(world(status="CLOSED")), Reply)


def test_a_balance_is_paid_first_in_its_own_plan_from_a_checking_account_the_customer_names():
    s = world(current_balance="$75.00")
    asked = decide(s)
    assert isinstance(asked, NeedInfo) and "$75.00" in asked.questions[0].text
    plan = decide(s, pay_from_checking_account_id=CHECKING)
    assert isinstance(plan, Plan) and tools(plan) == ["pay_credit_card_from_checking"]
    assert dict(plan.steps[0].args)["amount"] == 75.0, "the whole balance, worked out here"


def test_the_follow_up_after_paying_asks_why_and_after_closing_reminds_about_rewards_and_the_fee_refund():
    c = CreditClosure()
    ctx = WorkflowContext(world().database, ME, TODAY)
    from agent.workflows.base import Step

    assert "why" in c.after(ctx, [Step("pay_credit_card_from_checking_9182", {})]).lower()
    assert c.after(ctx, [Step("close_credit_card_account_7834", {})]) == AFTER_CLOSING
    assert "45 days" in AFTER_CLOSING and "37 days" in AFTER_CLOSING
    assert c.after(ctx, [Step("apply_statement_credit_8472", {})]) == ""


@pytest.mark.parametrize("card_type", ["Mystery Card"])
def test_an_untiered_card_goes_to_a_person_and_other_peoples_accounts_are_refused(card_type):
    assert isinstance(decide(world(card_type=card_type)), HandOff)
    assert isinstance(decide(world(), credit_card_account_id="cc_76ad9cc60e_gold"), Refuse)
    assert isinstance(decide(world(), credit_card_account_id="cc_made_up"), Refuse)


# ---------------------------------------------------------------- retention already tried

def test_a_reason_logged_in_the_last_year_means_the_card_is_closed_with_no_offers():
    s = world()
    record(s, "09/15/2025")
    plan = decide(s)
    assert tools(plan) == ["close_credit_card_account"], "no reason is asked and nothing is logged"


def test_a_record_older_than_a_year_allows_retention_but_cannot_be_logged_again():
    s = world()
    record(s, "09/15/2024")
    ask = decide(s)
    assert isinstance(ask, NeedInfo) and ask.questions[0].slot == "closure_reason"
    plan = decide(s, reason="not_using_card", credit_answer="yes")
    assert tools(plan) == ["apply_statement_credit"], "the bank's tool cannot log a second reason, so the log is left out"


def test_a_record_with_an_unreadable_date_counts_as_recent():
    s = world()
    record(s, "sometime")
    assert tools(decide(s)) == ["close_credit_card_account"]


# ---------------------------------------------------------------- the offers

def test_the_reason_is_asked_first():
    ask = decide(world())
    assert isinstance(ask, NeedInfo) and ask.questions[0].slot == "closure_reason"


def test_a_customer_of_two_years_who_dislikes_the_fee_is_offered_the_waiver_first():
    s = long_standing()
    ask = decide(s, reason="annual_fee")
    assert isinstance(ask, NeedInfo) and ask.questions[0].slot == "waiver_answer" and "annual fee" in ask.questions[0].text
    plan = decide(s, reason="annual_fee", waiver_answer="yes")
    assert tools(plan) == ["log_credit_card_closure_reason", "apply_credit_card_account_flag"]
    flag = dict(plan.steps[1].args)
    assert flag["flag_type"] == "annual_fee_waived" and flag["reason"] == "loyalty_benefit" and flag["expiration_date"] == "11/14/2026"


def test_a_newer_customer_gets_no_waiver_and_no_downgrade_but_the_credit_offer():
    s = world()  # opened 05/20/2024: under two years
    ask = decide(s, reason="annual_fee")
    assert ask.questions[0].slot == "credit_answer"
    assert "waive" not in ask.questions[0].text.lower() and "downgrade" not in ask.questions[0].text.lower()
    assert "$20" in ask.questions[0].text, "Silver is mid tier"


def test_declining_the_waiver_moves_on_to_the_credit_offer():
    ask = decide(long_standing(), reason="annual_fee", waiver_answer="no")
    assert ask.questions[0].slot == "credit_answer"


@pytest.mark.parametrize("card_type, amount", [("Bronze Rewards Card", 5.0), ("Silver Rewards Card", 20.0), ("Gold Rewards Card", 50.0), ("Crypto-Cash Back", 5.0)])
def test_the_credit_is_the_tiers_amount(card_type, amount):
    s = world(card_type=card_type)
    plan = decide(s, reason="found_better_card", credit_answer="yes")
    assert tools(plan) == ["log_credit_card_closure_reason", "apply_statement_credit"]
    assert dict(plan.steps[1].args) == {"credit_card_account_id": ACCT, "amount": amount, "reason": "retention_offer"}
    assert f"${amount:,.0f}" in decide(s, reason="found_better_card").questions[0].text


@pytest.mark.parametrize("reason", ["not_using_card", "found_better_card", "unhappy_with_rewards", "negative_experience"])
def test_the_credit_offer_carries_a_line_of_advice_for_the_reason(reason):
    text = decide(world(), reason=reason).questions[0].text
    assert text.startswith(("Setting up", "If another card", "Check which", "I'm sorry"))


def test_no_offer_ever_mentions_a_downgrade_an_upgrade_or_applying():
    for reason in ("annual_fee", "found_better_card", "not_using_card", "unhappy_with_rewards", "negative_experience", "simplifying_finances", "other"):
        for s in (world(), long_standing()):
            result = decide(s, reason=reason)
            text = " ".join(q.text for q in result.questions).lower() if isinstance(result, NeedInfo) else ""
            assert "downgrade" not in text and "upgrade" not in text and "i will apply" not in text


def test_keeping_the_card_logs_the_reason_and_closes_nothing():
    plan = decide(world(), reason="found_better_card", keeps_card=True)
    assert tools(plan) == ["log_credit_card_closure_reason"]


def test_declining_every_offer_logs_and_closes_in_a_plan_the_customer_sees():
    plan = decide(world(), reason="simplifying_finances", credit_answer="no")
    assert tools(plan) == ["log_credit_card_closure_reason", "close_credit_card_account"]
    plan = decide(long_standing(), reason="annual_fee", waiver_answer="no", credit_answer="no")
    assert tools(plan) == ["log_credit_card_closure_reason", "close_credit_card_account"]


@pytest.mark.parametrize("answers", [dict(waiver_answer="yes"), dict(credit_answer="yes"), dict(keeps_card=True)])
def test_a_close_is_never_in_a_plan_where_an_offer_was_accepted_or_the_card_kept(answers):
    for s in (world(), long_standing()):
        result = decide(s, reason="annual_fee", **answers)
        if isinstance(result, Plan):
            assert "close_credit_card_account" not in tools(result)


def test_the_close_never_comes_without_an_answer_to_the_offer():
    for reason in ("annual_fee", "found_better_card", "other"):
        result = decide(world(), reason=reason)
        assert isinstance(result, NeedInfo), "no plan, so no close, until the customer has answered"


# ---------------------------------------------------------------- the whole thing through the specialist

def run(spec, shared, intent, mode, approved=None):
    session = scoped(shared)(SessionRequest(intent.id, mode=mode, approved=approved))
    ctx = RouteContext(intent_id=intent.id, verified_user_id=ME, session=session, mode=mode, approved=approved)
    return asyncio.run(spec.run(intent, ctx))


def test_a_closed_card_ends_with_the_rewards_and_fee_refund_reminder():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    shared.database.credit_card_accounts.data[ACCT]["current_balance"] = "$0.00"
    record = shared.database.credit_card_closure_reasons.data
    record["r1"] = {"record_id": "r1", "credit_card_account_id": ACCT, "user_id": ME, "closure_reason": "other", "logged_at": "09/15/2025", "status": "LOGGED"}
    # The scripted model ends the run with the workflow's form, as a real one would.
    form = {"credit_card_account_id": ACCT, "reason": None, "pay_from_checking_account_id": None, "waiver_answer": None, "credit_answer": None, "keeps_card": False}
    llm = FakeToolLLM([AssistantTurn(tool_calls=(ToolCall(id="c", name="start_credit_closure", arguments_json=json.dumps(form)),), usage=Usage(input_tokens=5))])
    spec = Specialist(CREDIT_SPECIALIST, llm=llm, index=INDEX, database=shared.database, vault=SecretVault())
    intent = Intent(id="i", kind=K.CLOSE_CARD, summary="close it", slots={"credit_card_account_id": ACCT})
    proposed = run(spec, shared, intent, "plan")
    assert proposed.status is OutcomeStatus.NEEDS_CONFIRMATION
    done = run(spec, shared, intent, "execute", proposed.proposal)
    assert done.status is OutcomeStatus.DONE and AFTER_CLOSING in done.text
    assert shared.database.credit_card_accounts.data[ACCT].get("status") == "CLOSED"
