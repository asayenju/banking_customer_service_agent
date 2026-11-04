"""The credit write tools' rules: each validator accepts a good call and refuses each bad one, on
synthetic data. Another customer's account reads like one that does not exist."""

import asyncio
import json
from dataclasses import replace
from datetime import date

import pytest

from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.specialist import Specialist, ValidationContext
from agent.specialists import credit_rules as rules
from agent.specialists.credit import CREDIT_SPECIALIST, READBACKS, VALIDATORS
from agent.toolbox import SecretVault
from banking_tools import BankingSession, Fidelity
from banking_tools.catalog import get_registry
from knowledge.index import Document, KnowledgeIndex
from orchestrator.context import RouteContext, SessionRequest, scoped
from orchestrator.executor import OutcomeStatus
from orchestrator.types import Intent, IntentKind as K

ME = "lj82d4f1a9"  # a Silver Rewards card (mid tier), opened 05/20/2024, and two checking accounts
ACCT = "cc_lj82d4f1a9_silver"
CHECKING = "chk_lj82d4f1a9"
OTHER_ACCT = "cc_76ad9cc60e_gold"
TODAY = date(2025, 11, 14)
INDEX = KnowledgeIndex({"doc_credit_cards_x": Document("doc_credit_cards_x", "Internal: x", "credit card procedure")})


def world(**changes):
    shared = BankingSession(fidelity=Fidelity.TAU2)
    shared.database.credit_card_accounts.data[ACCT].update(changes)
    return shared


def v(shared, user=ME):
    return ValidationContext(shared.database, user, TODAY)


def add(shared, table, key, row):
    getattr(shared.database, table).data[key] = row


def order(status="SHIPPED", date_="11/10/2025", key="o1", account=ACCT):
    return {"order_id": key, "credit_card_account_id": account, "user_id": ME, "status": status, "order_date": date_}


def dispute_row(txn, status="UNDER_REVIEW", user=ME, key="d1"):
    return {"dispute_id": key, "transaction_id": txn, "user_id": user, "status": status}


def txn(shared, key="txn_a", amount="$156.78", when="10/02/2025", account=ACCT, user=ME):
    add(shared, "credit_card_transaction_history", key, {
        "transaction_id": key, "user_id": user, "credit_card_account_id": account, "credit_card_type": "Silver Rewards Card",
        "merchant_name": "Costco", "transaction_amount": amount, "transaction_date": when, "status": "COMPLETED",
    })
    return key


def reject(validator, shared, args, part, user=ME):
    reason = validator(v(shared, user), args)
    assert reason is not None and part in reason, reason


# ---------------------------------------------------------------- every credit write has a validator

def test_every_credit_write_tool_the_guard_lets_the_credit_kinds_use_has_a_validator():
    from agent.guard import allowed_writes

    reachable = {t for kind in CREDIT_SPECIALIST.kinds for t in allowed_writes(kind) if "debit" not in t}
    assert reachable == set(VALIDATORS), "a credit write without a validator is a hole"
    assert reachable == CREDIT_SPECIALIST_TOOLS
    assert all(get_registry()[t].kind.value == "write" for t in reachable)
    assert set(READBACKS) <= set(VALIDATORS)


CREDIT_SPECIALIST_TOOLS = {
    "file_credit_card_transaction_dispute_4829", "order_replacement_credit_card_7291", "log_credit_card_closure_reason_4521",
    "close_credit_card_account_7834", "apply_statement_credit_8472", "apply_credit_card_account_flag_6147",
    "pay_credit_card_from_checking_9182", "submit_credit_limit_increase_request_7392", "approve_credit_limit_increase_5847",
    "deny_credit_limit_increase_5848",
}


@pytest.mark.parametrize("tool", sorted(CREDIT_SPECIALIST_TOOLS - {"file_credit_card_transaction_dispute_4829"}))
def test_someone_elses_account_and_a_made_up_one_read_the_same(tool):
    shared = world()
    args = {"credit_card_account_id": OTHER_ACCT}
    other = VALIDATORS[tool](v(shared), args)
    made_up = VALIDATORS[tool](v(shared), {"credit_card_account_id": "cc_made_up"})
    assert other == made_up == "that credit card account does not exist"


@pytest.mark.parametrize("tool", sorted(CREDIT_SPECIALIST_TOOLS - {"file_credit_card_transaction_dispute_4829"}))
def test_a_closed_account_takes_no_further_change(tool):
    shared = world(status="CLOSED")
    assert "already closed" in VALIDATORS[tool](v(shared), {"credit_card_account_id": ACCT})


# ---------------------------------------------------------------- closing

def test_closing_is_allowed_with_nothing_in_the_way():
    shared = world(current_balance="$0.00")
    assert rules.close(v(shared), {"credit_card_account_id": ACCT}) is None


@pytest.mark.parametrize(
    "setup, part",
    [
        (lambda s: s.database.credit_card_accounts.data[ACCT].update(current_balance="$175.00"), "balance of $175.00"),
        (lambda s: s.database.credit_card_accounts.data[ACCT].update(current_balance="garbage"), "not on file"),
        (lambda s: add(s, "credit_card_orders", "o1", order("SHIPPED")), "replacement card is on its way"),
        (lambda s: add(s, "credit_card_orders", "o1", order("ORDERED")), "replacement card is on its way"),
        (lambda s: s.database.credit_card_accounts.data[ACCT].update(date_of_account_open="10/01/2025"), "at least 60 days"),
        (lambda s: s.database.credit_card_accounts.data[ACCT].update(date_of_account_open="nonsense"), "opened is not on file"),
        (lambda s: (txn(s), add(s, "transaction_disputes", "d1", dispute_row("txn_a"))), "dispute in progress"),
    ],
)
def test_closing_is_refused_for_each_blocker(setup, part):
    shared = world(current_balance="$0.00")
    setup(shared)
    reject(rules.close, shared, {"credit_card_account_id": ACCT}, part)


def test_a_dispute_of_a_bank_transaction_or_a_resolved_one_does_not_block_closing():
    shared = world(current_balance="$0.00")
    add(shared, "transaction_disputes", "d1", dispute_row("a_bank_transaction"))
    txn(shared)
    add(shared, "transaction_disputes", "d2", dispute_row("txn_a", status="RESOLVED", key="d2"))
    assert rules.close(v(shared), {"credit_card_account_id": ACCT}) is None


def test_exactly_sixty_days_old_can_be_closed_and_fifty_nine_cannot():
    sixty = world(current_balance="$0.00", date_of_account_open="09/15/2025")  # 60 days before 11/14/2025
    assert rules.close(v(sixty), {"credit_card_account_id": ACCT}) is None
    fifty_nine = world(current_balance="$0.00", date_of_account_open="09/16/2025")
    reject(rules.close, fifty_nine, {"credit_card_account_id": ACCT}, "at least 60 days")


def test_closing_is_read_back_from_the_database():
    shared = world()
    assert rules.close_readback(v(shared), {"credit_card_account_id": ACCT}) == "the account is not closed"
    shared.database.credit_card_accounts.data[ACCT].update(status="CLOSED")
    assert rules.close_readback(v(shared), {"credit_card_account_id": ACCT}) is None


# ---------------------------------------------------------------- the closure reason

def test_a_closure_reason_is_logged_once_and_only_from_the_list():
    shared = world()
    args = {"credit_card_account_id": ACCT, "closure_reason": "annual_fee"}
    assert rules.log_reason(v(shared), args) is None
    reject(rules.log_reason, shared, {**args, "closure_reason": "bored"}, "not a closure reason")
    add(shared, "credit_card_closure_reasons", "r1", {"record_id": "r1", "credit_card_account_id": ACCT, "closure_reason": "other"})
    reject(rules.log_reason, shared, args, "already on file")


# ---------------------------------------------------------------- statement credit and flags

@pytest.mark.parametrize(
    "card_type, amount, ok",
    [("Bronze Rewards Card", 5.0, True), ("Bronze Rewards Card", 20.0, False), ("Silver Rewards Card", 20.0, True),
     ("Silver Rewards Card", 50.0, False), ("Gold Rewards Card", 50.0, True), ("Crypto-Cash Back", 5.0, True),
     ("Crypto-Cash Back", 20.0, False), ("Unknown Card", 20.0, False)],
)
def test_the_retention_credit_is_the_tier_amount_and_nothing_else(card_type, amount, ok):
    shared = world(card_type=card_type)
    result = rules.statement_credit(v(shared), {"credit_card_account_id": ACCT, "amount": amount, "reason": "retention_offer"})
    assert (result is None) is ok, result


def test_statement_credit_basics():
    shared = world()
    base = {"credit_card_account_id": ACCT, "amount": 12.5, "reason": "goodwill_adjustment"}
    assert rules.statement_credit(v(shared), base) is None
    reject(rules.statement_credit, shared, {**base, "amount": 0}, "more than zero")
    reject(rules.statement_credit, shared, {**base, "amount": -3}, "more than zero")
    reject(rules.statement_credit, shared, {**base, "amount": True}, "more than zero")
    reject(rules.statement_credit, shared, {**base, "reason": "because"}, "not a reason")


FLAG = {
    "credit_card_account_id": ACCT, "flag_type": "annual_fee_waived", "expiration_date": "11/14/2026", "reason": "loyalty_benefit",
}


def test_an_annual_fee_waiver_lasts_one_year_for_customers_of_two_years_or_more():
    shared = world(date_of_account_open="11/10/2022")
    assert rules.flag(v(shared), FLAG) is None
    reject(rules.flag, shared, {**FLAG, "expiration_date": "11/13/2026"}, "one year from today")
    reject(rules.flag, shared, {**FLAG, "expiration_date": "11/14/2027"}, "one year from today")
    reject(rules.flag, shared, {**FLAG, "flag_type": "rewards_bonus", "expiration_date": "11/14/2027"}, "more than a year")
    reject(rules.flag, shared, {**FLAG, "expiration_date": "tomorrow"}, "MM/DD/YYYY")
    reject(rules.flag, shared, {**FLAG, "expiration_date": "11/01/2025"}, "future")
    reject(rules.flag, shared, {**FLAG, "flag_type": "gold"}, "not a flag type")
    young = world(date_of_account_open="06/20/2024")
    reject(rules.flag, young, FLAG, "two years or more")


# ---------------------------------------------------------------- paying the card

def test_paying_a_card_from_checking():
    shared = world(current_balance="$75.00")
    args = {"credit_card_account_id": ACCT, "checking_account_id": CHECKING, "amount": 75.0}
    assert rules.pay(v(shared), args) is None
    reject(rules.pay, shared, {**args, "amount": 80.0}, "less than that")
    reject(rules.pay, shared, {**args, "amount": 0}, "more than zero")
    reject(rules.pay, shared, {**args, "checking_account_id": "chk_538bfb9cba_nope"}, "checking account does not exist")
    reject(rules.pay, shared, {**args, "checking_account_id": "chk_76ad9cc60e"}, "checking account does not exist")
    shared.database.accounts.data[CHECKING]["current_holdings"] = "50.00"
    reject(rules.pay, shared, args, "not enough")
    shared.database.accounts.data[CHECKING].update(current_holdings="500.00", status="CLOSED")
    reject(rules.pay, shared, args, "open checking account")


# ---------------------------------------------------------------- replacement

REPL = {"credit_card_account_id": ACCT, "user_id": ME, "shipping_address": "1 Main St", "reason": "lost", "expedited_shipping": False}


def test_a_replacement_needs_a_reason_an_address_and_no_order_already_on_its_way():
    shared = world()
    assert rules.replacement(v(shared), REPL) is None
    reject(rules.replacement, shared, {**REPL, "reason": "bored"}, "not a reason")
    reject(rules.replacement, shared, {**REPL, "shipping_address": " "}, "shipping address")
    add(shared, "credit_card_orders", "o1", order("SHIPPED"))
    reject(rules.replacement, shared, REPL, "already been ordered")


@pytest.mark.parametrize("card_type, orders, ok", [
    ("Bronze Rewards Card", 1, True), ("Bronze Rewards Card", 2, False),
    ("Silver Rewards Card", 2, True), ("Silver Rewards Card", 3, False),
    ("Gold Rewards Card", 3, True), ("Gold Rewards Card", 4, False),
])
def test_the_replacement_limit_in_sixty_days_depends_on_the_tier(card_type, orders, ok):
    shared = world(card_type=card_type)
    for i in range(orders):  # delivered ones: they count against the limit but do not block
        add(shared, "credit_card_orders", f"o{i}", order("DELIVERED", "10/20/2025", key=f"o{i}"))
    assert (rules.replacement(v(shared), REPL) is None) is ok


def test_old_orders_do_not_count_toward_the_sixty_day_limit_and_an_untiered_card_goes_to_a_person():
    shared = world(card_type="Bronze Rewards Card")
    for i in range(3):
        add(shared, "credit_card_orders", f"o{i}", order("DELIVERED", "08/01/2025", key=f"o{i}"))
    assert rules.replacement(v(shared), REPL) is None
    reject(rules.replacement, world(card_type="Mystery Card"), REPL, "no tier")


# ---------------------------------------------------------------- disputes

DISPUTE = {
    "transaction_id": "txn_a", "card_action": "keep_active", "card_last_4_digits": "7823", "full_name": "Jo Doe", "user_id": ME,
    "phone": "555-0100", "email": "jo@example.com", "address": "1 Main St", "contacted_merchant": True,
    "purchase_date": "10/02/2025", "issue_noticed_date": "10/05/2025", "dispute_reason": "duplicate_charge",
    "resolution_requested": "full_refund", "eligible_for_provisional_credit": True, "partial_refund_amount": None,
}


def test_a_dispute_is_grounded_in_the_customers_own_transaction():
    shared = world()
    txn(shared)
    assert rules.dispute(v(shared), DISPUTE) is None
    reject(rules.dispute, shared, {**DISPUTE, "transaction_id": "txn_made_up"}, "transaction does not exist")
    txn(shared, "txn_theirs", user="76ad9cc60e")
    reject(rules.dispute, shared, {**DISPUTE, "transaction_id": "txn_theirs"}, "transaction does not exist")


@pytest.mark.parametrize(
    "change, part",
    [
        ({"dispute_reason": "bored"}, "not a dispute reason"),
        ({"card_action": "burn"}, "card action"),
        ({"card_last_4_digits": "78"}, "last four digits"),
        ({"card_last_4_digits": "78ab"}, "last four digits"),
        ({"phone": ""}, "phone"),
        ({"email": " "}, "email"),
        ({"resolution_requested": "partial_refund"}, "partial refund must be"),
        ({"resolution_requested": "partial_refund", "partial_refund_amount": 156.78}, "partial refund must be"),
        ({"resolution_requested": "partial_refund", "partial_refund_amount": 0}, "partial refund must be"),
        ({"partial_refund_amount": 10.0}, "no partial amount"),
        ({"resolution_requested": "reversal_of_charge"}, "full_refund or partial_refund"),
        ({"purchase_date": "10/03/2025"}, "not the date on the transaction"),
        ({"purchase_date": "2025"}, "MM/DD/YYYY"),
        ({"issue_noticed_date": "10/01/2025"}, "noticed it before"),
        ({"issue_noticed_date": "11/15/2025"}, "noticed it before"),
    ],
)
def test_a_dispute_with_a_wrong_detail_is_refused(change, part):
    shared = world()
    txn(shared)
    reject(rules.dispute, shared, {**DISPUTE, **change}, part)


def test_a_partial_refund_below_the_transaction_is_accepted_and_a_second_dispute_of_it_is_not():
    shared = world()
    txn(shared)
    partial = {**DISPUTE, "resolution_requested": "partial_refund", "partial_refund_amount": 50.0}
    assert rules.dispute(v(shared), partial) is None
    add(shared, "transaction_disputes", "d1", dispute_row("txn_a", status="RESOLVED"))
    reject(rules.dispute, shared, DISPUTE, "already been disputed")


# ---------------------------------------------------------------- limit increases

def with_limit(limit="$10,000.00", **more):
    return world(credit_limit=limit, **more)


def test_a_request_is_checked_against_the_tier_cap_before_it_is_submitted():
    shared = with_limit()  # Silver is mid tier: at most 50 percent
    args = {"credit_card_account_id": ACCT, "requested_increase_amount": 5000}
    assert rules.submit(v(shared), args) is None
    reject(rules.submit, shared, {**args, "requested_increase_amount": 5001}, "at most $5,000")
    bronze = with_limit("$4,000.00", card_type="Bronze Rewards Card")  # entry tier: 25 percent
    assert rules.submit(v(bronze), {**args, "requested_increase_amount": 1000}) is None
    reject(rules.submit, bronze, {**args, "requested_increase_amount": 1200}, "at most $1,000")


@pytest.mark.parametrize("amount", [0, -5, 10.5, True, "5000"])
def test_a_request_amount_must_be_whole_dollars_above_zero(amount):
    reject(rules.submit, with_limit(), {"credit_card_account_id": ACCT, "requested_increase_amount": amount}, "whole number")


def test_a_missing_credit_limit_or_an_untiered_card_goes_to_a_person_not_a_guess():
    args = {"credit_card_account_id": ACCT, "requested_increase_amount": 100}
    reject(rules.submit, world(), args, "no credit limit on file")  # most of the bank's accounts have none
    reject(rules.submit, with_limit(card_type="Mystery Card"), args, "no tier")
    reject(rules.approve, world(), {"credit_card_account_id": ACCT, "new_credit_limit": 100}, "no credit limit on file")


APPROVE = {"credit_card_account_id": ACCT, "new_credit_limit": 15000}


def test_an_approval_within_the_cap_is_accepted():
    assert rules.approve(v(with_limit()), APPROVE) is None


@pytest.mark.parametrize(
    "setup, part",
    [
        (lambda s: (txn(s), add(s, "transaction_disputes", "d1", dispute_row("txn_a", "SUBMITTED"))), "dispute is in progress"),
        (lambda s: add(s, "transaction_disputes", "d1", dispute_row("anything", "PENDING")), "dispute is in progress"),
        (lambda s: add(s, "credit_card_orders", "o1", order("PROCESSING")), "replacement card is on its way"),
        (lambda s: add(s, "credit_card_orders", "o1", order("ORDERED")), "replacement card is on its way"),
        (lambda s: s.database.credit_card_accounts.data[ACCT].update(account_status="PAST_DUE"), "not in good standing"),
        (lambda s: s.database.credit_card_accounts.data[ACCT].update(account_status="collections"), "not in good standing"),
        (lambda s: s.database.credit_card_accounts.data[ACCT].update(past_due_amount="$35.00"), "past-due balance"),
    ],
)
def test_the_hidden_reasons_the_banks_tool_refuses_an_approval_are_named_before_the_call(setup, part):
    shared = with_limit()
    setup(shared)
    reject(rules.approve, shared, APPROVE, part)


def test_a_past_due_amount_of_zero_does_not_block_and_a_resolved_dispute_or_delivered_order_does_not():
    shared = with_limit(past_due_amount="$0.00", account_status="ACTIVE")
    add(shared, "transaction_disputes", "d1", dispute_row("x", "RESOLVED"))
    add(shared, "credit_card_orders", "o1", order("DELIVERED"))
    assert rules.approve(v(shared), APPROVE) is None


@pytest.mark.parametrize("new, part", [(10000, "higher"), (9000, "higher"), (15001, "at most $5,000"), (12500.5, "whole number"), (True, "whole number")])
def test_an_approval_must_raise_the_limit_by_no_more_than_the_cap(new, part):
    reject(rules.approve, with_limit(), {**APPROVE, "new_credit_limit": new}, part)


def test_an_approval_is_read_back_from_the_limit():
    shared = with_limit()
    assert rules.approve_readback(v(shared), APPROVE) == "the limit was not changed"
    shared.database.credit_card_accounts.data[ACCT]["credit_limit"] = "$15000.00"
    assert rules.approve_readback(v(shared), APPROVE) is None


def test_a_denial_needs_one_of_the_banks_reasons():
    shared = with_limit()
    assert rules.deny(v(shared), {"credit_card_account_id": ACCT, "denial_reason": "cooldown_period_active"}) is None
    reject(rules.deny, shared, {"credit_card_account_id": ACCT, "denial_reason": "i said so"}, "not a denial reason")


# ---------------------------------------------------------------- a read-back that fails stops the run

def test_a_write_that_reports_success_but_changes_nothing_ends_the_run_as_failed():
    """The bank's tool for a closure reason says it worked and stores nothing on a duplicate; a read-back
    catches the same shape of problem for any tool that is given one."""
    shared = BankingSession(fidelity=Fidelity.TAU2)
    never_there = lambda ctx, args: "the change is not in the records"
    config = replace(CREDIT_SPECIALIST, validators={}, readbacks={"log_credit_card_closure_reason_4521": never_there}, prepare=None, workflows=())
    step = {"tool": "log_credit_card_closure_reason_4521", "description": "x", "arguments_json": json.dumps(
        {"credit_card_account_id": ACCT, "user_id": ME, "closure_reason": "annual_fee"})}
    plan = ToolCall(id="p", name="propose_plan", arguments_json=json.dumps({"steps": [step]}))
    spec = Specialist(config, llm=FakeToolLLM([AssistantTurn(tool_calls=(plan,), usage=Usage(input_tokens=5))]), index=INDEX,
                      database=shared.database, vault=SecretVault())
    intent = Intent(id="x", kind=K.CLOSE_CARD, summary="close it", slots={"credit_card_account_id": ACCT})

    def context(mode, approved=None):
        session = scoped(shared)(SessionRequest(intent.id, mode=mode, approved=approved))
        return RouteContext(intent_id=intent.id, verified_user_id=ME, session=session, mode=mode, approved=approved)

    proposed = asyncio.run(spec.run(intent, context("plan")))
    assert proposed.status is OutcomeStatus.NEEDS_CONFIRMATION
    done = asyncio.run(spec.run(intent, context("execute", proposed.proposal)))
    assert done.status is OutcomeStatus.FAILED and "reported done but the change is not in the records" in done.error
