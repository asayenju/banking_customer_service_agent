"""Credit limit increases: every documented rule denies with its own reason, in the document's order;
an amount over the cap is a question; the reference answers are reproduced."""

from datetime import date

import pytest

from agent.specialists.credit import CREDIT_SPECIALIST
from agent.tier1 import run_tier1
from agent.workflows.base import HandOff, NeedInfo, Plan, Refuse, Reply, WorkflowContext
from agent.workflows.credit_limit import CreditLimit, LimitFacts
from banking_tools import BankingSession, Fidelity
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex

TODAY = date(2025, 11, 14)
INDEX = KnowledgeIndex({"doc_credit_cards_x": Document("doc_credit_cards_x", "Internal: x", "credit card procedure")})
TASKS = {t["id"]: t for t in load_tasks()}
ME = "lj82d4f1a9"  # Silver Rewards (mid tier), opened 05/20/2024
ACCT = "cc_lj82d4f1a9_silver"


@pytest.mark.parametrize("task_id", ["task_050", "task_051", "task_052", "task_053", "task_054"])
def test_the_reference_answer_is_reproduced_through_the_limit_workflow(task_id):
    result = run_tier1(TASKS[task_id], CREDIT_SPECIALIST, index=INDEX)
    assert result.error is None and result.excluded is None, result
    assert result.passed, result.mismatches
    assert result.bypasses == 0
    assert any(s.startswith("CREDIT_LIMIT_REQUEST") for s in result.segments)


def test_a_denied_request_can_be_resubmitted_and_approved_after_paying_the_balance_down():
    result = run_tier1(TASKS["task_051"], CREDIT_SPECIALIST, index=INDEX)
    assert sum(s.startswith("CREDIT_LIMIT_REQUEST") for s in result.segments) == 2


def test_the_limit_comes_before_the_dispute_and_replacement_in_the_mixed_tasks():
    result = run_tier1(TASKS["task_054"], CREDIT_SPECIALIST, index=INDEX)
    kinds = [s.split(":")[0] for s in result.segments]
    assert kinds == ["CREDIT_LIMIT_REQUEST", "DISPUTE_TRANSACTION", "ORDER_REPLACEMENT_CARD"]


# ---------------------------------------------------------------- helpers

def world(**changes):
    s = BankingSession(fidelity=Fidelity.TAU2)
    s.database.credit_card_accounts.data[ACCT].update({"credit_limit": "$10,000.00", "current_balance": "$1,000.00", **changes})
    for i in range(6):  # six on-time payments, newest first by date
        s.database.payment_history.data[f"p{i}"] = {
            "payment_id": f"p{i}", "credit_card_account_id": ACCT, "user_id": ME, "payment_date": f"{10 - i:02d}/15/2025", "amount": "$100.00", "status": "ON_TIME",
        }
    return s


def decide(s, amount=1000, user=ME, account=ACCT):
    ctx = WorkflowContext(s.database, user, TODAY)
    return CreditLimit().decide(ctx, LimitFacts(credit_card_account_id=account, requested_increase_amount=amount))


def tools(plan):
    return [(st.tool.rsplit("_", 1)[0], dict(st.args)) for st in plan.steps]


def reason_of(s, amount=1000):
    plan = decide(s, amount)
    assert isinstance(plan, Plan), plan
    (name, args) = tools(plan)[-1]
    return args.get("denial_reason") if name.startswith("deny") else "APPROVED"


def test_an_eligible_request_is_submitted_then_approved_at_the_old_limit_plus_the_increase():
    s = world()
    plan = decide(s, 2500)
    assert [t for t, _ in tools(plan)] == ["submit_credit_limit_increase_request", "approve_credit_limit_increase"]
    assert tools(plan)[0][1] == {"credit_card_account_id": ACCT, "requested_increase_amount": 2500}
    assert tools(plan)[1][1]["new_credit_limit"] == 12500


# ---------------------------------------------------------------- the cap is checked before anything is submitted

@pytest.mark.parametrize(
    "card_type, limit, amount, ok",
    [("Bronze Rewards Card", "$4,000.00", 1000, True), ("Bronze Rewards Card", "$4,000.00", 1001, False),
     ("Silver Rewards Card", "$10,000.00", 5000, True), ("Silver Rewards Card", "$10,000.00", 5001, False),
     ("Gold Rewards Card", "$5,000.00", 2500, True), ("Gold Rewards Card", "$5,000.00", 4000, False)],
)
def test_the_increase_cap_is_a_share_of_the_limit_by_tier_and_over_it_is_a_question(card_type, limit, amount, ok):
    s = world(card_type=card_type, credit_limit=limit)
    result = decide(s, amount)
    if ok:
        assert isinstance(result, Plan)
    else:
        assert isinstance(result, NeedInfo) and "at most" in result.questions[0].text and result.questions[0].slot == "requested_increase_amount"


# ---------------------------------------------------------------- one test per rule, and the order between them

def test_a_young_account_is_denied_for_age_by_tier():
    young = world(date_of_account_open="08/16/2025")  # 90 days: exactly the mid minimum
    assert reason_of(young) == "APPROVED"
    assert reason_of(world(date_of_account_open="08/17/2025")) == "insufficient_account_age"  # 89 days
    bronze = world(card_type="Bronze Rewards Card", credit_limit="$4,000.00", date_of_account_open="07/17/2025")  # 120 days
    assert reason_of(bronze, 1000) == "APPROVED"
    bronze = world(card_type="Bronze Rewards Card", credit_limit="$4,000.00", date_of_account_open="07/18/2025")  # 119 days
    assert reason_of(bronze, 1000) == "insufficient_account_age"


def approved_request(s, when, status="approved", key="cli1"):
    s.database.credit_limit_increase_requests.data[key] = {
        "request_id": key, "credit_card_account_id": ACCT, "user_id": ME, "status": status, "submitted_date": when, "requested_increase_amount": "$1,000.00",
    }


def test_the_cooldown_counts_approved_requests_only_from_their_submit_date_by_tier():
    s = world()
    approved_request(s, "2025-08-17")  # 89 days ago: inside the mid tier's 90
    assert reason_of(s) == "cooldown_period_active"
    s = world()
    approved_request(s, "2025-08-16")  # 90 days ago: over
    assert reason_of(s) == "APPROVED"
    s = world()
    approved_request(s, "2025-10-01", status="denied")
    approved_request(s, "2025-10-02", status="DENIED", key="cli2")
    assert reason_of(s) == "APPROVED", "a denial starts no cooldown"
    s = world()
    approved_request(s, "2025-10-01", status="PENDING")
    assert reason_of(s) == "APPROVED", "a pending record (the one a submission creates) is no cooldown"
    s = world()
    approved_request(s, "sometime")
    assert reason_of(s) == "cooldown_period_active", "an unreadable date counts"


def test_the_cooldown_follows_the_tier():
    bronze = world(card_type="Bronze Rewards Card", credit_limit="$4,000.00")
    approved_request(bronze, "2025-09-15")  # 60 days: over for premium, inside the entry tier's 120 (task 052)
    assert reason_of(bronze, 400) == "cooldown_period_active"
    gold = world(card_type="Gold Rewards Card")
    approved_request(gold, "2025-09-15")
    assert reason_of(gold) == "APPROVED"


def test_a_dispute_a_pending_replacement_and_bad_standing_each_deny_with_their_reason():
    s = world()
    s.database.transaction_disputes.data["d"] = {"dispute_id": "d", "user_id": ME, "transaction_id": "x", "status": "UNDER_REVIEW"}
    assert reason_of(s) == "pending_disputes"
    s = world()
    s.database.credit_card_orders.data["o"] = {"order_id": "o", "credit_card_account_id": ACCT, "user_id": ME, "status": "ORDERED", "order_date": "11/14/2025"}
    assert reason_of(s) == "pending_replacement_card"
    assert reason_of(world(account_status="PAST_DUE")) == "past_due_balance"
    assert reason_of(world(past_due_amount="$20.00")) == "past_due_balance"
    assert reason_of(world(past_due_amount="$0.00", account_status="ACTIVE")) == "APPROVED"


@pytest.mark.parametrize(
    "card_type, limit, balance, expected",
    [("Bronze Rewards Card", "$4,000.00", "$2,799.00", "APPROVED"), ("Bronze Rewards Card", "$4,000.00", "$2,800.00", "high_utilization"),
     ("Silver Rewards Card", "$10,000.00", "$7,999.00", "APPROVED"), ("Silver Rewards Card", "$10,000.00", "$8,000.00", "high_utilization"),
     ("Gold Rewards Card", "$10,000.00", "$8,999.00", "APPROVED"), ("Gold Rewards Card", "$10,000.00", "$9,000.00", "high_utilization")],
)
def test_utilization_must_be_below_the_tiers_threshold_not_at_it(card_type, limit, balance, expected):
    s = world(card_type=card_type, credit_limit=limit, current_balance=balance)
    assert reason_of(s, 400) == expected


def test_payment_history_needs_the_tiers_number_of_on_time_payments():
    s = world()  # mid: three on-time payments; six exist
    assert reason_of(s) == "APPROVED"
    s.database.payment_history.data["p0"]["status"] = "LATE"  # the newest one
    assert reason_of(s) == "insufficient_payment_history"
    s = world()
    for k in ("p3", "p4", "p5"):
        del s.database.payment_history.data[k]
    s.database.payment_history.data["p1"]["status"] = "LATE"
    assert reason_of(s) == "insufficient_payment_history"
    assert reason_of(world(card_type="Bronze Rewards Card", credit_limit="$4,000.00", current_balance="$0.00"), 400) == "APPROVED"
    s = world(card_type="Bronze Rewards Card", credit_limit="$4,000.00", current_balance="$0.00")
    del s.database.payment_history.data["p5"]  # five payments, six needed for the entry tier
    assert reason_of(s, 400) == "insufficient_payment_history"


def test_when_several_rules_fail_the_first_in_the_documents_order_is_the_reason():
    s = world(date_of_account_open="11/01/2025", current_balance="$9,500.00", past_due_amount="$5.00")
    s.database.payment_history.data["p0"]["status"] = "LATE"
    assert reason_of(s) == "insufficient_account_age"
    s = world(current_balance="$9,500.00", past_due_amount="$5.00")
    s.database.payment_history.data["p0"]["status"] = "LATE"
    assert reason_of(s) == "past_due_balance"


# ---------------------------------------------------------------- what is refused, asked or handed over

def test_missing_limit_untiered_card_closed_account_and_other_peoples_accounts():
    s = world()
    s.database.credit_card_accounts.data[ACCT].pop("credit_limit")
    assert isinstance(decide(s), HandOff)
    assert isinstance(decide(world(card_type="Mystery Card")), HandOff)
    assert isinstance(decide(world(status="CLOSED")), Reply)
    assert isinstance(decide(world(), account="cc_76ad9cc60e_gold"), Refuse)
    assert isinstance(decide(world(), account="cc_made_up"), Refuse)
    assert isinstance(decide(world(), amount=0), Refuse)
    assert isinstance(decide(world(), amount=-5), Refuse)


def test_a_request_of_the_same_amount_already_on_file_is_not_submitted_twice():
    from tau2.domains.banking_knowledge.utils import generate_credit_limit_increase_request_id

    s = world()
    key = generate_credit_limit_increase_request_id(ACCT, ME, 1000)
    s.database.credit_limit_increase_requests.data[key] = {"request_id": key, "credit_card_account_id": ACCT, "user_id": ME, "status": "PENDING"}
    plan = decide(s, 1000)
    assert [t for t, _ in tools(plan)] == ["approve_credit_limit_increase"]
