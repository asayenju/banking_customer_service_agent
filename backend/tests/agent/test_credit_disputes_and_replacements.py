"""Credit disputes and replacements: the workflows' rules, and the replay of the reference answers
for the credit tasks that file disputes or order a replacement (offline, scripted model)."""

from datetime import date

import pytest

from agent.specialists.credit import CREDIT_SPECIALIST
from agent.tier1 import gold_steps, run_tier1
from agent.workflows.base import HandOff, NeedInfo, Plan, Refuse, Reply, WorkflowContext
from agent.workflows.credit_dispute import CreditDispute, CreditDisputeFacts, CreditDisputeRequest, provisional_credit
from agent.workflows.credit_replacement import CreditReplacement, CreditReplacementFacts, NewCreditCard
from banking_tools import BankingSession, Fidelity
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex

TODAY = date(2025, 11, 14)
INDEX = KnowledgeIndex({"doc_credit_cards_x": Document("doc_credit_cards_x", "Internal: x", "credit card procedure")})
TASKS = {t["id"]: t for t in load_tasks()}
ME = "lj82d4f1a9"  # Silver Rewards (mid tier), opened 05/20/2024: about 543 days old
ACCT = "cc_lj82d4f1a9_silver"


# ---------------------------------------------------------------- the reference answers, replayed

@pytest.mark.parametrize("task_id", ["task_031", "task_036", "task_037", "task_038", "task_039", "task_040", "task_041"])
def test_the_reference_answer_is_reproduced_through_the_credit_workflows(task_id):
    result = run_tier1(TASKS[task_id], CREDIT_SPECIALIST, index=INDEX)
    assert result.error is None and result.excluded is None, result
    assert result.passed, result.mismatches
    assert result.bypasses == 0
    assert result.segments, "the specialist, not direct code, made the changes"


def test_a_dispute_and_a_replacement_in_one_request_are_two_segments():
    result = run_tier1(TASKS["task_037"], CREDIT_SPECIALIST, index=INDEX)
    assert any(s.startswith("DISPUTE_TRANSACTION") for s in result.segments)
    assert any(s.startswith("ORDER_REPLACEMENT_CARD") for s in result.segments)


# ---------------------------------------------------------------- helpers

def shared():
    return BankingSession(fidelity=Fidelity.TAU2)


def ctx(s, user=ME):
    return WorkflowContext(s.database, user, TODAY)


def add_txn(s, key="txn_a", amount="$156.78", when="10/02/2025", user=ME, account=ACCT, card_type="Silver Rewards Card"):
    s.database.credit_card_transaction_history.data[key] = {
        "transaction_id": key, "user_id": user, "credit_card_account_id": account, "credit_card_type": card_type,
        "merchant_name": "Costco", "transaction_amount": amount, "transaction_date": when, "status": "COMPLETED",
    }
    return key


def dispute(txn="txn_a", **changes):
    base = dict(
        transaction_id=txn, reason="duplicate_charge", noticed_date="10/05/2025", contacted_merchant=True,
        card_last_4_digits="4821", partial_refund_amount=None, cancel_and_reissue=False,
    )
    return CreditDisputeFacts(**{**base, **changes})


def decide(s, *items, user=ME):
    return CreditDispute().decide(ctx(s, user), CreditDisputeRequest(disputes=list(items)))


# ---------------------------------------------------------------- the dispute form is filled in by code

def test_the_contact_details_dates_and_provisional_credit_come_from_the_records_not_the_model():
    s = shared()
    add_txn(s)
    plan = decide(s, dispute())
    assert isinstance(plan, Plan)
    (step,) = plan.steps
    user = s.database.users.data[ME]
    assert step.tool == "file_credit_card_transaction_dispute_4829"
    assert step.args["full_name"] == user["name"] and step.args["email"] == user["email"]
    assert step.args["phone"] == user["phone_number"] and step.args["address"] == user["address"]
    assert step.args["purchase_date"] == "10/02/2025" and step.args["issue_noticed_date"] == "10/05/2025"
    assert step.args["resolution_requested"] == "full_refund" and "partial_refund_amount" not in step.args
    assert step.args["eligible_for_provisional_credit"] is True and step.args["card_action"] == "keep_active"


def test_the_customer_asks_for_the_card_to_be_cancelled_and_that_is_all_it_records():
    s = shared()
    add_txn(s)
    (step,) = decide(s, dispute(cancel_and_reissue=True)).steps
    assert step.args["card_action"] == "cancel_and_reissue"


@pytest.mark.parametrize(
    "facts, expected",
    [
        (dict(reason="unauthorized_fraudulent_charge", contacted_merchant=False), True),
        (dict(reason="duplicate_charge", contacted_merchant=True), True),
        (dict(reason="duplicate_charge", contacted_merchant=False), False),  # not fraud: the merchant must be tried first
        (dict(reason="goods_services_not_as_described"), False),
        (dict(reason="incorrect_amount", partial_refund_amount=50.0), False),
        (dict(reason="canceled_subscription_still_charging"), False),
        (dict(reason="refund_never_processed"), False),
        (dict(reason="goods_services_not_received", noticed_date="11/10/2025"), True),  # bought 10/02: 43 days ago
    ],
)
def test_provisional_credit_follows_the_documents_reasons(facts, expected):
    s = shared()
    add_txn(s)
    (step,) = decide(s, dispute(**facts)).steps
    assert step.args["eligible_for_provisional_credit"] is expected


def test_goods_not_received_needs_a_purchase_more_than_thirty_days_old():
    s = shared()
    add_txn(s, when="10/15/2025")  # 30 days before 11/14
    add_txn(s, "txn_old", when="10/14/2025")  # 31 days
    young = decide(s, dispute(reason="goods_services_not_received", noticed_date="10/20/2025")).steps[0]
    old = decide(s, dispute("txn_old", reason="goods_services_not_received", noticed_date="10/20/2025")).steps[0]
    assert young.args["eligible_for_provisional_credit"] is False
    assert old.args["eligible_for_provisional_credit"] is True


@pytest.mark.parametrize("amount, expected", [("$24.99", False), ("$25.00", True), ("$5,000.00", True), ("$5,000.01", False)])
def test_provisional_credit_needs_at_least_25_dollars_and_at_most_the_cards_cap(amount, expected):
    s = shared()  # Silver: cap $5,000
    add_txn(s, amount=amount)
    (step,) = decide(s, dispute()).steps
    assert step.args["eligible_for_provisional_credit"] is expected


def test_a_young_account_gets_no_provisional_credit():
    s = shared()
    add_txn(s)
    s.database.credit_card_accounts.data[ACCT]["date_of_account_open"] = "09/16/2025"  # 59 days
    assert decide(s, dispute()).steps[0].args["eligible_for_provisional_credit"] is False
    s.database.credit_card_accounts.data[ACCT]["date_of_account_open"] = "09/15/2025"  # 60 days
    assert decide(s, dispute()).steps[0].args["eligible_for_provisional_credit"] is True


def test_only_the_first_three_in_a_year_get_provisional_credit_counting_the_ones_already_filed():
    s = shared()
    keys = [add_txn(s, f"txn_{i}", when="10/02/2025") for i in range(5)]
    plan = decide(s, *[dispute(k) for k in keys])
    assert [st.args["eligible_for_provisional_credit"] for st in plan.steps] == [True, True, True, False, False]
    s.database.transaction_disputes.data["old"] = {"dispute_id": "old", "user_id": ME, "transaction_id": "other", "status": "UNDER_REVIEW", "submitted_at": "09/01/2025"}
    plan = decide(s, *[dispute(k) for k in keys])
    assert [st.args["eligible_for_provisional_credit"] for st in plan.steps] == [True, True, False, False, False]


def test_an_old_dispute_does_not_count_and_one_with_no_date_does():
    s = shared()
    key = add_txn(s)
    s.database.transaction_disputes.data["ancient"] = {"dispute_id": "a", "user_id": ME, "transaction_id": "x", "status": "RESOLVED", "submitted_at": "01/01/2024"}
    for n in range(3):
        s.database.transaction_disputes.data[f"undated{n}"] = {"dispute_id": f"u{n}", "user_id": ME, "transaction_id": f"y{n}", "status": "RESOLVED"}
    assert decide(s, dispute(key)).steps[0].args["eligible_for_provisional_credit"] is False, "three undated ones count"
    for n in range(3):
        del s.database.transaction_disputes.data[f"undated{n}"]
    assert decide(s, dispute(key)).steps[0].args["eligible_for_provisional_credit"] is True


def test_the_cap_is_by_card_type():
    for card_type, cap in [("Bronze Rewards Card", 2500.0), ("Crypto-Cash Back", 2500.0), ("Gold Rewards Card", 10000.0), ("Platinum Rewards Card", 15000.0), ("Diamond Elite Card", 25000.0)]:
        s = shared()
        s.database.credit_card_accounts.data[ACCT]["card_type"] = card_type
        add_txn(s, amount=f"${cap:,.2f}", card_type=card_type)
        assert decide(s, dispute()).steps[0].args["eligible_for_provisional_credit"] is True, card_type
        add_txn(s, amount=f"${cap + 0.01:,.2f}", card_type=card_type)
        assert decide(s, dispute()).steps[0].args["eligible_for_provisional_credit"] is False, card_type


# ---------------------------------------------------------------- what the form is refused or asked for

def test_someone_elses_or_an_unknown_transaction_is_refused_in_the_same_words():
    s = shared()
    add_txn(s, "txn_theirs", user="76ad9cc60e", account="cc_76ad9cc60e_gold")
    a, b = decide(s, dispute("txn_theirs")), decide(s, dispute("txn_made_up"))
    assert isinstance(a, Refuse) and isinstance(b, Refuse)
    assert a.reason.replace("txn_theirs", "X") == b.reason.replace("txn_made_up", "X")


def test_a_charge_already_disputed_is_told_not_filed_twice_and_a_repeat_in_the_list_is_refused():
    s = shared()
    add_txn(s)
    assert isinstance(decide(s, dispute(), dispute()), Refuse)
    s.database.transaction_disputes.data["d1"] = {"dispute_id": "d1", "user_id": ME, "transaction_id": "txn_a", "status": "RESOLVED"}
    assert isinstance(decide(s, dispute()), Reply)


def test_missing_last_four_digits_is_a_question_and_wrong_ones_are_refused():
    s = shared()
    add_txn(s)
    asked = decide(s, dispute(card_last_4_digits=None))
    assert isinstance(asked, NeedInfo) and asked.questions[0].slot == "card_last_4_digits"
    assert isinstance(decide(s, dispute(card_last_4_digits="48")), Refuse)
    s.database.credit_card_accounts.data[ACCT]["card_last_4_digits"] = "7823"
    assert isinstance(decide(s, dispute(card_last_4_digits="4821")), Refuse)
    assert isinstance(decide(s, dispute(card_last_4_digits="7823")), Plan)


@pytest.mark.parametrize("noticed", ["10/01/2025", "11/15/2025", "later", ""])
def test_a_noticed_date_before_the_charge_after_today_or_unreadable_is_refused(noticed):
    s = shared()
    add_txn(s)
    assert isinstance(decide(s, dispute(noticed_date=noticed)), Refuse)


@pytest.mark.parametrize("partial", [0.0, -1.0, 156.78, 200.0])
def test_a_partial_refund_must_be_between_zero_and_the_charge(partial):
    s = shared()
    add_txn(s)
    assert isinstance(decide(s, dispute(reason="incorrect_amount", partial_refund_amount=partial)), Refuse)


def test_a_partial_refund_is_recorded_as_one():
    s = shared()
    add_txn(s)
    (step,) = decide(s, dispute(reason="incorrect_amount", partial_refund_amount=50.0)).steps
    assert step.args["resolution_requested"] == "partial_refund" and step.args["partial_refund_amount"] == 50.0


def test_an_empty_list_is_refused_and_missing_contact_details_go_to_a_person():
    s = shared()
    assert isinstance(decide(s), Refuse)
    add_txn(s)
    s.database.users.data[ME]["phone_number"] = ""
    assert isinstance(decide(s, dispute()), HandOff)


def test_a_charge_on_a_card_that_cannot_be_told_apart_goes_to_a_person():
    s = shared()
    key = add_txn(s, account="")
    s.database.credit_card_transaction_history.data[key].pop("credit_card_account_id")
    s.database.credit_card_accounts.data["cc_dup"] = {**s.database.credit_card_accounts.data[ACCT], "account_id": "cc_dup"}
    assert isinstance(decide(s, dispute()), HandOff)


def test_a_dispute_on_a_card_replaced_earlier_in_the_visit_is_still_filed():
    s = shared()
    add_txn(s)
    s.database.credit_card_accounts.data[ACCT].update(status="CLOSED")
    assert isinstance(decide(s, dispute()), Plan), "the replacement closes the old card; its charges are still disputable"


# ---------------------------------------------------------------- replacements

def replace(**changes):
    base = dict(credit_card_account_id=ACCT, reason="lost", expedited=False, accepts_expedite_fee=False, shipping_address=None)
    return NewCreditCard(**{**base, **changes})


def order(s, *items, user=ME):
    return CreditReplacement().decide(ctx(s, user), CreditReplacementFacts(replacements=list(items)))


def test_a_replacement_uses_the_address_on_file_unless_the_customer_gives_another():
    s = shared()
    (step,) = order(s, replace()).steps
    assert step.tool == "order_replacement_credit_card_7291"
    assert step.args == {
        "credit_card_account_id": ACCT, "shipping_address": s.database.users.data[ME]["address"], "reason": "lost", "expedited_shipping": False,
    }
    (other,) = order(s, replace(shipping_address="1 New St")).steps
    assert other.args["shipping_address"] == "1 New St"


@pytest.mark.parametrize("card_type, fee", [("Bronze Rewards Card", 15), ("Silver Rewards Card", 10), ("Crypto-Cash Back", 15)])
def test_expedited_delivery_asks_for_the_fee_of_the_tier_before_a_plan(card_type, fee):
    s = shared()
    s.database.credit_card_accounts.data[ACCT]["card_type"] = card_type
    asked = order(s, replace(expedited=True))
    assert isinstance(asked, NeedInfo) and f"${fee}" in asked.questions[0].text
    assert isinstance(order(s, replace(expedited=True, accepts_expedite_fee=True)), Plan)


def test_expedited_is_free_for_gold_and_above_and_standard_is_always_free():
    s = shared()
    s.database.credit_card_accounts.data[ACCT]["card_type"] = "Gold Rewards Card"
    assert isinstance(order(s, replace(expedited=True)), Plan)
    s.database.credit_card_accounts.data[ACCT]["card_type"] = "Bronze Rewards Card"
    assert isinstance(order(s, replace(expedited=False)), Plan)


def test_a_pending_order_the_limit_a_closed_account_an_untiered_card_and_other_peoples_cards_are_handled():
    s = shared()
    s.database.credit_card_orders.data["o1"] = {"order_id": "o1", "credit_card_account_id": ACCT, "user_id": ME, "status": "SHIPPED", "order_date": "11/10/2025"}
    assert isinstance(order(s, replace()), Reply), "already on its way"
    s = shared()
    for i in range(3):  # a mid tier card: three in 60 days is the limit
        s.database.credit_card_orders.data[f"o{i}"] = {"order_id": f"o{i}", "credit_card_account_id": ACCT, "user_id": ME, "status": "DELIVERED", "order_date": "10/20/2025"}
    assert isinstance(order(s, replace()), Reply)
    s = shared()
    s.database.credit_card_accounts.data[ACCT]["status"] = "CLOSED"
    assert isinstance(order(s, replace()), Reply)
    s = shared()
    s.database.credit_card_accounts.data[ACCT]["card_type"] = "Mystery Card"
    assert isinstance(order(s, replace()), HandOff)
    s = shared()
    assert isinstance(order(s, replace(credit_card_account_id="cc_76ad9cc60e_gold")), Refuse)
    assert isinstance(order(s, replace(credit_card_account_id="cc_made_up")), Refuse)
    assert isinstance(order(s, replace(), replace()), Refuse)
    assert isinstance(order(s), Refuse)


def test_a_missing_address_is_asked_for():
    s = shared()
    s.database.users.data[ME]["address"] = ""
    asked = order(s, replace())
    assert isinstance(asked, NeedInfo) and asked.questions[0].slot == "shipping_address"


def test_provisional_credit_function_is_pure():
    account = {"date_of_account_open": "05/20/2024", "card_type": "Silver Rewards Card"}
    facts = dispute(reason="unauthorized_fraudulent_charge", contacted_merchant=False)
    args = (account, facts, 100.0, date(2025, 10, 1), TODAY)
    assert provisional_credit(*args, 0) and provisional_credit(*args, 2) and not provisional_credit(*args, 3)
