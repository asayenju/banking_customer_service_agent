"""The bank-account write tools' rules: each validator accepts a good call and refuses each bad one, on
synthetic data, and accepts every write of the 19 reference trajectories. Another customer's account
reads like one that does not exist."""

import json
from copy import deepcopy
from datetime import date

import pytest

from agent.guard import allowed_writes
from agent.specialist import ValidationContext
from agent.specialists import accounts_rules as rules
from agent.specialists.accounts import ACCOUNTS_SPECIALIST, READBACKS, VALIDATORS
from banking_tools import BankingSession, Fidelity
from banking_tools.catalog import get_registry
from orchestrator.types import IntentKind as K

SAM = "125"  # checking 03 (Green Account, 09/02/2023, $3,400) and savings 04 (Silver Account, $900)
MIA = "123"  # checking 01 (Blue Account, 11/08/2025, $0) and savings 02 (Bronze Account, 11/08/2025, $129.02)
CARD_OWNER = "lj82d4f1a9"  # checking chk_lj82d4f1a9 ($127,450), a Silver card with $842.15 owed
CARD = "cc_lj82d4f1a9_silver"
CARD_CHECKING = "chk_lj82d4f1a9"
TODAY = date(2025, 11, 14)


def world() -> BankingSession:
    return BankingSession(fidelity=Fidelity.TAU2)


def v(shared: BankingSession, user: str = SAM) -> ValidationContext:
    return ValidationContext(shared.database, user, TODAY)


def put(shared, account_id, user, cls, level, opened, balance, status="OPEN"):
    shared.database.accounts.data[account_id] = {
        "account_id": account_id, "user_id": user, "class": cls, "level": level,
        "date_opened": opened, "status": status, "current_holdings": balance,
    }


def reject(validator, shared, args, part, user=SAM):
    reason = validator(v(shared, user), args)
    assert reason is not None and part in reason, reason


def accept(validator, shared, args, user=SAM):
    assert validator(v(shared, user), args) is None


def opening(user=SAM, type_="savings", cls="Silver Plus Account"):
    return {"user_id": user, "account_type": type_, "account_class": cls}


# ---------------------------------------------------------------- every write has a validator

def test_every_write_tool_the_guard_lets_the_accounts_kinds_use_has_a_validator():
    reachable = set().union(*(allowed_writes(k) for k in ACCOUNTS_SPECIALIST.kinds))
    assert reachable == set(VALIDATORS), "an accounts write without a validator is a hole"
    assert all(get_registry()[t].kind.value == "write" for t in reachable)
    assert set(READBACKS) <= set(VALIDATORS)


# ---------------------------------------------------------------- open: names, owner, hash

def test_open_accepts_a_good_call_and_refuses_each_bad_one():
    shared = world()
    accept(rules.open_account, shared, opening())
    reject(rules.open_account, shared, opening(user=MIA), "verified")
    reject(rules.open_account, shared, opening(type_="brokerage"), "not a type")
    reject(rules.open_account, shared, opening(type_="checking", cls="Totally Made Up"), "exact name")
    for near in ("silver plus account", "Silver Plus", "Silver Plus Account "):
        reject(rules.open_account, shared, opening(cls=near), "exact name")
    reject(rules.open_account, shared, opening(cls="Navy Blue"), "exact name")  # a business name is not a savings name


def test_business_checking_takes_the_bare_name_and_business_savings_the_full_one():
    shared = world()
    put(shared, "p1", SAM, "checking", "Green Account", "01/01/2024", "900")
    accept(rules.open_account, shared, opening(type_="business_checking", cls="Sky Blue"))
    reject(rules.open_account, shared, opening(type_="business_checking", cls="Sky Blue Account"), "exact name")


def test_opening_a_name_that_exists_or_existed_is_refused_instead_of_the_banks_silent_no_op():
    shared = world()
    args = opening(type_="checking", cls="Purple Account")
    accept(rules.open_account, shared, args)
    assert shared.call("open_bank_account_4821", args).ok
    reject(rules.open_account, shared, args, "second one with the same name")
    row = next(r for r in shared.database.accounts.data.values() if r.get("account_class") == "Purple Account")
    row["status"] = "CLOSED"
    reject(rules.open_account, shared, args, "second one with the same name")


def test_the_open_readback_catches_an_account_that_is_not_there():
    shared = world()
    args = opening(type_="checking", cls="Purple Account")
    assert rules.opened_readback(v(shared), args) == "the account is not there"
    shared.call("open_bank_account_4821", args)
    assert rules.opened_readback(v(shared), args) is None
    assert rules.opened_readback(v(shared, MIA), args) == "the account is not there"


# ---------------------------------------------------------------- open: personal checking

def test_personal_checking_needs_age_18_and_fewer_than_four():
    shared = world()
    args = opening(type_="checking", cls="Purple Account")
    accept(rules.open_account, shared, args)
    shared.database.users.data[SAM]["date_of_birth"] = "11/15/2007"  # 17 until tomorrow
    reject(rules.open_account, shared, args, "18 or older")
    shared.database.users.data[SAM]["date_of_birth"] = "11/14/2007"  # 18 today
    accept(rules.open_account, shared, args)
    del shared.database.users.data[SAM]["date_of_birth"]
    reject(rules.open_account, shared, args, "date of birth is not on file")
    shared.database.users.data[SAM]["date_of_birth"] = "01/01/1990"
    for n in range(3):
        put(shared, f"x{n}", SAM, "checking", "Blue Account", "01/01/2024", "0")
    reject(rules.open_account, shared, args, "most personal checking accounts allowed (4)")
    shared.database.accounts.data["x0"]["status"] = "CLOSED"
    accept(rules.open_account, shared, args)  # a closed one no longer counts


# ---------------------------------------------------------------- open: personal savings

def test_savings_needs_a_checking_account_open_fourteen_days():
    shared = world()
    accept(rules.open_account, shared, opening())
    reject(rules.open_account, shared, opening(user=MIA, cls="Silver Account"), "verified", user=SAM)
    reject(rules.open_account, shared, opening(user=MIA, cls="Silver Account"), "open at least 14 days; your checking account has been open 6 days", user=MIA)


def test_the_fourteenth_day_counts_and_the_thirteenth_does_not():
    shared = world()
    put(shared, "n13", "newbie", "checking", "Blue Account", "11/01/2025", "10")  # 13 days
    reject(rules.open_account, shared, opening("newbie", cls="Silver Account"), "open 13 days", user="newbie")
    shared.database.accounts.data["n13"]["date_opened"] = "10/31/2025"  # 14 days
    accept(rules.open_account, shared, opening("newbie", cls="Silver Account"), user="newbie")


def test_savings_needs_a_checking_account_at_all_and_an_open_one():
    shared = world()
    reject(rules.open_account, shared, opening("nobody", cls="Silver Account"), "needs an open personal checking", user="nobody")
    shared.database.accounts.data["03"]["status"] = "CLOSED"
    reject(rules.open_account, shared, opening(cls="Silver Plus Account"), "needs an open personal checking")


def test_savings_is_capped_at_five_and_blocked_by_a_negative_balance():
    shared = world()
    for n in range(4):
        put(shared, f"s{n}", SAM, "saving", "Bronze Account", "01/01/2024", "5")
    reject(rules.open_account, shared, opening(), "most savings accounts allowed (5)")
    shared.database.accounts.data["s0"]["status"] = "CLOSED"
    accept(rules.open_account, shared, opening())
    shared.database.accounts.data["s1"]["current_holdings"] = "-3.00"
    reject(rules.open_account, shared, opening(), "negative balance")


# ---------------------------------------------------------------- open: business

def test_business_checking_needs_five_hundred_dollars_an_open_checking_and_no_closed_account():
    shared = world()
    args = opening(type_="business_checking", cls="Navy Blue")
    accept(rules.open_account, shared, args)
    shared.database.accounts.data["03"]["current_holdings"] = "499.99"
    reject(rules.open_account, shared, args, "$500.00 in your personal checking")
    shared.database.accounts.data["03"]["current_holdings"] = "500.00"
    accept(rules.open_account, shared, args)
    shared.database.accounts.data["04"]["status"] = "CLOSED"
    reject(rules.open_account, shared, args, "any of your accounts has been closed")
    reject(rules.open_account, shared, opening("nobody", type_="business_checking", cls="Navy Blue"), "needs an open personal checking", user="nobody")


def test_business_checking_is_capped_at_six():
    shared = world()
    for n in range(6):
        put(shared, f"b{n}", SAM, "business_checking", "Navy Blue", "01/01/2024", "100")
    reject(rules.open_account, shared, opening(type_="business_checking", cls="Sky Blue"), "most business checking accounts allowed (6)")


def test_business_savings_needs_a_thirty_day_business_checking_account_with_twenty_five_hundred():
    shared = world()
    args = opening(type_="business_savings", cls="Gold Saver Account")
    reject(rules.open_account, shared, args, "needs an open business checking account")
    put(shared, "bc", SAM, "business_checking", "Navy Blue", "10/16/2025", "3200")  # 29 days
    reject(rules.open_account, shared, args, "open at least 30 days with $2,500.00")
    shared.database.accounts.data["bc"]["date_opened"] = "10/15/2025"  # 30 days
    accept(rules.open_account, shared, args)
    shared.database.accounts.data["bc"]["current_holdings"] = "2499.99"
    reject(rules.open_account, shared, args, "$2,500.00 in it")
    shared.database.accounts.data["bc"]["current_holdings"] = "2500"
    accept(rules.open_account, shared, args)
    shared.database.accounts.data["04"]["current_holdings"] = "-1"
    reject(rules.open_account, shared, args, "negative balance")


def test_business_savings_is_capped_at_four():
    shared = world()
    put(shared, "bc", SAM, "business_checking", "Navy Blue", "01/01/2025", "9000")
    for n in range(4):
        put(shared, f"bs{n}", SAM, "business_savings", "Bronze Saver Account", "01/01/2025", "1")
    reject(rules.open_account, shared, opening(type_="business_savings", cls="Gold Saver Account"), "most business savings accounts allowed (4)")


# ---------------------------------------------------------------- close

def test_close_needs_a_zero_balance_outside_the_fee_window():
    shared = world()
    args = {"account_id": "03", "reason": "Customer requested closure"}
    reject(rules.close_account, shared, args, "has $3,400.00; move it out")
    shared.database.accounts.data["03"]["current_holdings"] = "0.00"
    accept(rules.close_account, shared, args)


def test_close_of_someone_elses_or_an_unknown_or_a_closed_account():
    shared = world()
    for account_id in ("01", "nope"):
        reject(rules.close_account, shared, {"account_id": account_id}, "that account does not exist")
    shared.database.accounts.data["03"]["status"] = "CLOSED"
    reject(rules.close_account, shared, {"account_id": "03"}, "already closed")


def test_inside_the_fee_window_the_balance_must_be_exactly_the_fee():
    shared = world()  # Mia's Blue Account: six days old, $25 inside 60 days
    args = {"account_id": "01"}
    reject(rules.close_account, shared, args, "needs $25.00 in it to close; it has $0.00", MIA)
    shared.database.accounts.data["01"]["current_holdings"] = "24.99"
    reject(rules.close_account, shared, args, "needs $25.00", MIA)
    shared.database.accounts.data["01"]["current_holdings"] = "25"
    accept(rules.close_account, shared, args, MIA)
    shared.database.accounts.data["01"]["current_holdings"] = "100"
    reject(rules.close_account, shared, args, "move $75.00 out first; only the $25.00 early-closure fee stays", MIA)


def test_the_window_ends_on_its_last_day():
    shared = world()
    shared.database.accounts.data["01"]["current_holdings"] = "0"
    shared.database.accounts.data["01"]["date_opened"] = "09/16/2025"  # 59 days: still inside 60
    reject(rules.close_account, shared, {"account_id": "01"}, "inside its early-closure window", MIA)
    shared.database.accounts.data["01"]["date_opened"] = "09/15/2025"  # 60 days: outside
    accept(rules.close_account, shared, {"account_id": "01"}, MIA)


def test_the_savings_fee_the_bank_never_applies_is_applied_here():
    shared = world()
    shared.database.accounts.data["02"]["current_holdings"] = "0"  # Bronze, six days old: $20
    assert shared.call("close_bank_account_7392", {"account_id": "02", "reason": "x"}).ok, "the bank lets it through"
    shared2 = world()
    shared2.database.accounts.data["02"]["current_holdings"] = "0"
    reject(rules.close_account, shared2, {"account_id": "02"}, "needs $20.00 in it to close", MIA)


def test_a_waiver_pending_transactions_unknown_tiers_and_approvals_are_not_closed_by_the_agent():
    shared = world()
    shared.database.accounts.data["03"]["current_holdings"] = "0"
    reject(rules.close_account, shared, {"account_id": "03", "waive_early_closure_fee": True}, "cannot be waived")
    shared.database.bank_account_transaction_history.data["t"] = {"transaction_id": "t", "account_id": "03", "status": "pending"}
    reject(rules.close_account, shared, {"account_id": "03"}, "has not posted yet")
    del shared.database.bank_account_transaction_history.data["t"]
    put(shared, "pp", SAM, "checking", "Purple Account", "01/01/2024", "0")
    reject(rules.close_account, shared, {"account_id": "pp"}, "no closing rules on file for a Purple Account")
    put(shared, "pl", SAM, "saving", "Platinum Account", "01/01/2020", "0")
    reject(rules.close_account, shared, {"account_id": "pl"}, "needs manager approval")
    put(shared, "bc", SAM, "business_checking", "Cobalt Blue", "01/01/2020", "0")
    reject(rules.close_account, shared, {"account_id": "bc"}, "needs supervisor approval")
    shared.database.accounts.data["03"]["date_opened"] = ""
    reject(rules.close_account, shared, {"account_id": "03"}, "was opened is not on file")


def test_business_accounts_close_in_the_order_their_link_needs():
    shared = world()
    put(shared, "bc", SAM, "business_checking", "Navy Blue", "01/01/2024", "0")
    put(shared, "bs", SAM, "business_savings", "Bronze Saver Account", "01/01/2024", "0")
    reject(rules.close_account, shared, {"account_id": "bc"}, "close that first")
    accept(rules.close_account, shared, {"account_id": "bs"})
    shared.database.accounts.data["bc"]["status"] = "CLOSED"
    reject(rules.close_account, shared, {"account_id": "bs"}, "needs an open business checking account")


def test_the_close_readback():
    shared = world()
    shared.database.accounts.data["03"]["current_holdings"] = "0"
    assert rules.closed_readback(v(shared), {"account_id": "03"}) == "the account is not closed"
    assert shared.call("close_bank_account_7392", {"account_id": "03", "reason": "x"}).ok
    assert rules.closed_readback(v(shared), {"account_id": "03"}) is None


def test_a_plan_that_moves_the_money_then_closes_is_judged_on_the_state_each_step_leaves():
    from agent.specialists.accounts_rules import close_account, transfer

    shared = world()
    transfer_args = {"source_account_id": "03", "destination_account_id": "04", "amount": 3400.0}
    reject(close_account, shared, {"account_id": "03"}, "move it out")
    assert transfer(v(shared), transfer_args) is None
    assert shared.call("transfer_funds_between_bank_accounts_7291", transfer_args).ok
    accept(close_account, shared, {"account_id": "03"})


# ---------------------------------------------------------------- transfer

def moving(src="03", dst="04", amount=100.0):
    return {"source_account_id": src, "destination_account_id": dst, "amount": amount}


def test_a_transfer_between_the_customers_own_open_accounts_with_funds_is_accepted():
    shared = world()
    accept(rules.transfer, shared, moving())
    accept(rules.transfer, shared, moving(amount=3400.0))  # exactly the balance
    accept(rules.transfer, shared, moving(amount="$100.00"))


def test_a_transfer_to_or_from_someone_elses_or_an_unknown_account_is_refused_alike():
    shared = world()
    reject(rules.transfer, shared, moving(dst="01"), "the account the money goes to does not exist")
    reject(rules.transfer, shared, moving(dst="nope"), "the account the money goes to does not exist")
    reject(rules.transfer, shared, moving(src="01"), "the account the money comes from does not exist")
    reject(rules.transfer, shared, moving(src="nope"), "the account the money comes from does not exist")


def test_a_transfer_needs_two_different_open_accounts_a_positive_amount_and_funds():
    shared = world()
    reject(rules.transfer, shared, moving(dst="03"), "two different accounts")
    reject(rules.transfer, shared, moving(amount=3400.01), "not enough")
    for bad in (0, -5, "ten", None, True):
        reject(rules.transfer, shared, moving(amount=bad), "more than zero")
    shared.database.accounts.data["04"]["status"] = "CLOSED"
    reject(rules.transfer, shared, moving(), "the account the money goes to is not open")
    shared.database.accounts.data["04"]["status"] = "OPEN"
    shared.database.accounts.data["03"]["status"] = "CLOSED"
    reject(rules.transfer, shared, moving(), "is not open")


def test_the_transfer_readback_checks_what_must_still_hold():
    shared = world()
    assert rules.transfer_readback(v(shared), moving()) is None
    shared.database.accounts.data["03"]["current_holdings"] = "-1"
    assert rules.transfer_readback(v(shared), moving()) == "the account the money came from went below zero"
    assert rules.transfer_readback(v(shared), moving(dst="01")) == "an account involved is no longer there or open"


# ---------------------------------------------------------------- paying a card from checking

def paying(amount=100.0, checking=CARD_CHECKING, card=CARD, user=CARD_OWNER):
    return {"user_id": user, "checking_account_id": checking, "credit_card_account_id": card, "amount": amount}


def test_a_card_payment_from_an_open_checking_account_within_both_balances_is_accepted():
    shared = world()
    accept(rules.pay_card, shared, paying(), CARD_OWNER)
    accept(rules.pay_card, shared, paying(842.15), CARD_OWNER)


def test_a_card_payment_is_refused_for_each_thing_that_can_be_wrong():
    shared = world()
    reject(rules.pay_card, shared, paying(842.16), "less than", CARD_OWNER)
    for bad in (0, -1, "x", None):
        reject(rules.pay_card, shared, paying(bad), "more than zero", CARD_OWNER)
    reject(rules.pay_card, shared, paying(user=MIA), "verified", CARD_OWNER)
    reject(rules.pay_card, shared, paying(checking="03"), "that checking account does not exist", CARD_OWNER)
    reject(rules.pay_card, shared, paying(card="cc_76ad9cc60e_gold"), "that credit card account does not exist", CARD_OWNER)
    shared.database.accounts.data[CARD_CHECKING]["current_holdings"] = "50.00"
    reject(rules.pay_card, shared, paying(), "which is not enough", CARD_OWNER)
    shared.database.accounts.data[CARD_CHECKING]["current_holdings"] = "127450.00"
    shared.database.accounts.data[CARD_CHECKING]["status"] = "CLOSED"
    reject(rules.pay_card, shared, paying(), "open personal checking", CARD_OWNER)


def test_a_savings_account_and_a_business_checking_account_cannot_pay_a_card():
    shared = world()
    put(shared, "sv", CARD_OWNER, "saving", "Silver Account", "01/01/2024", "5000")
    put(shared, "bk", CARD_OWNER, "business_checking", "Navy Blue", "01/01/2024", "5000")
    for account in ("sv", "bk"):
        reject(rules.pay_card, shared, paying(checking=account), "open personal checking", CARD_OWNER)


def test_a_checking_account_opened_in_this_visit_cannot_pay_because_the_bank_cannot_see_it():
    shared = world()
    shared.call("open_bank_account_4821", {"user_id": CARD_OWNER, "account_type": "checking", "account_class": "Purple Account"})
    new = next(r["account_id"] for r in shared.database.accounts.data.values() if r.get("account_class") == "Purple Account")
    shared.database.accounts.data[new]["current_holdings"] = "500"
    reject(rules.pay_card, shared, paying(checking=new), "too new for the bank's payment system", CARD_OWNER)
    result = shared.call("pay_credit_card_from_checking_9182", paying(checking=new))
    assert not result.ok, "the bank refuses it too"


def test_a_closed_card_account_cannot_be_paid():
    shared = world()
    shared.database.credit_card_accounts.data[CARD]["status"] = "CLOSED"
    reject(rules.pay_card, shared, paying(), "is closed", CARD_OWNER)


# ---------------------------------------------------------------- the reference trajectories

ACCOUNT_TASKS = [f"task_{n:03d}" for n in [*range(55, 72), 75, 76]]


@pytest.mark.parametrize("task_id", ACCOUNT_TASKS)
def test_every_write_in_the_reference_answer_passes_the_validators_in_order(task_id):
    """Replays a task's reference writes, judging each against the state the earlier ones left. A write the
    documents forbid would show here as a departure to record in evals/policy_departures.json."""
    from tau2.domains.banking_knowledge.tools import KnowledgeUserTools

    from agent.tier1 import gold_steps
    from banking_tools.manifest import load_tasks

    task = next(t for t in load_tasks() if str(t["id"]) == task_id)
    session = BankingSession.from_task(task, fidelity=Fidelity.TAU2)
    customer = KnowledgeUserTools(session.database)
    steps = gold_steps(task)
    user_id = next(str(s.args["user_id"]) for s in steps if s.kind == "verify")
    checked = 0
    for step in steps:
        if step.kind == "user":
            getattr(customer, step.name)(**step.args)
        elif step.kind == "write" and step.name in VALIDATORS:
            args = {**step.args, "user_id": user_id} if "user_id" in step.args else dict(step.args)
            problem = VALIDATORS[step.name](ValidationContext(session.database, user_id, TODAY), args)
            assert problem is None, f"{task_id} {step.name}: {problem}"
            assert session.call(step.name, step.args).ok
            readback = READBACKS.get(step.name)
            assert readback is None or readback(ValidationContext(session.database, user_id, TODAY), args) is None
            checked += 1
        elif step.kind == "other":  # handing the customer a tool (give_discoverable_user_tool), as tier-1 does
            getattr(session.toolkit, step.name)(**step.args)
        elif step.kind in ("write", "read"):
            session.call(step.name, step.args)
    assert checked > 0, "the task has no account write to check"
