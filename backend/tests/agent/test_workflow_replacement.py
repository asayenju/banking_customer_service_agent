"""The replacement workflow, given the customer's choices, builds the reference closures and orders."""

from datetime import date

import pytest

from agent.tier1 import gold_steps
from agent.workflows.base import HandOff, NeedInfo, Plan, Refuse, Reply
from agent.workflows.base import WorkflowContext
from agent.workflows.debit_replacement import DebitReplacement, ReplacementFacts
from agent.workflows.gold_facts import facts_from_gold
from banking_tools.manifest import load_tasks
from banking_tools.session import _base_db
from banking_tools.task_state import apply_initial_state

TODAY = date(2025, 11, 14)


def tasks_with_orders():
    return [(t, gold_steps(t)) for t in load_tasks() if any(s.name == "order_debit_card_5739" for s in gold_steps(t))]


@pytest.mark.parametrize("task,steps", tasks_with_orders(), ids=lambda v: v["id"][-3:] if isinstance(v, dict) else "")
def test_workflow_builds_the_reference_orders_and_closures(task, steps):
    db = apply_initial_state(_base_db().model_copy(deep=True), task)
    writes = [s for s in steps if s.kind == "write"]
    facts = ReplacementFacts.model_validate(facts_from_gold("debit_replacement", writes, db))
    user = next(s.args["user_id"] for s in writes if s.name == "order_debit_card_5739")
    decision = DebitReplacement().decide(WorkflowContext(db, user, TODAY), facts)
    assert isinstance(decision, Plan), decision
    made = [(s.tool, dict(s.args)) for s in decision.steps]
    orders = [(t, a) for t, a in made if t == "order_debit_card_5739"]
    gold_orders = [(s.name, {k: v for k, v in s.args.items() if k != "user_id"}) for s in writes if s.name == "order_debit_card_5739"]
    assert orders == gold_orders
    closed = {(t, a["card_id"], a["reason"]) for t, a in made if t == "close_debit_card_4721"}
    wanted = {
        ("close_debit_card_4721", s.args["card_id"], s.args["reason"])
        for s in writes
        if s.name == "close_debit_card_4721" and db.debit_cards.data[s.args["card_id"]]["status"] != "CLOSED"
    }
    assert closed == wanted


def a_light_blue_card():
    task = next(t for t in load_tasks() if t["id"].endswith("091"))
    db = apply_initial_state(_base_db().model_copy(deep=True), task)
    card = next(
        c for c in db.debit_cards.data.values() if db.accounts.data[c["account_id"]]["level"] == "Light Blue Account"
    )
    return db, card


def test_a_speed_the_tier_does_not_offer_is_refused_with_the_options():
    db, card = a_light_blue_card()
    facts = ReplacementFacts.model_validate(
        {
            "closures": [],
            "new_cards": [
                {"account_id": card["account_id"], "delivery_option": "RUSH", "card_design": "CLASSIC",
                 "shipping_address": None, "accepts_excess_fee": "not_asked"}
            ],
        }
    )
    decision = DebitReplacement().decide(WorkflowContext(db, card["user_id"], TODAY), facts)
    assert isinstance(decision, Refuse) and "STANDARD" in decision.reason


def test_another_customers_card_is_refused():
    db, card = a_light_blue_card()
    facts = ReplacementFacts.model_validate(
        {"closures": [{"card_id": card["card_id"], "reason": "lost", "accepts_refund_to_checking": False}], "new_cards": []}
    )
    decision = DebitReplacement().decide(WorkflowContext(db, "someone_else", TODAY), facts)
    assert isinstance(decision, Refuse)


def test_an_account_level_with_no_known_tier_is_handed_to_a_person():
    db, card = a_light_blue_card()
    db.accounts.data[card["account_id"]]["level"] = "Mystery Account"
    facts = ReplacementFacts.model_validate(
        {
            "closures": [],
            "new_cards": [
                {"account_id": card["account_id"], "delivery_option": "STANDARD", "card_design": "CLASSIC",
                 "shipping_address": None, "accepts_excess_fee": "not_asked"}
            ],
        }
    )
    assert isinstance(DebitReplacement().decide(WorkflowContext(db, card["user_id"], TODAY), facts), HandOff)


def test_fastest_free_is_worked_out_per_tier():
    from agent.workflows.debit_replacement import DELIVERY_FEE, SPEED

    for tier, fees in DELIVERY_FEE.items():
        fastest = [o for o in SPEED if fees.get(o) == 0.0][-1]
        assert {"Entry": "STANDARD", "Mid": "STANDARD", "Premium": "EXPEDITED", "Elite": "RUSH"}[tier] == fastest

    task = next(t for t in load_tasks() if t["id"].endswith("090"))
    db = apply_initial_state(_base_db().model_copy(deep=True), task)
    order = next(s for s in gold_steps(task) if s.name == "order_debit_card_5739")
    facts = ReplacementFacts.model_validate(
        {
            "closures": [],
            "new_cards": [
                {"account_id": order.args["account_id"], "delivery_option": "FASTEST_FREE", "card_design": "CLASSIC",
                 "shipping_address": None, "accepts_excess_fee": "not_asked"}
            ],
        }
    )
    plan = DebitReplacement().decide(WorkflowContext(db, order.args["user_id"], TODAY), facts)
    assert isinstance(plan, Plan) and plan.steps[0].args["delivery_option"] == "EXPEDITED"
    assert plan.steps[0].args["delivery_fee"] == 0.0


def test_best_free_design_is_worked_out_per_tier():
    from agent.workflows.debit_replacement import DESIGN_FEE, LOOKS

    best = {t: [d for d in LOOKS if fees[d] == 0.0][-1] for t, fees in DESIGN_FEE.items()}
    assert best == {"Entry": "CLASSIC", "Mid": "CLASSIC", "Premium": "PREMIUM", "Elite": "CUSTOM"}

    task = next(t for t in load_tasks() if t["id"].endswith("087"))
    db = apply_initial_state(_base_db().model_copy(deep=True), task)
    order = next(s for s in gold_steps(task) if s.name == "order_debit_card_5739")
    facts = ReplacementFacts.model_validate(
        {"closures": [], "new_cards": [{"account_id": order.args["account_id"], "delivery_option": "FASTEST_FREE",
                                        "card_design": "BEST_FREE", "shipping_address": None, "accepts_excess_fee": "not_asked"}]}
    )
    plan = DebitReplacement().decide(WorkflowContext(db, order.args["user_id"], TODAY), facts)
    assert isinstance(plan, Plan)
    assert {k: plan.steps[0].args[k] for k in ("delivery_option", "card_design", "delivery_fee", "design_fee")} == {
        "delivery_option": order.args["delivery_option"], "card_design": order.args["card_design"],
        "delivery_fee": 0.0, "design_fee": 0.0,
    }


# ---------------------------------------------------------------- pending transactions (tasks 078 and 079)

def _task_world(task: str):
    from banking_tools import BankingSession
    from banking_tools.manifest import load_tasks

    session = BankingSession.from_task({t["id"]: t for t in load_tasks()}[task])
    return session.database


def _close(db, user, card, *, accepts, new_cards=()):
    facts = ReplacementFacts.model_validate(
        {
            "closures": [{"card_id": card, "reason": "lost", "accepts_refund_to_checking": accepts}],
            "new_cards": list(new_cards),
        }
    )
    return DebitReplacement().decide(WorkflowContext(db, user, TODAY), facts)


def test_a_card_with_pending_purchases_is_frozen_not_closed_by_the_debit_workflow():
    """Task 078: Light Green has two pending purchases. Any debit request holds it, not only a lost wallet."""
    db = _task_world("task_078")
    decision = _close(db, "mc78a5b9d2", "dbc_mc78a5b9d2_lg", accepts=True)
    assert isinstance(decision, Plan)
    assert [s.tool for s in decision.steps] == ["freeze_debit_card_3892"]
    assert "pending" in decision.steps[0].note


def test_a_held_card_gets_no_new_card_ordered():
    db = _task_world("task_078")
    order = {
        "account_id": "chk_mc78a5b9d2_3", "delivery_option": "STANDARD", "card_design": "CLASSIC",
        "shipping_address": None, "accepts_excess_fee": "not_asked",
    }
    decision = _close(db, "mc78a5b9d2", "dbc_mc78a5b9d2_lg", accepts=True, new_cards=[order])
    assert isinstance(decision, Plan) and all(s.tool != "order_debit_card_5739" for s in decision.steps)


def test_a_pending_refund_is_asked_about_and_the_accept_unlocks_the_close():
    """Task 079: Green has a pending refund; the customer must agree it goes to checking."""
    db = _task_world("task_079")
    card = next(
        c for c in db.debit_cards.data.values() if c["account_id"] == "chk_cr89a2b3c4_3" and c["status"] != "CLOSED"
    )["card_id"]
    asked = _close(db, "cr89a2b3c4", card, accepts=False)
    assert isinstance(asked, NeedInfo) and asked.questions[0].slot == "accepts_refund_to_checking"
    agreed = _close(db, "cr89a2b3c4", card, accepts=True)
    assert isinstance(agreed, Plan) and "close_debit_card_4721" in [s.tool for s in agreed.steps]


# ---------------------------------------------------------------- the excess fee answer and the Entry wait (078, 079)

LB_ACCOUNT = "chk_cr89a2b3c4_2"
LB_CARD = "dbc_cr89a2b3c4_lb"


def _light_blue(answer, *, closing=True):
    db = _task_world("task_079")
    order = {
        "account_id": LB_ACCOUNT, "delivery_option": "STANDARD", "card_design": "CLASSIC",
        "shipping_address": None, "accepts_excess_fee": answer,
    }
    closures = [{"card_id": LB_CARD, "reason": "stolen", "accepts_refund_to_checking": True}] if closing else []
    facts = ReplacementFacts.model_validate({"closures": closures, "new_cards": [order]})
    return DebitReplacement().decide(WorkflowContext(db, "cr89a2b3c4", TODAY), facts)


def test_a_declined_excess_fee_closes_the_card_and_orders_nothing():
    decision = _light_blue("no")
    assert isinstance(decision, Plan) and [s.tool for s in decision.steps][-1] == "close_debit_card_4721"
    assert "excess replacement fee" in decision.steps[0].note


def test_an_unanswered_excess_fee_is_asked_not_assumed():
    decision = _light_blue("not_asked", closing=False)
    assert isinstance(decision, NeedInfo) and decision.questions[0].slot == "accepts_excess_fee"


def test_an_entry_card_past_its_limit_closed_in_the_same_plan_waits_48_hours():
    decision = _light_blue("yes")
    assert isinstance(decision, Plan) and all(s.tool != "order_debit_card_5739" for s in decision.steps)
    assert "48 hours" in decision.steps[0].note


def test_an_entry_card_past_its_limit_whose_card_is_already_closed_can_be_ordered_with_the_fee():
    decision = _light_blue("yes", closing=False)
    assert isinstance(decision, Plan) and decision.steps[-1].tool == "order_debit_card_5739"


# ---------------------------------------------------------------- doc 023 and 025 rules the bank's tools never check

def test_a_card_under_14_days_old_is_not_closed_for_a_non_urgent_reason():
    db = _task_world("task_078")
    card = db.debit_cards.data["dbc_mc78a5b9d2_green"]
    card["issue_date"] = TODAY.strftime("%m/%d/%Y")
    facts = ReplacementFacts.model_validate(
        {"closures": [{"card_id": card["card_id"], "reason": "no_longer_needed", "accepts_refund_to_checking": False}], "new_cards": []}
    )
    decision = DebitReplacement().decide(WorkflowContext(db, "mc78a5b9d2", TODAY), facts)
    assert isinstance(decision, Reply) and "under 14 days old" in decision.text
    urgent = ReplacementFacts.model_validate(
        {"closures": [{"card_id": card["card_id"], "reason": "lost", "accepts_refund_to_checking": False}], "new_cards": []}
    )
    assert isinstance(DebitReplacement().decide(WorkflowContext(db, "mc78a5b9d2", TODAY), urgent), Plan), "lost bypasses the age"


@pytest.mark.parametrize(
    ("change", "words"),
    [
        ({"account": {"date_opened": "TODAY"}}, "business days"),
        ({"account": {"current_holdings": "10.00"}}, "$25"),
        ({"user": {"date_of_birth": "01/01/2015"}}, "18 years"),
        ({"user": {"address": "10 Rue de Rivoli, Paris, France"}}, "US address"),
    ],
)
def test_an_order_the_bank_would_not_allow_is_left_out_with_the_reason(change, words):
    db = _task_world("task_078")
    account = db.accounts.data["chk_mc78a5b9d2_2"]
    for key, value in change.get("account", {}).items():
        account[key] = TODAY.strftime("%m/%d/%Y") if value == "TODAY" else value
    db.users.data["mc78a5b9d2"].update(change.get("user", {}))
    closing = {"card_id": "dbc_mc78a5b9d2_green", "reason": "lost", "accepts_refund_to_checking": False}
    order = {
        "account_id": "chk_mc78a5b9d2_2", "delivery_option": "STANDARD", "card_design": "CLASSIC",
        "shipping_address": None, "accepts_excess_fee": "not_asked",
    }
    facts = ReplacementFacts.model_validate({"closures": [closing], "new_cards": [order]})
    decision = DebitReplacement().decide(WorkflowContext(db, "mc78a5b9d2", TODAY), facts)
    assert isinstance(decision, Plan) and all(s.tool != "order_debit_card_5739" for s in decision.steps)
    assert words in decision.steps[0].note
