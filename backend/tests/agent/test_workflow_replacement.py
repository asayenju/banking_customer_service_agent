"""The replacement workflow, given the customer's choices, builds the reference closures and orders."""

from datetime import date

import pytest

from agent.tier1 import gold_steps
from agent.workflows.base import HandOff, Plan, Refuse
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
                 "shipping_address": None, "accepts_excess_fee": False}
            ],
        }
    )
    decision = DebitReplacement().decide(WorkflowContext(db, card["user_id"], TODAY), facts)
    assert isinstance(decision, Refuse) and "STANDARD" in decision.reason


def test_another_customers_card_is_refused():
    db, card = a_light_blue_card()
    facts = ReplacementFacts.model_validate(
        {"closures": [{"card_id": card["card_id"], "reason": "lost"}], "new_cards": []}
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
                 "shipping_address": None, "accepts_excess_fee": False}
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
                 "shipping_address": None, "accepts_excess_fee": False}
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
                                        "card_design": "BEST_FREE", "shipping_address": None, "accepts_excess_fee": False}]}
    )
    plan = DebitReplacement().decide(WorkflowContext(db, order.args["user_id"], TODAY), facts)
    assert isinstance(plan, Plan)
    assert {k: plan.steps[0].args[k] for k in ("delivery_option", "card_design", "delivery_fee", "design_fee")} == {
        "delivery_option": order.args["delivery_option"], "card_design": order.args["card_design"],
        "delivery_fee": 0.0, "design_fee": 0.0,
    }
