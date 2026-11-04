"""The dispute workflow, given only what a customer would say, must build the reference steps."""

from datetime import date

import pytest

from agent.tier1 import gold_steps
from agent.workflows.base import Plan, WorkflowContext
from agent.workflows.debit_dispute import DebitDispute, DisputeFacts, DisputeRequest
from banking_tools.manifest import load_tasks
from banking_tools.session import _base_db
from banking_tools.task_state import apply_initial_state

SAID = (
    "transaction_id dispute_category discovery_date disputed_amount transaction_type card_in_possession "
    "pin_compromised contacted_merchant police_report_filed written_statement_provided"
).split()


# Where the reference answer goes past the written policy. The workflow follows the policy.
#   082: a recurring-charge dispute gets provisional credit (the guideline calls that optional),
#        and the card behind a card-not-present fraud dispute is left open (the card-action
#        table says close and reissue).
#   091: the second closure is the customer's separate lost-card request, not part of a dispute.
KNOWN = {
    "082": {"provisional_credit_eligible": {"recurring_charge_after_cancellation"}, "extra_closures": 1},
    "091": {"missing_closures": 1},
}


def tasks_with_disputes():
    out = []
    for t in load_tasks():
        steps = gold_steps(t)
        disputes = [s for s in steps if s.name == "file_debit_card_transaction_dispute_6281"]
        if disputes and all("customer_max_liability_amount" in s.args for s in disputes):
            out.append((t, steps, disputes))
    return out


@pytest.mark.parametrize("task,steps,disputes", tasks_with_disputes(), ids=lambda v: v["id"][-3:] if isinstance(v, dict) else "")
def test_workflow_builds_the_reference_dispute_steps(task, steps, disputes):
    db = apply_initial_state(_base_db().model_copy(deep=True), task)
    user = disputes[0].args["user_id"]
    facts = DisputeRequest(
        disputes=[
            DisputeFacts(
                transaction_id=d.args["transaction_id"],
                category=d.args["dispute_category"],
                discovery_date=d.args["discovery_date"],
                disputed_amount=d.args["disputed_amount"],
                transaction_type=d.args["transaction_type"],
                card_in_possession=d.args["card_in_possession"],
                pin_compromised=d.args["pin_compromised"],
                contacted_merchant=d.args["contacted_merchant"],
                police_report_filed=d.args["police_report_filed"],
                written_statement_provided=d.args["written_statement_provided"],
            )
            for d in disputes
        ]
    )
    decision = DebitDispute().decide(WorkflowContext(db, user, date(2025, 11, 14)), facts)
    assert isinstance(decision, Plan), decision
    made = [(s.tool, dict(s.args)) for s in decision.steps]
    gold = [
        (s.name, {k: v for k, v in s.args.items() if k != "user_id"})
        for s in steps
        if s.name in ("file_debit_card_transaction_dispute_6281", "close_debit_card_4721", "freeze_debit_card_3892")
    ]
    # Compare dispute steps exactly; card actions as a set (gold closes in its own order).
    known = KNOWN.get(task["id"][-3:], {})
    allowed = known.get("provisional_credit_eligible", set())
    mine = [m for m in made if m[0].startswith("file")]
    theirs = [g for g in gold if g[0].startswith("file")]
    assert len(mine) == len(theirs)
    for (_, m), (_, g) in zip(mine, theirs):
        differing = {k for k in m if m[k] != g[k]}
        assert differing <= ({"provisional_credit_eligible"} if g["dispute_category"] in allowed else set()), (g["dispute_category"], differing)
    acts = {(t, a["card_id"]) for t, a in made if not t.startswith("file")}
    wanted = {(t, a["card_id"]) for t, a in gold if not t.startswith("file")}
    assert len(acts - wanted) == known.get("extra_closures", 0)
    assert len(wanted - acts) == known.get("missing_closures", 0)
