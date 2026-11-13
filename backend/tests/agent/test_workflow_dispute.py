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
#   082 used to be here; the closure is now the customer's choice and the optional provisional credit follows the
#        reference for a cancelled subscription the customer took to the merchant.
#   091: the second closure is the customer's separate lost-card request, not part of a dispute.
KNOWN = {
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
                card_closure="yes"
                if d.args["card_id"] in {w.args["card_id"] for w in steps if w.name == "close_debit_card_4721"}
                else "no",
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


# ---------------------------------------------------------------- a partial category never disputes the whole charge (084)

def _dispute_084(amount, category="incorrect_amount"):
    from banking_tools import BankingSession

    task = next(t for t in load_tasks() if t["id"] == "task_084")
    db = BankingSession.from_task(task).database
    facts = DisputeRequest(
        disputes=[
            DisputeFacts(
                transaction_id="btxn_f9d4c10e3d56", category=category, discovery_date="11/09/2025",
                disputed_amount=amount, transaction_type="signature_purchase", card_in_possession=True,
                pin_compromised="no", contacted_merchant=True, police_report_filed=False,
                written_statement_provided=True, card_closure="not_asked",
            )
        ]
    )
    return DebitDispute().decide(WorkflowContext(db, "a7c2f85d91", date(2025, 11, 14)), facts)


def test_an_incorrect_amount_dispute_for_the_whole_charge_is_refused_with_what_to_do():
    from agent.workflows.base import Refuse

    decision = _dispute_084(189.23)
    assert isinstance(decision, Refuse) and "only part of the $189.23 charge" in decision.reason


def test_an_incorrect_amount_dispute_for_the_overcharge_is_planned():
    assert isinstance(_dispute_084(100.0), Plan)


# ---------------------------------------------------------------- details taken from the records, not the model

def test_a_wrong_year_on_the_discovery_date_is_settled_from_the_charge():
    from agent.workflows.debit_dispute import settle_year

    charge, today = date(2025, 10, 4), date(2025, 11, 14)
    assert settle_year(date(2025, 10, 9), charge, today) == date(2025, 10, 9), "a valid date is left alone"
    assert settle_year(date(2024, 10, 9), charge, today) == date(2025, 10, 9), "the year before the charge becomes its year"
    assert settle_year(date(2025, 1, 12), charge, today) == date(2025, 1, 12), "no year fits, so the date is refused as before"
    assert settle_year(date(2025, 1, 12), date(2024, 12, 20), today) == date(2025, 1, 12)
    assert settle_year(date(2024, 1, 12), date(2024, 12, 20), today) == date(2025, 1, 12), "the year after the charge"


def test_an_atm_transaction_type_comes_from_the_record():
    from agent.workflows.base import Plan

    task = next(t for t in load_tasks() if t["id"] == "task_082")
    from banking_tools import BankingSession

    db = BankingSession.from_task(task).database
    facts = DisputeRequest(
        disputes=[
            DisputeFacts(
                transaction_id="btxn_3bd1ba6ad800", category="atm_cash_discrepancy", discovery_date="11/09/2025",
                disputed_amount=100.0, transaction_type="pin_purchase", card_in_possession=True, pin_compromised="no",
                contacted_merchant=False, police_report_filed=False, written_statement_provided=True, card_closure="not_asked",
            )
        ]
    )
    decision = DebitDispute().decide(WorkflowContext(db, "mc47a2b9e1", date(2025, 11, 14)), facts)
    assert isinstance(decision, Plan) and decision.steps[0].args["transaction_type"] == "atm_withdrawal"


def test_a_double_charge_is_disputed_once_on_the_row_the_reference_disputes():
    """Doc 031: dispute the earliest when a charge is duplicated. The two rows of task 084 are identical."""
    from agent.workflows.base import Plan
    from banking_tools import BankingSession

    task = next(t for t in load_tasks() if t["id"] == "task_084")
    db = BankingSession.from_task(task).database

    def dup(txn):
        return DisputeFacts(
            transaction_id=txn, category="duplicate_charge", discovery_date="11/10/2025", disputed_amount=47.5,
            transaction_type="pin_purchase", card_in_possession=True, pin_compromised="no", contacted_merchant=False,
            police_report_filed=False, written_statement_provided=True, card_closure="not_asked",
        )

    for listed in ([dup("btxn_d7f2a98c1b34"), dup("btxn_e8c3b09d2c45")], [dup("btxn_d7f2a98c1b34")], [dup("btxn_e8c3b09d2c45")]):
        decision = DebitDispute().decide(WorkflowContext(db, "a7c2f85d91", date(2025, 11, 14)), DisputeRequest(disputes=listed))
        assert isinstance(decision, Plan) and [s.args["transaction_id"] for s in decision.steps] == ["btxn_e8c3b09d2c45"]


# ---------------------------------------------------------------- only the questions that matter for this kind of dispute

def _ask_about(category, **said):
    from agent.workflows.base import NeedInfo
    from banking_tools import BankingSession

    task = next(t for t in load_tasks() if t["id"] == "task_084")
    db = BankingSession.from_task(task).database
    base = dict(
        transaction_id="btxn_e8c3b09d2c45", category=category, discovery_date=None, disputed_amount=47.5,
        transaction_type="pin_purchase", card_in_possession=None, pin_compromised=None, contacted_merchant=None,
        police_report_filed=None, written_statement_provided=None, card_closure="not_asked",
    )
    decision = DebitDispute().decide(
        WorkflowContext(db, "a7c2f85d91", date(2025, 11, 14)), DisputeRequest(disputes=[DisputeFacts(**{**base, **said})])
    )
    return decision, NeedInfo


def test_a_duplicate_charge_asks_three_things_and_not_the_card_or_the_pin():
    decision, NeedInfo = _ask_about("duplicate_charge")
    assert isinstance(decision, NeedInfo)
    assert [a.slot for a in decision.questions] == ["contacted_merchant", "discovery_date", "written_statement_provided"]


def test_a_fraud_dispute_asks_about_the_card_and_pin_and_not_the_merchant():
    decision, NeedInfo = _ask_about("card_present_fraud")
    assert [a.slot for a in decision.questions] == ["card_in_possession", "pin_compromised", "discovery_date", "written_statement_provided"]


def test_everything_the_customer_said_is_not_asked_again_and_the_rest_is_filled_in_by_the_kind_of_dispute():
    from agent.workflows.base import Plan

    decision, _ = _ask_about("duplicate_charge", contacted_merchant=False, discovery_date="11/10/2025", written_statement_provided=True)
    assert isinstance(decision, Plan)
    args = decision.steps[0].args
    assert args["card_in_possession"] is True and args["pin_compromised"] == "no" and args["police_report_filed"] is False


def test_a_police_report_is_asked_for_only_above_500_dollars():
    from banking_tools import BankingSession

    decision, _ = _ask_about("card_present_fraud", transaction_id="btxn_c2g7f43h6g89", disputed_amount=412.88)
    assert "police_report_filed" not in [a.slot for a in decision.questions], "412.88 is under 500"
