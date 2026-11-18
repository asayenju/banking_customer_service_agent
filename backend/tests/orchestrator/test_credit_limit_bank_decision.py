"""A denial is the bank's own decision under its own rules: recorded without a yes, and the customer is told why."""

from datetime import date

import pytest
from pydantic import ValidationError

from agent.workflows.base import Plan, Step, WorkflowContext
from agent.workflows.credit_limit import CreditLimit, LimitFacts
from banking_tools import BankingSession, Fidelity
from banking_tools.manifest import load_tasks
from orchestrator.plan_preview import BANK_DECISION_TOOLS, PlanPreview, PlanStep
from orchestrator.types import IntentKind as K
from orchestrator.decision import fingerprint
from orchestrator.executor import OutcomeStatus, RouteOutcome
from scenario_kit import GOOD_CLAIMS, I, S, World, interp
from orchestrator.context import scoped

TODAY = date(2026, 11, 17)


def world(task_id):
    """Zhang Mei: task_051 (balance $3,000 of $4,000, 75%: too high) and task_052 (balance $1,500: fine)."""
    task = next(t for t in load_tasks() if t["id"] == task_id)
    session = BankingSession.from_task(task)
    user = next(k for k, v in session.database.users.data.items() if v.get("email") == "meimeimeimei@icloud.com")
    return session.database, user, next(a["account_id"] for a in session.database.credit_card_accounts.data.values() if a["user_id"] == user and "Bronze" in a["card_type"])


def decide(task_id, amount=300):
    db, user, card = world(task_id)
    ctx = WorkflowContext(db, user, TODAY, {})
    return db, ctx, CreditLimit().decide(ctx, LimitFacts(credit_card_account_id=card, requested_increase_amount=amount))


def test_a_request_that_fails_a_rule_is_the_banks_decision_and_needs_no_yes():
    _, _, out = decide("task_051")
    assert isinstance(out, Plan) and out.bank_decision
    assert [s.tool for s in out.steps] == ["submit_credit_limit_increase_request_7392", "deny_credit_limit_increase_5848"]
    assert out.steps[1].args["denial_reason"] == "high_utilization"
    assert all(s.note for s in out.steps), "every step reads as plain words, not the bank tool's description"


def test_a_request_that_passes_every_rule_is_shown_as_a_plan_and_waits_for_the_customers_yes():
    _, _, out = decide("task_052")
    assert isinstance(out, Plan) and not out.bank_decision
    assert [s.tool for s in out.steps] == ["submit_credit_limit_increase_request_7392", "approve_credit_limit_increase_5847"]
    assert "from $4,000 to $4,300" in out.steps[1].note and "$300" in out.steps[0].note


def test_the_customer_is_told_why_in_plain_words_after_a_decline():
    _, ctx, out = decide("task_051")
    said = CreditLimit().reply(ctx, out.steps)
    assert "can't raise the limit" in said and "$3,000.00 of your $4,000.00 limit (75%)" in said and "under 70%" in said
    assert "declined" in said and "$300" in said and "deny" not in said.lower()
    assert CreditLimit().reply(ctx, decide("task_052")[2].steps) == "", "an approval is worded by the plan, not here"


def step(index, tool):
    return PlanStep(index=index, tool=tool, args={"credit_card_account_id": "x"}, reversible=False, description=tool)


def test_only_the_banks_decision_steps_can_ever_skip_the_yes():
    assert BANK_DECISION_TOOLS == {"submit_credit_limit_increase_request_7392", "deny_credit_limit_increase_5848"}
    ok = PlanPreview.build("i", "fp", (step(1, "submit_credit_limit_increase_request_7392"), step(2, "deny_credit_limit_increase_5848")), bank_decision=True)
    assert ok.bank_decision
    for tools in (("close_credit_card_account_7834",), ("submit_credit_limit_increase_request_7392", "approve_credit_limit_increase_5847")):
        with pytest.raises(ValidationError):
            PlanPreview.build("i", "fp", tuple(step(n, t) for n, t in enumerate(tools, 1)), bank_decision=True)


class Planner:
    proposes_plans = True

    def __init__(self, steps, bank_decision):
        self.steps, self.bank_decision, self.modes = steps, bank_decision, []

    async def run(self, intent, ctx):
        self.modes.append(ctx.mode)
        if ctx.mode == "plan":
            return RouteOutcome(
                intent_id=intent.id,
                status=OutcomeStatus.NEEDS_CONFIRMATION,
                proposal=PlanPreview.build(intent.id, fingerprint(intent), self.steps, bank_decision=self.bank_decision),
            )
        return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="recorded")


def run_turn(planner):
    shared = BankingSession(fidelity=Fidelity.TAU2)
    w = World([interp(I("x", K.CREDIT_LIMIT_REQUEST, {"card": "bronze"}), claims=GOOD_CLAIMS)], [[S("x", 0.99)]], sub=planner, session_for=scoped(shared))
    return w.turn("raise my limit by $300")


def test_the_orchestrator_runs_a_bank_decision_in_the_same_turn_without_asking():
    planner = Planner((step(1, "submit_credit_limit_increase_request_7392"), step(2, "deny_credit_limit_increase_5848")), bank_decision=True)
    result = run_turn(planner)
    assert planner.modes == ["plan", "execute"] and "yes" not in result.reply.text.lower()
    assert [d.reason for d in result.decisions] == ["the bank records its own decision"]


def test_any_other_irreversible_plan_still_waits_for_the_yes():
    planner = Planner((step(1, "submit_credit_limit_increase_request_7392"), step(2, "approve_credit_limit_increase_5847")), bank_decision=False)
    result = run_turn(planner)
    assert planner.modes == ["plan"] and "Here is exactly what I will do" in result.reply.text
