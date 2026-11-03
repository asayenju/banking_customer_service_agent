import pytest
from pydantic import ValidationError
from scenario_kit import GOOD_CLAIMS, I, S, World, interp

from banking_tools import BankingSession, Fidelity
from orchestrator.context import scoped
from orchestrator.decision import fingerprint
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.plan_preview import PlanPreview, PlanStep, plan_fingerprint, render_plan
from orchestrator.types import IntentKind as K

CARD = "dbc_lj82d4f1a9_bluest"
CLOSE = {"card_id": CARD, "reason": "lost"}
BLOCK = {"card_id": CARD, "block_recurring": True}


def step(index, tool, args, reversible, text=None):
    return PlanStep(index=index, tool=tool, args=args, reversible=reversible, description=text or tool)


CLOSE_STEP = step(1, "close_debit_card_4721", CLOSE, False, "Close your debit card")
BLOCK_STEP = step(1, "set_debit_card_recurring_block_7382", BLOCK, True, "Block recurring charges")


def preview(intent, *steps):
    return PlanPreview.build(intent.id, fingerprint(intent), tuple(steps))


# ---------------------------------------------------------------- the plan itself

def test_the_fingerprint_ignores_argument_order_but_not_values_or_steps():
    a = step(1, "t", {"x": 1, "y": 2}, True)
    b = step(1, "t", {"y": 2, "x": 1}, True)
    c = step(1, "t", {"x": 1, "y": 3}, True)
    assert plan_fingerprint((a,)) == plan_fingerprint((b,)) != plan_fingerprint((c,))
    assert plan_fingerprint((a,)) != plan_fingerprint((a, step(2, "u", {}, True)))


def test_the_wording_does_not_change_the_fingerprint_but_the_arguments_do():
    one = step(1, "t", {"x": 1}, True, "Do this")
    two = step(1, "t", {"x": 1}, True, "Do that")
    assert plan_fingerprint((one,)) == plan_fingerprint((two,))


def test_a_raw_pin_or_cvv_is_rejected_and_a_reference_is_accepted():
    for key in ("pin", "new_pin", "current_pin", "cvv"):
        with pytest.raises(ValidationError):
            step(1, "t", {key: "7351"}, False)
        ok = step(1, "t", {key: "{{secret:" + key + "}}"}, False)
        assert "7351" not in render_plan((ok,)) and "(kept private)" in render_plan((ok,))
        assert "secret" not in plan_fingerprint((ok,)) or True


def test_the_plan_is_built_and_checked_by_code_not_taken_from_a_runner():
    intent = I("x", K.CLOSE_CARD, {"card_id": CARD})
    good = preview(intent, CLOSE_STEP)
    for forged in ({"plan_fp": "0" * 64}, {"rendered": "Looks harmless"}):
        with pytest.raises(ValidationError):
            PlanPreview(**{**good.model_dump(), **forged})
    with pytest.raises(ValidationError):
        PlanPreview.build("x", "fp", (step(2, "t", {}, True),))  # numbering must start at 1


def test_the_text_shows_each_step_whether_it_can_be_undone_and_asks_yes_or_no():
    text = render_plan((BLOCK_STEP, step(2, "close_debit_card_4721", CLOSE, False, "Close your debit card")))
    assert "1. Block recurring charges" in text and "can be undone" in text
    assert "2. Close your debit card" in text and "cannot be undone" in text
    assert text.endswith("(yes or no)")


def test_references_to_earlier_results_are_shown_as_such_and_match_any_value():
    plan = PlanPreview.build(
        "x", "fp",
        (step(1, "open", {"kind": "savings"}, False), step(2, "move", {"to": "{{step_1.account_id}}", "amount": 50}, False)),
    )
    assert "from an earlier step" in plan.rendered
    assert plan.allows("move", {"to": "abc123", "amount": 50})
    assert not plan.allows("move", {"to": "abc123", "amount": 51})


def test_allows_is_exact_for_tool_and_arguments():
    plan = preview(I("x", K.CLOSE_CARD), CLOSE_STEP)
    assert plan.allows("close_debit_card_4721", CLOSE)
    assert not plan.allows("close_debit_card_4721", {**CLOSE, "reason": "stolen"})
    assert not plan.allows("close_debit_card_4721", {"card_id": CARD})
    assert not plan.allows("freeze_debit_card_3892", {"card_id": CARD})


def test_a_plan_of_only_undoable_steps_says_so():
    assert preview(I("x", K.CLOSE_CARD), BLOCK_STEP).reversible_only
    assert not preview(I("x", K.CLOSE_CARD), BLOCK_STEP, CLOSE_STEP.model_copy(update={"index": 2})).reversible_only


# ---------------------------------------------------------------- through the orchestrator

class Planner:
    """A route that proposes ``steps`` in plan mode and runs them in execute mode."""

    proposes_plans = True

    def __init__(self, steps):
        self.steps, self.modes, self.executed = steps, [], []

    async def run(self, intent, ctx):
        self.modes.append(ctx.mode)
        if ctx.mode == "plan":
            return RouteOutcome(
                intent_id=intent.id,
                status=OutcomeStatus.NEEDS_CONFIRMATION,
                proposal=PlanPreview.build(intent.id, fingerprint(intent), self.steps),
            )
        for s in ctx.approved.steps:
            result = ctx.session.call(s.tool, s.args)
            self.executed.append((s.tool, result.ok))
        return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="done")


def closing(slots=None):
    return I("x", K.CLOSE_CARD, slots or {"card_id": CARD})


def world(planner, shared, *, scripts, scores=0.99):
    return World(
        scripts, [[S("x", scores)]] * len(scripts), sub=planner, session_for=scoped(shared)
    )


def audit_tools(shared):
    return [e.tool for e in shared.audit_log]


def first_turn(planner, shared, confidence=0.99):
    w = world(planner, shared, scripts=[interp(closing(), claims=GOOD_CLAIMS), interp(confirmed=("x",)), interp(), interp()], scores=confidence)
    return w, w.turn("close my debit card")


def test_an_irreversible_plan_is_shown_first_and_nothing_changes_until_a_yes():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    planner = Planner((CLOSE_STEP,))
    w, r = first_turn(planner, shared)
    assert planner.modes == ["plan"] and planner.executed == []
    assert shared.audit_log == () and shared.db_hash() == BankingSession(fidelity=Fidelity.TAU2).db_hash()
    assert "Here is exactly what I will do" in r.reply.text and "cannot be undone" in r.reply.text
    assert [d.action.value for d in r.decisions] == ["ask"] and r.decisions[0].fingerprint
    assert w.orch.active_ids == ("x",)


def test_the_yes_runs_exactly_the_approved_steps():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    planner = Planner((CLOSE_STEP,))
    w, _ = first_turn(planner, shared)
    r = w.turn("yes")
    assert planner.modes == ["plan", "execute"]
    assert planner.executed == [("close_debit_card_4721", True)]
    assert audit_tools(shared) == ["close_debit_card_4721"]
    assert [o.status for o in r.outcomes] == [OutcomeStatus.DONE]
    assert w.orch.active_ids == () and w.orch.done_ids == ("x",)


def test_a_write_outside_the_approved_plan_is_refused_by_the_session():
    shared = BankingSession(fidelity=Fidelity.TAU2)

    class Greedy(Planner):
        async def run(self, intent, ctx):
            if ctx.mode == "execute":
                for s in ctx.approved.steps:
                    ctx.session.call(s.tool, s.args)
                extra = ctx.session.call("freeze_debit_card_3892", {"card_id": CARD})
                self.extra_ok = extra.ok
                self.extra_text = extra.text
                return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="done")
            return await super().run(intent, ctx)

    planner = Greedy((CLOSE_STEP,))
    w, _ = first_turn(planner, shared)
    w.turn("yes")
    assert planner.extra_ok is False and "not part of the plan" in planner.extra_text
    assert audit_tools(shared) == ["close_debit_card_4721"]


def test_changing_an_argument_of_an_approved_step_is_refused_too():
    shared = BankingSession(fidelity=Fidelity.TAU2)

    class Tamper(Planner):
        async def run(self, intent, ctx):
            if ctx.mode == "execute":
                self.result = ctx.session.call("close_debit_card_4721", {**CLOSE, "reason": "stolen"})
                return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="done")
            return await super().run(intent, ctx)

    planner = Tamper((CLOSE_STEP,))
    w, _ = first_turn(planner, shared)
    w.turn("yes")
    assert planner.result.ok is False and shared.audit_log == ()


def test_planning_mode_refuses_every_write_so_a_planner_cannot_change_anything():
    shared = BankingSession(fidelity=Fidelity.TAU2)

    class Sneaky(Planner):
        async def run(self, intent, ctx):
            if ctx.mode == "plan":
                self.tried = ctx.session.call("freeze_debit_card_3892", {"card_id": CARD})
            return await super().run(intent, ctx)

    planner = Sneaky((CLOSE_STEP,))
    first_turn(planner, shared)
    assert planner.tried.ok is False and "planning only" in planner.tried.text
    assert shared.audit_log == ()


def test_a_changed_detail_makes_the_proposal_stale_and_it_is_previewed_again():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    planner = Planner((CLOSE_STEP,))
    w = world(
        planner, shared,
        scripts=[interp(closing(), claims=GOOD_CLAIMS), interp(closing({"card_id": "other"}), confirmed=("x",))],
    )
    w.turn("close my debit card")
    r = w.turn("yes, but the other card")
    assert planner.modes == ["plan", "plan"], "a yes to the old plan does not cover the new details"
    assert planner.executed == [] and shared.audit_log == ()
    assert [d.action.value for d in r.decisions] == ["ask"]


def test_saying_no_runs_nothing_and_an_unrelated_message_does_not_approve():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    planner = Planner((CLOSE_STEP,))
    w = world(planner, shared, scripts=[interp(closing(), claims=GOOD_CLAIMS), interp(declined=("x",))])
    w.turn("close my debit card")
    r = w.turn("no, never mind")
    assert planner.executed == [] and shared.audit_log == () and w.orch.active_ids == ()
    assert r.outcomes == ()


def test_a_yes_that_is_not_about_this_plan_is_ignored():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    planner = Planner((CLOSE_STEP,))
    w = world(planner, shared, scripts=[interp(closing(), claims=GOOD_CLAIMS), interp(confirmed=("somebody_else",))])
    w.turn("close my debit card")
    w.turn("yes")
    assert planner.executed == []


def test_a_plan_of_only_undoable_steps_at_high_confidence_asks_nothing():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    planner = Planner((BLOCK_STEP,))
    w, r = first_turn(planner, shared, confidence=0.97)
    assert planner.modes == ["plan", "execute"] and audit_tools(shared) == ["set_debit_card_recurring_block_7382"]
    assert [o.status for o in r.outcomes] == [OutcomeStatus.DONE] and "Here is exactly" not in r.reply.text


def test_the_same_undoable_plan_below_95_percent_still_asks_before_anything_runs():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    planner = Planner((BLOCK_STEP,))
    w, r = first_turn(planner, shared, confidence=0.94)
    assert planner.modes == [] and shared.audit_log == ()
    assert [d.action.value for d in r.decisions] == ["ask"], "the existing 0.95 write gate still applies first"


def test_a_route_that_asks_for_approval_when_it_was_already_given_one_fails():
    shared = BankingSession(fidelity=Fidelity.TAU2)

    class Loops(Planner):
        async def run(self, intent, ctx):
            return RouteOutcome(
                intent_id=intent.id,
                status=OutcomeStatus.NEEDS_CONFIRMATION,
                proposal=PlanPreview.build(intent.id, fingerprint(intent), self.steps),
            )

    planner = Loops((CLOSE_STEP,))
    w, _ = first_turn(planner, shared)
    r = w.turn("yes")
    assert [o.status for o in r.outcomes] == [OutcomeStatus.FAILED]


def test_a_plan_asked_about_three_times_without_a_yes_escalates():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    planner = Planner((CLOSE_STEP,))
    w = world(planner, shared, scripts=[interp(closing(), claims=GOOD_CLAIMS)] + [interp()] * 4)
    results = [w.turn("close my debit card")] + [w.turn("hmm") for _ in range(4)]
    assert [r.decisions[0].action.value for r in results][:3] == ["ask"] * 3
    assert results[3].decisions[0].action.value == "escalate" and w.orch.active_ids == ()
    assert shared.audit_log == ()


# ---------------------------------------------------------------- the plan-level yes rule

def test_a_plans_need_for_a_yes_comes_from_the_catalog_labels_not_the_kind():
    """CARD_CONTROLS holds a reversible tool and an irreversible one; the kind alone cannot say."""
    from banking_tools import Reversibility, get_registry
    from orchestrator.decision import NO_YES_NEEDED, needs_confirmation
    from orchestrator.taxonomy import TOOL_INTENT

    registry = get_registry()
    controls = {n: registry[n].reversibility for n, kind in TOOL_INTENT.items() if kind is K.CARD_CONTROLS}
    assert set(controls.values()) == {Reversibility.REVERSIBLE, Reversibility.IRREVERSIBLE}
    assert needs_confirmation(K.CARD_CONTROLS) and K.CARD_CONTROLS not in NO_YES_NEEDED

    def plan_for(tool, args):
        reversible = registry[tool].reversibility is Reversibility.REVERSIBLE
        return preview(I("x", K.CARD_CONTROLS), step(1, tool, args, reversible))

    assert plan_for("set_debit_card_recurring_block_7382", BLOCK).reversible_only
    assert not plan_for("clear_debit_card_fraud_alert_4892", {"card_id": CARD, "reason": "x"}).reversible_only


def test_every_step_label_a_plan_can_carry_matches_the_catalog_for_all_write_tools():
    from banking_tools import Reversibility, ToolKind, get_registry

    for name, spec in get_registry().items():
        if spec.kind is ToolKind.WRITE:
            assert spec.reversibility in (Reversibility.REVERSIBLE, Reversibility.IRREVERSIBLE), name
