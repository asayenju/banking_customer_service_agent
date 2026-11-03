import asyncio

import pytest
from pydantic import ValidationError

from orchestrator.executor import ExecutionResult, OutcomeStatus, RouteOutcome, execute_plan
from orchestrator.mocks import Behavior, ScriptedRoute
from orchestrator.plan import Plan, plan
from orchestrator.routes import Route
from orchestrator.trace import TraceKind
from orchestrator.types import Intent, IntentKind

K = IntentKind
T = TraceKind
SECRET = "boom with card 4111111111111111"


def I(id, kind=K.LOOKUP_INFO, slots=None, depends_on=()):
    return Intent(id=id, kind=kind, summary=f"summary of {id}", slots=slots or {}, depends_on=depends_on)


def write(id, kind=K.CLOSE_CARD, card="c1", depends_on=()):
    return I(id, kind, {"card_id": card}, depends_on)


def run(intents, behaviors=None, *, sub_behaviors=None, timeout=0.05, runners=None):
    direct = ScriptedRoute(behaviors)
    sub = ScriptedRoute(sub_behaviors if sub_behaviors is not None else behaviors)
    the_plan = plan(intents)
    assert isinstance(the_plan, Plan)
    result = asyncio.run(
        execute_plan(
            the_plan,
            {i.id: i for i in intents},
            runners or {Route.DIRECT: direct, Route.SUB_AGENT: sub},
            timeout=timeout,
        )
    )
    return result, direct, sub


def status(result):
    return {o.intent_id: o.status for o in result.outcomes}


def kinds(result, intent_id):
    return [e.kind for e in result.trace.events if e.intent_id == intent_id]


def test_one_intent_runs_and_leaves_a_routed_started_finished_trace():
    result, direct, _ = run([I("a")])
    assert status(result) == {"a": OutcomeStatus.DONE}
    assert [(e.kind, e.wave) for e in result.trace.events] == [
        (T.ROUTED, 1),
        (T.STARTED, 1),
        (T.FINISHED, 1),
    ]
    assert direct.calls == ["a"]


def test_each_intent_goes_to_the_runner_for_its_route():
    result, direct, sub = run([I("a", K.FREEZE_CARD), I("b", K.DISPUTE_TRANSACTION)])
    assert direct.calls == ["a"] and sub.calls == ["b"]
    assert [e.detail for e in result.trace.events if e.kind is T.ROUTED] == ["direct", "sub_agent"]


def test_independent_reads_run_at_the_same_time():
    result, direct, _ = run([I("a"), I("b")], {"a": Behavior.SLOW, "b": Behavior.SLOW})
    assert direct.peak_concurrency == 2
    order = [(e.kind, e.intent_id) for e in result.trace.events if e.kind in (T.STARTED, T.FINISHED)]
    assert order[:2] == [(T.STARTED, "a"), (T.STARTED, "b")]
    assert {e.wave for e in result.trace.events} == {1}


def test_writes_on_the_same_card_run_one_after_the_other():
    result, _, sub = run(
        [write("a"), write("b")], {"a": Behavior.SLOW, "b": Behavior.SLOW}
    )
    assert sub.peak_concurrency == 1
    order = [(e.kind, e.intent_id) for e in result.trace.events if e.kind in (T.STARTED, T.FINISHED)]
    assert order == [(T.STARTED, "a"), (T.FINISHED, "a"), (T.STARTED, "b"), (T.FINISHED, "b")]
    assert {e.wave for e in result.trace.events if e.intent_id == "b"} == {2}


def test_a_route_that_raises_becomes_a_failure_without_raising_or_leaking_the_message():
    result, _, _ = run([I("a")], {"a": Behavior.RAISES})
    outcome = result.outcomes[0]
    assert outcome.status is OutcomeStatus.FAILED
    assert outcome.error == "route failed: RuntimeError"
    assert "4111" not in repr(result)
    assert kinds(result, "a") == [T.ROUTED, T.STARTED, T.FAILED]
    assert result.trace.events[-1].reason == "route failed: RuntimeError"


def test_a_route_that_takes_too_long_fails_with_a_timeout():
    result, _, _ = run([I("a")], {"a": Behavior.TIMEOUT}, timeout=0.05)
    assert result.outcomes[0].status is OutcomeStatus.FAILED
    assert result.outcomes[0].error == "timed out after 0.05s"


def test_a_malformed_result_becomes_a_failure():
    result, _, _ = run([I("a")], {"a": Behavior.MALFORMED})
    assert result.outcomes[0].status is OutcomeStatus.FAILED
    assert result.outcomes[0].error == "route returned a malformed result"


def test_a_partial_result_keeps_what_was_done_and_is_traced_as_a_failure():
    result, _, _ = run([I("a")], {"a": Behavior.PARTIAL})
    outcome = result.outcomes[0]
    assert outcome.status is OutcomeStatus.PARTIAL
    assert outcome.text == "did the first step" and outcome.error == "second step not completed"
    assert kinds(result, "a")[-1] is T.FAILED


def test_a_failure_skips_everything_downstream_but_independent_intents_still_run():
    intents = [
        I("a"),
        I("b", depends_on=("a",)),
        I("c", depends_on=("b",)),
        I("d"),
    ]
    result, direct, _ = run(intents, {"a": Behavior.RAISES})
    assert status(result) == {
        "a": OutcomeStatus.FAILED,
        "b": OutcomeStatus.SKIPPED,
        "c": OutcomeStatus.SKIPPED,
        "d": OutcomeStatus.DONE,
    }
    assert direct.calls == ["a", "d"]
    skipped = [e for e in result.trace.events if e.kind is T.SKIPPED]
    assert [(e.intent_id, e.reason) for e in skipped] == [
        ("b", "dependency 'a' did not complete"),
        ("c", "dependency 'b' did not complete"),
    ]


def test_a_partial_result_also_skips_its_dependents():
    result, _, _ = run([I("a"), I("b", depends_on=("a",))], {"a": Behavior.PARTIAL})
    assert status(result)["b"] is OutcomeStatus.SKIPPED


def test_work_already_done_is_left_alone_when_a_later_step_fails():
    result, direct, _ = run(
        [I("a"), I("b", depends_on=("a",))], {"b": Behavior.RAISES}
    )
    assert status(result) == {"a": OutcomeStatus.DONE, "b": OutcomeStatus.FAILED}
    assert direct.calls == ["a", "b"]


def test_a_route_with_no_runner_is_a_failure_not_an_exception():
    sub = ScriptedRoute()
    result, _, _ = run([I("a")], runners={Route.SUB_AGENT: sub})
    assert result.outcomes[0].status is OutcomeStatus.FAILED
    assert result.outcomes[0].error == "no runner for route direct"


def test_an_intent_the_plan_names_but_we_do_not_have_is_a_failure():
    the_plan = Plan(waves=(("ghost",),), waiting=())
    result = asyncio.run(execute_plan(the_plan, {}, {}, timeout=0.05))
    assert result.outcomes[0].status is OutcomeStatus.FAILED
    assert result.outcomes[0].error == "unknown intent"


def test_an_empty_plan_does_nothing():
    result = asyncio.run(execute_plan(Plan(waves=(), waiting=()), {}, {}))
    assert result == ExecutionResult(outcomes=(), trace=result.trace)
    assert result.trace.events == ()


def test_outcome_shape_is_enforced():
    RouteOutcome(intent_id="a", status=OutcomeStatus.DONE, text="ok")
    with pytest.raises(ValidationError):
        RouteOutcome(intent_id="a", status=OutcomeStatus.DONE, error="x")
    for bad in (OutcomeStatus.FAILED, OutcomeStatus.PARTIAL, OutcomeStatus.SKIPPED):
        with pytest.raises(ValidationError):
            RouteOutcome(intent_id="a", status=bad)


def test_outcome_is_frozen_and_strict():
    outcome = RouteOutcome(intent_id="a", status=OutcomeStatus.DONE)
    with pytest.raises(ValidationError):
        outcome.text = "changed"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RouteOutcome(intent_id="a", status=OutcomeStatus.DONE, surprise="x")


# --- NEEDS_INFO and reconciliation with the session audit ---


class FakeSession:
    def __init__(self, steps=()):
        self.steps = list(steps)

    def call(self, name, arguments=None):
        raise AssertionError("not used")

    def committed_steps(self):
        return self.steps

    def completed(self):
        return ()


class Asks:
    async def run(self, intent, ctx=None):
        from orchestrator.types import Question

        return RouteOutcome(
            intent_id=intent.id,
            status=OutcomeStatus.NEEDS_INFO,
            questions=(Question(intent_id=intent.id, slot="card_id"),),
        )


def test_needs_info_is_traced_as_a_question_not_a_failure_and_dependents_wait():
    intents = [I("a"), I("b", depends_on=("a",))]
    result, _, _ = run(intents, runners={Route.DIRECT: Asks(), Route.SUB_AGENT: Asks()})
    first, second = result.outcomes
    assert first.status is OutcomeStatus.NEEDS_INFO
    assert second.status is OutcomeStatus.SKIPPED and "waiting for an answer" in second.error
    kinds = [e.kind for e in result.trace.events if e.intent_id == "a"]
    assert T.ASKED in kinds and T.FAILED not in kinds


def test_a_route_that_fails_after_committing_a_write_is_reported_as_partial():
    class Fails:
        async def run(self, intent, ctx=None):
            return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.FAILED, error="gave up")

    session = FakeSession(["Freeze a debit card"])
    from orchestrator.context import RouteContext

    intents = [I("a"), I("b", depends_on=("a",))]
    the_plan = plan(intents)
    result = asyncio.run(
        execute_plan(
            the_plan,
            {i.id: i for i in intents},
            {Route.DIRECT: Fails(), Route.SUB_AGENT: Fails()},
            timeout=0.05,
            context_for=lambda intent, seconds: RouteContext(intent_id=intent.id, session=session),
        )
    )
    first, second = result.outcomes
    assert first.status is OutcomeStatus.PARTIAL and "Freeze a debit card" in first.text
    assert first.error == "gave up", "the original reason is kept for the trace"
    assert second.status is OutcomeStatus.SKIPPED, "dependents of a partial result are still skipped"


def test_an_audit_that_cannot_be_read_does_not_hide_the_original_failure():
    class Broken(FakeSession):
        def committed_steps(self):
            raise RuntimeError("audit down")

    from orchestrator.context import RouteContext

    class Fails:
        async def run(self, intent, ctx=None):
            raise RuntimeError("route down")

    intents = [I("a")]
    result = asyncio.run(
        execute_plan(
            plan(intents),
            {"a": intents[0]},
            {Route.DIRECT: Fails(), Route.SUB_AGENT: Fails()},
            timeout=0.05,
            context_for=lambda intent, seconds: RouteContext(intent_id=intent.id, session=Broken()),
        )
    )
    assert result.outcomes[0].status is OutcomeStatus.FAILED
    assert result.outcomes[0].error == "route failed: RuntimeError"
