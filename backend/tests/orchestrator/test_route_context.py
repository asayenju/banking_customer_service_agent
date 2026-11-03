import asyncio

import pytest
from scenario_kit import GOOD_CLAIMS, RECORD, S, World, I, interp

from banking_tools import BankingSession, ScopedSession
from orchestrator.executor import (
    ROUTE_TIMEOUTS,
    OutcomeStatus,
    RouteOutcome,
)
from orchestrator.routes import Route
from orchestrator.types import IntentKind as K

CARD = "dbc_lj82d4f1a9_bluest"
FREEZE = ("freeze_debit_card_3892", {"card_id": CARD})


def done(intent):
    return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="ok")


class Recording:
    def __init__(self):
        self.contexts = []

    async def run(self, intent, ctx):
        self.contexts.append(ctx)
        return done(intent)


def freeze_intent(**slots):
    return I("frz", K.FREEZE_CARD, {"card_id": CARD, **slots})


def only_outcome(result):
    (outcome,) = result.outcomes
    return outcome


def test_the_callers_identity_comes_from_the_verification_not_from_the_intent():
    direct = Recording()
    w = World(
        [interp(freeze_intent(user_id="someone_else"), claims=GOOD_CLAIMS)],
        [[S("frz", 0.97)]],
        direct=direct,
    )
    w.turn("freeze my card")
    (ctx,) = direct.contexts
    assert ctx.verified_user_id == RECORD.user_id != "someone_else"
    assert ctx.intent_id == "frz"


def test_kinds_open_to_anyone_run_with_no_verified_user():
    sub = Recording()
    w = World([interp(I("q", K.GENERAL_QUESTION))], [[S("q", 0.99)]], sub=sub)
    w.turn("what are your branch hours")
    (ctx,) = sub.contexts
    assert ctx.verified_user_id is None


def test_the_context_carries_a_deadline_inside_the_route_timeout():
    sub = Recording()

    class Clock(Recording):
        async def run(self, intent, ctx):
            self.left = ctx.deadline - asyncio.get_running_loop().time()
            return done(intent)

    sub = Clock()
    w = World([interp(I("q", K.GENERAL_QUESTION))], [[S("q", 0.99)]], sub=sub, route_timeouts={Route.SUB_AGENT: 5.0, Route.DIRECT: 1.0})
    w.turn("what are your branch hours")
    assert 0 < sub.left <= 5.0


def test_a_sub_agent_has_far_longer_than_a_direct_recipe_by_default():
    assert ROUTE_TIMEOUTS[Route.SUB_AGENT] >= 90
    assert ROUTE_TIMEOUTS[Route.DIRECT] <= 10 < ROUTE_TIMEOUTS[Route.SUB_AGENT]
    # a 40 second loop, which the old single 30 second limit would have killed, fits
    assert 40 < ROUTE_TIMEOUTS[Route.SUB_AGENT]


class Slow:
    def __init__(self, seconds):
        self.seconds = seconds

    async def run(self, intent, ctx):
        await asyncio.sleep(self.seconds)
        return done(intent)


def test_each_route_gets_its_own_timeout():
    timeouts = {Route.DIRECT: 0.05, Route.SUB_AGENT: 1.0}
    w = World([interp(I("q", K.GENERAL_QUESTION))], [[S("q", 0.99)]], sub=Slow(0.2), route_timeouts=timeouts)
    assert only_outcome(w.turn("what are your branch hours")).status is OutcomeStatus.DONE

    short = {Route.DIRECT: 1.0, Route.SUB_AGENT: 0.05}
    w = World([interp(I("q", K.GENERAL_QUESTION))], [[S("q", 0.99)]], sub=Slow(0.2), route_timeouts=short)
    outcome = only_outcome(w.turn("what are your branch hours"))
    assert outcome.status is OutcomeStatus.FAILED and "timed out after 0.05s" in outcome.error


def world_with_session(runner, shared):
    return World(
        [interp(freeze_intent(), claims=GOOD_CLAIMS)],
        [[S("frz", 0.97)]],
        direct=runner,
        session_for=lambda intent_id: ScopedSession(shared, intent_id),
        route_timeout=0.1,
    )


class WriteThen:
    """Freezes the card through the context's session, then does ``after``."""

    def __init__(self, after):
        self.after = after

    async def run(self, intent, ctx):
        assert ctx.session.call(*FREEZE).ok
        return await self.after(intent)


async def stall(intent):
    await asyncio.sleep(10)


async def explode(intent):
    raise RuntimeError("card 4111111111111111 exploded")


async def say_failed(intent):
    return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.FAILED, error="gave up")


@pytest.mark.parametrize("after, error_part", [(stall, "timed out"), (explode, "route failed: RuntimeError"), (say_failed, "gave up")])
def test_a_route_that_stops_after_a_committed_write_is_partial_and_says_what_was_done(after, error_part):
    shared = BankingSession()
    w = world_with_session(WriteThen(after), shared)
    result = w.turn("freeze my card")
    outcome = only_outcome(result)
    assert outcome.status is OutcomeStatus.PARTIAL
    assert error_part in outcome.error
    assert outcome.text.startswith("Already done before the problem:")
    assert "4111" not in outcome.text + outcome.error, "an exception message must never be echoed"
    # the write really happened, and is tagged with the intent it was made for
    (entry,) = shared.audit_for("frz")
    assert entry.ok and entry.tool == "freeze_debit_card_3892"


def test_a_route_that_stops_before_any_write_stays_failed():
    shared = BankingSession()

    class Stall:
        async def run(self, intent, ctx):
            await asyncio.sleep(10)

    outcome = only_outcome(world_with_session(Stall(), shared).turn("freeze my card"))
    assert outcome.status is OutcomeStatus.FAILED
    assert shared.audit_for("frz") == ()


def test_a_write_that_failed_is_not_reported_as_done():
    shared = BankingSession()

    class BadWriteThenStall:
        async def run(self, intent, ctx):
            assert not ctx.session.call("freeze_debit_card_3892", {"card_id": "no_such_card"}).ok
            await asyncio.sleep(10)

    outcome = only_outcome(world_with_session(BadWriteThenStall(), shared).turn("freeze my card"))
    assert outcome.status is OutcomeStatus.FAILED


def test_a_runner_without_a_session_still_fails_cleanly():
    w = World([interp(freeze_intent(), claims=GOOD_CLAIMS)], [[S("frz", 0.97)]], direct=Slow(10), route_timeout=0.05)
    assert only_outcome(w.turn("freeze my card")).status is OutcomeStatus.FAILED


def test_the_partial_text_reaches_the_customer_reply():
    shared = BankingSession()
    result = world_with_session(WriteThen(stall), shared).turn("freeze my card")
    assert "Already done before the problem" in result.reply.text
