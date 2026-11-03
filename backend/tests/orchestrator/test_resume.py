import asyncio

from scenario_kit import GOOD_CLAIMS, S, World, I, interp

from banking_tools import BankingSession, Fidelity, ScopedSession
from banking_tools.types import CompletedStep
from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.types import IntentKind as K

CARD = "dbc_lj82d4f1a9_bluest"
FREEZE = ("freeze_debit_card_3892", {"card_id": CARD})
UNFREEZE = ("unfreeze_debit_card_3893", {"card_id": CARD})
BLOCK = ("set_debit_card_recurring_block_7382", {"card_id": CARD, "block_recurring": True})


def done(intent):
    return RouteOutcome(intent_id=intent.id, status=OutcomeStatus.DONE, text="all done")


class Steps:
    """Makes the given writes in order, crashing before step ``crash_before`` on the
    first run only (like a model call that errors out partway)."""

    def __init__(self, steps, crash_before=None):
        self.steps, self.crash_before, self.runs, self.progress_seen = steps, crash_before, 0, []

    async def run(self, intent, ctx):
        self.runs += 1
        self.progress_seen.append(len(ctx.progress))
        for index, (name, arguments) in enumerate(self.steps):
            if self.runs == 1 and index == self.crash_before:
                raise RuntimeError("model call failed")
            assert ctx.session.call(name, arguments).ok
        return done(intent)


def freeze():
    return I("frz", K.FREEZE_CARD, {"card_id": CARD})


def world(runner, shared, *, extra=()):
    return World(
        [interp(freeze(), *extra, claims=GOOD_CLAIMS)] + [interp(freeze())] * 3,
        [[S("frz", 0.97)] + [S(e.id, 0.97) for e in extra]] + [[S("frz", 0.97)]] * 3,
        direct=runner,
        session_for=lambda intent_id, progress: ScopedSession(shared, intent_id, progress),
    )


def tools_written(shared):
    return [e.tool for e in shared.audit_for("frz")]


def test_a_partly_finished_intent_is_kept_for_a_retry_and_not_rerun_on_its_own():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    runner = Steps([BLOCK, FREEZE], crash_before=1)
    w = World(
        [interp(freeze(), claims=GOOD_CLAIMS), interp()],
        [[S("frz", 0.97)]],
        direct=runner,
        session_for=lambda intent_id, progress: ScopedSession(shared, intent_id, progress),
    )
    first = w.turn("freeze my card")
    assert [o.status for o in first.outcomes] == [OutcomeStatus.PARTIAL]
    assert "human agent" in first.reply.text and "Already done before the problem" in first.reply.text
    assert w.orch.resumable_ids == ("frz",) and w.orch.active_ids == ()
    unrelated = w.turn("I will check my statement later")
    assert runner.runs == 1, "nothing is retried until the customer asks"
    assert w.orch.resumable_ids == ("frz",) and unrelated.outcomes == ()


def test_a_retry_skips_the_writes_already_made_and_does_the_rest():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    runner = Steps([BLOCK, FREEZE], crash_before=1)
    w = world(runner, shared)
    w.turn("freeze my card")
    # the customer asks again; the decomposer finds the open intent by its id
    retry = w.turn("yes, please try again")
    assert [o.status for o in retry.outcomes] == [OutcomeStatus.DONE]
    assert runner.progress_seen == [0, 1], "the second run was told one step was done"
    assert tools_written(shared) == ["set_debit_card_recurring_block_7382", "freeze_debit_card_3892"], (
        "the block was made once, not twice"
    )
    assert w.orch.resumable_ids == () and w.orch._progress == {}


def test_a_repeated_legitimate_write_is_not_mistaken_for_a_retry():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    # freeze, unfreeze, freeze: the third call has the same tool and arguments as the first
    runner = Steps([FREEZE, UNFREEZE, FREEZE], crash_before=2)
    w = world(runner, shared)
    w.turn("freeze my card")
    assert tools_written(shared) == ["freeze_debit_card_3892", "unfreeze_debit_card_3893"]
    w.turn("try again")
    assert tools_written(shared) == [
        "freeze_debit_card_3892",
        "unfreeze_debit_card_3893",
        "freeze_debit_card_3892",
    ], "steps 1 and 2 were skipped and step 3 was really made"


def test_dependents_of_a_partial_result_are_skipped_and_a_human_is_offered():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    runner = Steps([BLOCK, FREEZE], crash_before=1)
    follow = I("q", K.GENERAL_QUESTION, depends_on=("frz",))
    w = world(runner, shared, extra=(follow,))
    first = w.turn("freeze my card and tell me the fees")
    statuses = {o.intent_id: o.status for o in first.outcomes}
    assert statuses == {"frz": OutcomeStatus.PARTIAL, "q": OutcomeStatus.SKIPPED}
    assert "human agent" in first.reply.text


def test_withdrawing_a_partly_finished_intent_drops_it():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    runner = Steps([BLOCK, FREEZE], crash_before=1)
    w = World(
        [interp(freeze(), claims=GOOD_CLAIMS), interp(declined=("frz",))],
        [[S("frz", 0.97)]],
        direct=runner,
        session_for=lambda intent_id, progress: ScopedSession(shared, intent_id, progress),
    )
    w.turn("freeze my card")
    assert w.orch.resumable_ids == ("frz",)
    w.turn("never mind")
    assert w.orch.resumable_ids == () and w.orch._progress == {}


# --- ScopedSession on its own ---


def completed(shared, intent_id):
    return ScopedSession(shared, intent_id).completed()


def test_scoped_session_skips_a_matching_prefix_and_reports_the_original_audit_id():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    first = ScopedSession(shared, "i")
    original = first.call(*FREEZE)
    prior = first.completed()
    again = ScopedSession(shared, "i", prior)
    result = again.call(*FREEZE)
    assert result.ok and result.audit_id == original.audit_id and "Already done" in result.text
    assert len(shared.audit_for("i")) == 1


def test_scoped_session_reads_pass_through_and_do_not_advance_the_position():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    first = ScopedSession(shared, "i")
    first.call(*BLOCK)
    again = ScopedSession(shared, "i", first.completed())
    assert again.call("get_user_information_by_id", {"user_id": "123"}).ok
    assert "Already done" in again.call(*BLOCK).text


def test_scoped_session_stops_skipping_at_the_first_difference():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    first = ScopedSession(shared, "i")
    first.call(*BLOCK)
    first.call(*FREEZE)
    again = ScopedSession(shared, "i", first.completed())
    # a different first write: nothing is skipped from here on, even the later match
    assert "Already done" not in again.call(*UNFREEZE).text
    assert "Already done" not in again.call(*BLOCK).text
    assert len([e for e in shared.audit_for("i") if e.tool == BLOCK[0]]) == 2


def test_pins_are_compared_redacted_and_never_stored_in_clear():
    shared = BankingSession(fidelity=Fidelity.TAU2)
    first = ScopedSession(shared, "i")
    args = {"card_id": CARD, "last_4_digits": "4821", "new_pin": "7351"}
    assert first.call("reset_debit_card_pin_6284", args).ok
    (step,) = first.completed()
    assert "7351" not in repr(step) and step.arguments["new_pin"] == "***"
