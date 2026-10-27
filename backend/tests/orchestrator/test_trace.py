import pytest
from pydantic import ValidationError

from orchestrator.trace import Trace, TraceEvent, TraceKind

NEEDS_A_REASON = [TraceKind.FAILED, TraceKind.SKIPPED, TraceKind.ESCALATED]
NEEDS_NONE = [
    TraceKind.SCORED,
    TraceKind.DECIDED,
    TraceKind.ROUTED,
    TraceKind.STARTED,
    TraceKind.FINISHED,
    TraceKind.ASKED,
    TraceKind.FLAGGED,
]


@pytest.mark.parametrize("kind", NEEDS_A_REASON)
def test_failure_like_events_must_carry_a_reason(kind):
    with pytest.raises(ValidationError, match="must carry a reason"):
        TraceEvent(kind=kind, intent_id="a")


@pytest.mark.parametrize("kind", NEEDS_A_REASON)
def test_failure_like_events_accept_a_reason(kind):
    assert TraceEvent(kind=kind, intent_id="a", reason="timeout").reason == "timeout"


@pytest.mark.parametrize("kind", NEEDS_NONE)
def test_other_events_need_no_reason(kind):
    assert TraceEvent(kind=kind, intent_id="a").reason is None


def test_an_event_can_be_about_the_whole_turn():
    assert TraceEvent(kind=TraceKind.ASKED).intent_id is None


def test_an_event_can_say_which_wave_it_happened_in():
    assert TraceEvent(kind=TraceKind.STARTED, intent_id="a", wave=2).wave == 2
    assert TraceEvent(kind=TraceKind.ASKED).wave is None


def test_event_is_frozen():
    event = TraceEvent(kind=TraceKind.STARTED, intent_id="a")
    with pytest.raises(ValidationError):
        event.intent_id = "b"


def test_event_rejects_extra_fields():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TraceEvent(kind=TraceKind.STARTED, surprise="x")


def test_event_kind_must_be_the_enum():
    with pytest.raises(ValidationError):
        TraceEvent(kind="started")


def test_trace_keeps_events_in_order():
    events = (
        TraceEvent(kind=TraceKind.SCORED, intent_id="a"),
        TraceEvent(kind=TraceKind.STARTED, intent_id="a"),
        TraceEvent(kind=TraceKind.FINISHED, intent_id="a"),
    )
    assert Trace(events=events).events == events
