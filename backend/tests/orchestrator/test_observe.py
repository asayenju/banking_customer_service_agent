"""Seeing where a turn went wrong: the notes, the failure label, the record, and what must never be in them."""

import json

from fastapi.testclient import TestClient

import main
from agent.assembly import build_conversation
from orchestrator import observe
from orchestrator.decomposer import Interpretation, ScriptedDecomposer
from orchestrator.observe import Recorder, failure_class, note, render_turn, summarise
from orchestrator.scoring import IntentScore
from orchestrator.types import Decomposition, Intent, IntentKind as K, Question
from banking_tools import BankingSession
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex
from agent.llm_tools import FakeToolLLM


def test_a_note_does_nothing_when_no_turn_is_being_recorded():
    note("agent", stop="terminal")  # no collector: must not raise


def test_a_note_is_redacted_and_shortened_and_keeps_only_plain_values():
    notes = observe.begin()
    note("x", why="my pin is 7351 " + "word " * 100, count=3, ok=True, tools=["a", "b"], junk={"no": "dict"})
    (only,) = notes
    assert "7351" not in only["why"] and len(only["why"]) <= 160
    assert only["count"] == 3 and only["ok"] is True and only["tools"] == ["a", "b"] and "junk" not in only


def decision(intent, action, reason, asked=()):
    return {"intent": intent, "action": action, "reason": reason, "asked": list(asked)}


def label(decisions=(), outcomes=(), notes=(), intents=("a",), before=None, asked=None):
    return failure_class(decisions, outcomes, notes, intents, before or {}, asked or {})


def test_each_kind_of_failure_gets_its_own_label_most_severe_first():
    assert label(intents=("unprocessed",)) == "decomposer_failed"
    assert label(notes=[{"kind": "decomposer_failed"}]) == "decomposer_failed"
    assert label([decision("a", "escalate", "still not enough information after 3 questions from the route")]) == "ask_limit_escalation"
    assert label(notes=[{"kind": "agent", "stop": "model_failed"}]) == "agent_stopped"
    assert label(outcomes=[{"status": "failed"}]) == "route_failed"
    assert label([decision("a", "ask", "awaiting the customer's yes to the plan; the plan the customer saw went stale: details changed (card_id)")]) == "plan_went_stale"
    assert label(notes=[{"kind": "workflow", "outcome": "Refuse"}]) == "workflow_refused"
    assert label([decision("a", "escalate", "this kind hands off to a human")]) == "escalated"
    assert label([decision("a", "run", "ok")]) is None


def test_asking_the_same_question_two_turns_in_a_row_is_a_repeated_ask():
    assert label(before={"a": "credit_card_account_id"}, asked={"a": "credit_card_account_id"}) == "repeated_ask"
    assert label(before={"a": "credit_card_account_id"}, asked={"a": "checking_account_id"}) is None


def test_the_summary_counts_failures_and_names_the_slowest_turns():
    rows = [
        {"conversation": "c1xxxxxxxx", "turn": 1, "ms": 40, "failure_class": None},
        {"conversation": "c1xxxxxxxx", "turn": 2, "ms": 900, "failure_class": "repeated_ask"},
        {"conversation": "c2xxxxxxxx", "turn": 1, "ms": 70, "failure_class": "repeated_ask"},
    ]
    summary = summarise(rows)
    assert summary["turns"] == 3 and summary["failures"] == {"repeated_ask": 2}
    assert summary["slowest_ms"][0] == {"conversation": "c1xxxxxx", "turn": 2, "ms": 900}


# --------------------------------------------------------------------- through the app

class Scores:
    async def score(self, message, decomposition):
        return [IntentScore.flat(i.id, 0.99) for i in decomposition.intents]


def test_the_api_records_each_turn_and_serves_the_trace_and_a_summary():
    task = next(t for t in load_tasks() if t["id"] == "task_078")
    index = KnowledgeIndex({"doc_x": Document("doc_x", "Internal: x", "lost card protocol")})
    hello = Interpretation(decomposition=Decomposition(intents=(Intent(id="g", kind=K.UNCLEAR_OR_GREETING, summary="hello"),)))

    def factory():
        return build_conversation(
            decomposer=ScriptedDecomposer([hello, hello]), scorer=Scores(), llm=FakeToolLLM([]), index=index,
            session=BankingSession.from_task(task),
        )

    main.app.state.factory = factory
    main.app.state.conversations = {}
    main.app.state.recorder = Recorder(path=None)
    client = TestClient(main.app)
    client.post("/conversations/c1/messages", json={"message_id": "m1", "message": "hello my pin is 7351"})
    client.post("/conversations/c1/messages", json={"message_id": "m2", "message": "hello again"})
    trace = client.get("/debug/conversations/c1/trace").json()
    assert [r["turn"] for r in trace] == [1, 2] and trace[0]["decisions"] and trace[0]["ms"] >= 0
    assert "7351" not in json.dumps(trace), "the trace never carries what the customer said"
    assert client.get("/debug/conversations/nope/trace").json() == []
    assert client.get("/debug/summary").json()["turns"] == 2


def test_a_turn_is_recorded_with_its_decisions_and_never_its_words(tmp_path):
    path = tmp_path / "turns.jsonl"
    recorder = Recorder(path)

    class Reply:
        text = "Done: it worked. my pin is 7351"
        questions = ()
        escalated_intent_ids = ()
        failed_intent_ids = ()
        choices = ()

    from orchestrator.decision import Action, IntentDecision
    from orchestrator.executor import OutcomeStatus, RouteOutcome
    from orchestrator.trace import Trace, TraceEvent, TraceKind

    class Result:
        reply = Reply()
        decisions = (IntentDecision(intent_id="a", action=Action.RUN, reason="plan approved"),)
        outcomes = (RouteOutcome(intent_id="a", status=OutcomeStatus.DONE, text="secret amount $5,000"),)
        trace = Trace(events=(TraceEvent(kind=TraceKind.DECIDED, intent_id="a", detail="run", reason="plan approved"),))

    import time

    observe.begin()
    row = recorder.record("conv-1", "m1", Result(), [{"kind": "agent", "stop": "terminal"}], time.monotonic(), ["a"])
    assert row["reply_kind"] == "done" and row["failure_class"] is None and row["turn"] == 1
    written = path.read_text()
    assert "7351" not in written and "5,000" not in written, "no message text and no amounts"
    assert json.loads(written)["decisions"][0]["reason"] == "plan approved"
    assert "turn 1" in render_turn(row) and recorder.conversation("conv-1") == [row]
    assert row["reply_asks"] == [], "a reply with no question sends the chat no asks"
    assert recorder.conversation("other") == []


def test_a_question_held_back_behind_a_plan_is_not_counted_as_a_repeated_ask():
    import time

    from orchestrator.decision import Action, IntentDecision
    from orchestrator.executor import RouteOutcome
    from orchestrator.trace import Trace

    class Reply:
        text = "Done: it worked."
        escalated_intent_ids = ()
        failed_intent_ids = ()
        choices = ()

        def __init__(self, questions):
            self.questions = questions

    held = IntentDecision(intent_id="l", action=Action.ASK, reason="not sure enough to act yet", questions=(Question(intent_id="l", slot="clarification"),))

    class Result:
        decisions = (held,)
        outcomes = ()
        trace = Trace(events=())

        def __init__(self, shown):
            self.reply = Reply(shown)

    recorder = Recorder(path=None)
    observe.begin()
    for turn in (1, 2):  # the same intent is asking both turns, but the customer is shown a plan, not its question
        row = recorder.record("c", f"m{turn}", Result(()), [], time.monotonic(), ["l"])
        assert row["failure_class"] is None, turn
    shown = Question(intent_id="l", slot="clarification", text="To be sure?")
    first = recorder.record("c", "m3", Result((shown,)), [], time.monotonic(), ["l"])
    second = recorder.record("c", "m4", Result((shown,)), [], time.monotonic(), ["l"])
    assert first["failure_class"] is None and second["failure_class"] == "repeated_ask", "shown twice in a row is a repeat"
