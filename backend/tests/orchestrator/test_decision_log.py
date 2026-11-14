"""Every decision is logged with the numbers behind it and what the old gate would have done; nothing sensitive."""

import json

from scenario_kit import GOOD_CLAIMS, I, S, World, interp

from orchestrator.decision_log import DecisionLog, JsonlLog, MemoryLog, legacy_action
from orchestrator.scoring import IntentScore
from orchestrator.types import IntentKind as K


def world(log, scripts, scores):
    w = World(scripts, scores)
    w.orch._decision_log = log
    return w


def test_a_decision_and_its_outcome_are_recorded_with_the_shadow_of_the_old_gate():
    sink = MemoryLog()
    w = world(
        DecisionLog(sink),
        [interp(I("bal", K.ACCOUNT_BALANCE_OR_HISTORY, {"card_id": "c1"}), I("fee", K.FEE_OR_INTEREST_ADJUSTMENT), claims=GOOD_CLAIMS)],
        [[S("bal", 0.75), S("fee", 0.40)]],
    )
    w.turn("check my balance and fix my fees")
    decisions = {r["intent"]: r for r in sink.records if r["type"] == "decision"}
    # A read at 0.75 now runs (read bar 0.70); the old gate asked. A 0.40 request is asked about, not escalated.
    assert decisions["bal"]["action"] == "run" and decisions["bal"]["shadow_action"] == "ask" and decisions["bal"]["disagrees"]
    assert decisions["fee"]["action"] == "ask" and decisions["fee"]["shadow_action"] == "escalate"
    assert decisions["bal"]["risk"] == "read" and decisions["fee"]["risk"] == "irreversible"
    assert {"type": "outcome", "turn": 1, "intent": "bal", "kind": "ACCOUNT_BALANCE_OR_HISTORY", "outcome": "completed"} in sink.records


def test_nothing_from_the_customers_words_or_details_is_written():
    sink = MemoryLog()
    secret = {"card_id": "dbc_secret_card", "new_pin": "4821"}
    w = world(DecisionLog(sink), [interp(I("x", K.FREEZE_CARD, secret), claims=GOOD_CLAIMS)], [[S("x", 0.99)]])
    w.turn("my pin is 4821 and my card is 4111 1111 1111 1111")
    text = json.dumps(sink.records)
    assert "4821" not in text and "4111" not in text and "dbc_secret_card" not in text and "summary of x" not in text


def test_a_scorer_failure_is_logged_by_type_only():
    sink = MemoryLog()
    log = DecisionLog(sink)
    log.decision(I("a", K.LOOKUP_INFO), IntentScore.unscored("a", "scorer failed: TimeoutError"), _ask("a"), asks_so_far=0)
    assert sink.records[0]["scorer_error"] == "TimeoutError"


def _ask(intent_id):
    from orchestrator.decision import Action, IntentDecision
    from orchestrator.types import Question

    return IntentDecision(intent_id=intent_id, action=Action.ASK, reason="r", questions=(Question(intent_id=intent_id, slot="clarification"),))


def test_the_file_log_writes_one_json_line_per_record(tmp_path):
    path = tmp_path / "logs" / "decisions.jsonl"
    log = DecisionLog(JsonlLog(path))
    log.outcome("a", "LOOKUP_INFO", "completed")
    log.outcome("b", "LOOKUP_INFO", "escalated")
    assert [json.loads(line)["outcome"] for line in path.read_text().splitlines()] == ["completed", "escalated"]


def test_legacy_action_is_the_old_single_gate():
    i = I("a", K.LOOKUP_INFO)
    assert [legacy_action(i, S("a", c)) for c in (0.9, 0.7, 0.5)] == ["run", "ask", "escalate"]
