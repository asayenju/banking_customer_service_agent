"""The calibration report reads the decision log offline and refuses to recommend on too little data."""

import json
import socket

import pytest

from agent import calibration


def decision(turn, intent, kind=0.9, risk="read", action="run", disagrees=False):
    return {"type": "decision", "turn": turn, "intent": intent, "kind": "LOOKUP_INFO", "risk": risk, "kind_confidence": kind,
            "slots_confidence": 1.0, "scope_confidence": 1.0, "scorer_error": None, "action": action, "reason": "r",
            "asks_so_far": 0, "shadow_action": "run", "disagrees": disagrees, "signals": {}}


def outcome(turn, intent, what):
    return {"type": "outcome", "turn": turn, "intent": intent, "kind": "LOOKUP_INFO", "outcome": what}


def test_a_small_log_is_reported_but_not_trusted():
    text = calibration.report([decision(1, "a"), outcome(1, "a", "completed")])
    assert "Too few decisions" in text and "decisions: 1" in text


def test_separation_and_the_sweep_count_what_would_run():
    rows = []
    for n in range(20):
        rows += [decision(n, "ok", 0.9), outcome(n, "ok", "completed"), decision(n, "bad", 0.45, action="ask"), outcome(n, "bad", "escalated")]
    text = calibration.report(rows)
    assert "separation (completed vs handed over): 1.00" in text
    assert "proceed at 0.50: 20 would run, 20 of them completed, 20 asked about" in text
    assert "Too few" not in text


def test_the_report_makes_no_network_call(monkeypatch, tmp_path):
    def blocked(*a, **k):
        raise AssertionError("the calibration report must not use the network")

    monkeypatch.setattr(socket, "socket", blocked)
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps(decision(1, "a")) + "\n")
    assert "decisions: 1" in calibration.report(calibration.load([path]))


def test_auroc_edges():
    assert calibration.auroc([0.9], [0.1]) == 1.0 and calibration.auroc([0.5], [0.5]) == 0.5
    assert calibration.auroc([], [0.1]) is None
