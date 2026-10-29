import asyncio
import json

import pytest

from orchestrator.decomposer import Interpretation
from orchestrator.evaluation.messages import build_cases, load_cases, save_cases
from orchestrator.evaluation.metrics import pass_at_k, pass_hat_k
from orchestrator.evaluation.runner import render, run_eval, summarize
from orchestrator.evaluation.tasks import (
    EvalCase,
    expected_kinds,
    load_tasks,
    select_tasks,
)
from orchestrator.llm import FakeLLMClient
from orchestrator.types import Decomposition, Intent, IntentKind

K = IntentKind


def action(name, requestor="assistant", **arguments):
    return {"name": name, "arguments": arguments, "requestor": requestor}


def task(*actions, id="t"):
    return {
        "id": id,
        "evaluation_criteria": {"actions": list(actions)},
        "user_scenario": {"instructions": "You are a customer who wants things."},
    }


def test_pass_at_k_and_pass_hat_k_for_three_trials():
    assert [pass_at_k(3, 1, k) for k in (1, 2, 3)] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert [pass_hat_k(3, 2, k) for k in (1, 2, 3)] == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert pass_at_k(3, 0, 3) == 0.0 and pass_hat_k(3, 3, 3) == 1.0


def test_metrics_reject_impossible_inputs():
    with pytest.raises(ValueError):
        pass_at_k(3, 4, 1)
    with pytest.raises(ValueError):
        pass_hat_k(3, 1, 4)


def test_expected_kinds_unwraps_discoverable_tools_and_drops_reads_and_user_tools():
    t = task(
        action("unlock_discoverable_agent_tool", agent_tool_name="close_debit_card_4721"),
        action("call_discoverable_agent_tool", agent_tool_name="close_debit_card_4721"),
        action("call_discoverable_agent_tool", agent_tool_name="get_pending_replacement_orders_5765"),
        action("log_verification"),
        action("transfer_to_human_agents"),
        action("apply_for_credit_card", requestor="user"),
    )
    assert expected_kinds(t) == {K.CLOSE_CARD, K.REQUEST_HUMAN}


def test_selection_is_fixed_sized_and_varied():
    tasks = load_tasks()
    first, second = select_tasks(tasks, 50), select_tasks(tasks, 50)
    assert [t["id"] for t in first] == [t["id"] for t in second]
    assert len(first) == len({t["id"] for t in first}) == 50
    assert all(expected_kinds(t) for t in first)
    assert len(set().union(*(expected_kinds(t) for t in first))) >= 12


class Stub:
    """Predicts a fixed list of kinds per call, cycling through ``script``."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    async def interpret(self, message, open_intents):
        kinds = self.script[self.calls % len(self.script)]
        self.calls += 1
        intents = tuple(Intent(id=f"i{n}", kind=k, summary="s") for n, k in enumerate(kinds))
        return Interpretation(decomposition=Decomposition(intents=intents))


def test_runner_scores_exact_match_and_reports_pass_at_k_and_pass_hat_k():
    cases = [
        EvalCase(task_id="a", message="m", stated=(K.FREEZE_CARD, K.CLOSE_CARD), expected=(K.FREEZE_CARD, K.CLOSE_CARD)),
        EvalCase(task_id="b", message="m", stated=(K.CLOSE_CARD,), expected=(K.CLOSE_CARD, K.FREEZE_CARD)),
    ]
    # Task a is predicted right, then right, then missing a kind (2 of 3 pass).
    # Task b is predicted with an extra write kind every time (0 of 3 pass).
    per_task = {
        "a": Stub([[K.CLOSE_CARD, K.FREEZE_CARD, K.LOOKUP_INFO], [K.FREEZE_CARD, K.CLOSE_CARD], [K.CLOSE_CARD]]),
        "b": Stub([[K.CLOSE_CARD, K.FREEZE_CARD]]),
    }

    class ByTask:
        def __init__(self):
            self.queue = ["a", "a", "a", "b", "b", "b"]

        def make(self):
            return per_task[self.queue.pop(0)]

    factory = ByTask()
    results = asyncio.run(run_eval(cases, factory.make, trials=3, concurrency=1))
    assert [r.passed for r in results] == [True, True, False, False, False, False]
    assert results[2].missed == (K.FREEZE_CARD,)
    assert results[3].extra == (K.FREEZE_CARD,)

    s = summarize(results, 3)
    assert s.tasks == 2
    assert s.pass_at_k == pytest.approx((1 / 3, 1 / 2, 1 / 2))
    assert s.pass_hat_k == pytest.approx((1 / 3, 1 / 6, 0.0))
    assert s.mean_trial_pass_rate == pytest.approx(1 / 3)
    assert s.policy_pass_rate == pytest.approx(5 / 6)
    text = render(s, results)
    assert "pass@2" in text and "pass^3" in text


def test_a_failed_decomposition_counts_as_a_failed_trial():
    case = EvalCase(task_id="a", message="m", stated=(K.CLOSE_CARD,), expected=(K.CLOSE_CARD,))
    [result] = asyncio.run(run_eval([case], lambda: Stub([[K.OTHER]]), trials=1))
    assert not result.passed and result.missed == (K.CLOSE_CARD,)


def test_building_and_reloading_the_frozen_set(tmp_path):
    client = FakeLLMClient([json.dumps({"message": "  Please close my debit card.  "})])
    labeler = FakeLLMClient([json.dumps({"kinds": ["CLOSE_CARD"]})])
    tasks = [task(action("call_discoverable_agent_tool", agent_tool_name="close_debit_card_4721"), id="task_9")]
    cases = asyncio.run(build_cases(client, labeler, tasks))
    assert cases == [EvalCase(task_id="task_9", message="Please close my debit card.", stated=(K.CLOSE_CARD,), expected=(K.CLOSE_CARD,))]
    assert "Please close my debit card." in labeler.calls[0]["messages"][1]["content"]
    assert "who wants things" in client.calls[0]["messages"][1]["content"]
    save_cases(cases, tmp_path / "set.json")
    assert load_cases(tmp_path / "set.json") == cases


def test_auroc_measures_how_well_low_scores_flag_wrong_intents():
    from orchestrator.evaluation.jev_check import auroc

    assert auroc([0.1, 0.2], [0.9, 0.95]) == 1.0
    assert auroc([0.9], [0.9]) == 0.5
    assert auroc([0.9, 0.1], [0.5]) == 0.5
    assert auroc([], [0.5]) != auroc([], [0.5])  # nan: no wrong intents, no signal
