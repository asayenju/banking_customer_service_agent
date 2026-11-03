import json

import pytest

from agent.pilot import LIVE_BAR, Gate, Verdict, decide, pilot_tasks, render_report, run_pilot_tier1
from agent.tier1 import Tier1Report, Tier1Result
from banking_tools.manifest import build_manifest, load_tasks
from knowledge.index import Document, KnowledgeIndex

INDEX = KnowledgeIndex({"doc_bank_accounts_x": Document("doc_bank_accounts_x", "t", "x")})
TASKS = load_tasks()


def ok(task_id, **kw):
    return Tier1Result(task_id, True, True, (), ("DISPUTE_TRANSACTION: x",), **kw)


def bad(task_id, **kw):
    return Tier1Result(task_id, False, False, (), (), error="did not plan", **kw)


def report(*results):
    return Tier1Report(tuple(results))


def live(*outcomes):
    return [{"task_id": f"task_{n:03d}", "passed": p, "note": ""} for n, p in outcomes]


def ten_of_twelve_live():
    return live(*[(n, n > 2) for n in range(1, 13)])


USABLE = tuple(ok(f"task_{n:03d}") for n in range(1, 13))


# ------------------------------------------------------------------ the rule

def test_all_green_offline_with_no_live_run_is_not_yet_never_go():
    gate = decide(report(*USABLE), None)
    assert gate.verdict is Verdict.NOT_YET and gate.live_pass_rate is None
    assert any("live run has not been done" in r for r in gate.reasons)
    assert decide(report(*USABLE), []).verdict is Verdict.NOT_YET, "an empty live result is no result"


def test_a_failing_offline_task_is_no_go_whatever_the_live_run_says():
    gate = decide(report(*USABLE[:-1], bad("task_012")), live(*[(n, True) for n in range(1, 13)]))
    assert gate.verdict is Verdict.NO_GO and "task_012" in " ".join(gate.reasons)
    assert gate.live_pass_rate is None, "the live result is not even consulted"


def test_a_write_outside_the_approved_plan_is_no_go_even_when_everything_passes():
    gate = decide(report(*USABLE[:-1], ok("task_012", bypasses=1)), live(*[(n, True) for n in range(1, 13)]))
    assert gate.verdict is Verdict.NO_GO and gate.bypasses == 1 and "outside the approved plan" in " ".join(gate.reasons)


@pytest.mark.parametrize("passed, verdict", [(12, Verdict.GO), (10, Verdict.GO), (9, Verdict.NO_GO), (0, Verdict.NO_GO)])
def test_the_live_bar_is_eighty_percent_of_the_usable_tasks(passed, verdict):
    outcomes = live(*[(n, n <= passed) for n in range(1, 13)])
    gate = decide(report(*USABLE), outcomes)
    assert gate.verdict is verdict and gate.live_pass_rate == pytest.approx(passed / 12)
    assert LIVE_BAR == 0.8


def test_ten_of_twelve_is_about_eighty_three_percent_and_clears_the_bar_but_nine_does_not():
    assert decide(report(*USABLE), live(*[(n, n <= 10) for n in range(1, 13)])).verdict is Verdict.GO
    assert decide(report(*USABLE), live(*[(n, n <= 9) for n in range(1, 13)])).verdict is Verdict.NO_GO


def test_a_live_run_that_covered_only_some_tasks_cannot_clear_the_bar():
    gate = decide(report(*USABLE), live(*[(n, True) for n in range(1, 6)]))
    assert gate.verdict is Verdict.NOT_YET and gate.live_tasks == 5
    assert "5 of 12" in " ".join(gate.reasons)


def test_tasks_skipped_for_the_cost_cap_do_not_count_as_passes_or_failures():
    outcomes = live(*[(n, True) for n in range(1, 10)]) + [{"task_id": f"task_0{n}", "passed": False, "note": "skipped: cost cap reached"} for n in (10, 11, 12)]
    gate = decide(report(*USABLE), outcomes)
    assert gate.verdict is Verdict.NOT_YET and gate.live_tasks == 9


def test_excluded_tasks_are_neither_passes_nor_failures():
    excluded = Tier1Result("task_083", False, False, (), (), excluded="its gold is invalid")
    gate = decide(report(excluded, *USABLE), None)
    assert gate.tier1_usable == 12 and gate.tier1_passed == 12 and gate.verdict is Verdict.NOT_YET


def test_with_nothing_usable_it_is_no_go():
    assert decide(report(), None).verdict is Verdict.NO_GO
    assert decide(report(Tier1Result("task_083", False, False, (), (), excluded="x")), None).verdict is Verdict.NO_GO


# ------------------------------------------------------------------ the real pilot, offline

def test_the_pilot_tasks_are_the_thirteen_wholly_debit_ones():
    ids = [str(t["id"]) for t in pilot_tasks(TASKS)]
    assert len(ids) == 13 and "task_083" in ids and "task_077" not in ids


def test_the_real_pilot_is_green_offline_with_no_bypass_and_the_verdict_waits_for_the_live_run():
    result = run_pilot_tier1(INDEX)
    usable = [r for r in result.results if not r.excluded]
    assert len(usable) == 12 and all(r.passed and r.strict_equal for r in usable)
    assert sum(r.bypasses for r in result.results) == 0
    assert all(r.seconds >= 0 for r in result.results)
    gate = decide(result, None)
    assert gate.verdict is Verdict.NOT_YET
    text = render_report(result, gate)
    assert "VERDICT: NOT YET" in text and "task_083: EXCLUDED" in text and "seconds per task" in text and "tokens" in text


def test_the_report_names_the_verdict_and_every_reason():
    gate = decide(report(*USABLE), live(*[(n, True) for n in range(1, 13)]))
    text = render_report(report(*USABLE), gate)
    assert "VERDICT: GO" in text and "live: 12/12 semantic pass@1 = 100%" in text


def test_a_saved_live_result_file_can_be_read_back(tmp_path):
    path = tmp_path / "live.json"
    path.write_text(json.dumps(live(*[(n, True) for n in range(1, 13)])))
    gate = decide(report(*USABLE), json.loads(path.read_text()))
    assert gate.verdict is Verdict.GO
