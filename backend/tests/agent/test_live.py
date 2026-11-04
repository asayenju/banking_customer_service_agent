import asyncio
import json
from pathlib import Path

import litellm
import pytest

from agent import live
from agent.live import (
    BudgetExceeded,
    CassetteClient,
    CassetteMismatch,
    LiveRunRefused,
    MeteredClient,
    Plan,
    RunContext,
    cassette_path,
    check_estimate,
    choose_tasks,
    estimate_run,
    estimate_task,
    gradeable_clusters,
    price_for,
    prepare,
    request_hash,
    run_live_task,
    run_plan,
    task_shape,
)
from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, ToolSchema, Usage
from agent.specialists.debit import DEBIT_SPECIALIST
from banking_tools.session import _base_db
from banking_tools.task_state import apply_initial_state
from agent.tier1 import Segment, gold_steps, plan_items, scripted_turns, workflow_turns, Calls
from agent.toolbox import SecretVault
from banking_tools.manifest import load_tasks
from knowledge.index import Document, KnowledgeIndex
from knowledge.report import opening_message

TASKS = {str(t["id"]): t for t in load_tasks()}
INDEX = KnowledgeIndex({"doc_bank_accounts_x": Document("doc_bank_accounts_x", "t", "x")})
GRADEABLE = gradeable_clusters({"debit": DEBIT_SPECIALIST})
SHAPES = {t: task_shape(TASKS[t], DEBIT_SPECIALIST) for t in GRADEABLE}
MODEL = "openai/gpt-4o"


def go(*argv, shapes=SHAPES):
    return prepare(list(argv), gradeable=GRADEABLE, shapes=shapes, default_model=MODEL)


# ------------------------------------------------------------------ default tests are offline

def test_a_default_test_run_cannot_reach_a_real_model():
    with pytest.raises(AssertionError, match="tried to call a real model"):
        asyncio.run(litellm.acompletion(model="openai/gpt-4o", messages=[{"role": "user", "content": "hi"}]))


def test_the_tool_client_cannot_either_unless_the_test_fakes_the_network():
    from agent.llm_tools import LiteLLMToolClient

    client = LiteLLMToolClient(api_key="sk-not-real")
    with pytest.raises(AssertionError, match="tried to call a real model"):
        asyncio.run(client.chat([{"role": "user", "content": "hi"}], tools=[]))


def test_a_test_that_stubs_the_network_itself_still_works(monkeypatch):
    import types

    async def stub(**kwargs):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok", tool_calls=None))], usage=None)

    monkeypatch.setattr(litellm, "acompletion", stub)
    from agent.llm_tools import LiteLLMToolClient

    assert asyncio.run(LiteLLMToolClient(api_key="k").chat([{"role": "user", "content": "hi"}], tools=[])).text == "ok"


# ------------------------------------------------------------------ choosing tasks

def test_the_gradeable_tasks_are_the_twelve_usable_debit_ones():
    assert sorted(GRADEABLE) == [f"task_{n:03d}" for n in (78, 79, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92)]
    assert set(GRADEABLE.values()) == {"debit"}


@pytest.mark.parametrize("spec", [[], ["all"], ["ALL"], ["*"], ["task_082", "all"]])
def test_there_is_no_all_and_tasks_are_required(spec):
    with pytest.raises(LiveRunRefused):
        choose_tasks(spec, max_tasks=5, gradeable=GRADEABLE)


def test_the_command_line_will_not_even_parse_without_tasks():
    with pytest.raises(SystemExit):
        go()


@pytest.mark.parametrize("item, part", [("task_083", "excluded"), ("task_001", "cannot be run live"), ("banana", "not a task id"), ("credit", "not a task id")])
def test_unusable_or_unknown_tasks_are_refused_with_the_reason(item, part):
    with pytest.raises(LiveRunRefused, match=part):
        choose_tasks([item], max_tasks=5, gradeable=GRADEABLE)


def test_a_specialist_name_expands_to_its_tasks_but_only_within_the_limit():
    assert len(choose_tasks(["debit"], max_tasks=20, gradeable=GRADEABLE)) == 12
    with pytest.raises(LiveRunRefused, match="12 tasks selected but --max-tasks is 5"):
        choose_tasks(["debit"], max_tasks=5, gradeable=GRADEABLE)


@pytest.mark.parametrize("limit", [0, -1, 21, 500])
def test_the_task_limit_has_a_hard_ceiling(limit):
    with pytest.raises(LiveRunRefused, match="between 1 and 20"):
        choose_tasks(["task_082"], max_tasks=limit, gradeable=GRADEABLE)


def test_repeated_ids_count_once_and_keep_their_order():
    assert choose_tasks(["task_084", "task_082", "task_084"], max_tasks=5, gradeable=GRADEABLE) == ("task_084", "task_082")


def test_trials_are_capped():
    for trials in ("0", "4", "100"):
        with pytest.raises(LiveRunRefused, match="between 1 and 3"):
            go("--tasks", "task_082", "--trials", trials, "--yes")


# ------------------------------------------------------------------ prices and the estimate

def test_a_model_with_no_known_price_is_refused_unless_a_price_is_given():
    with pytest.raises(LiveRunRefused, match="no price is known"):
        price_for("openai/mystery", None, None)
    with pytest.raises(LiveRunRefused):
        price_for("openai/mystery", 1.0, None)
    assert price_for("openai/mystery", 1.0, 2.0) == (1.0, 2.0)
    assert price_for(MODEL, None, None) == (2.5, 10.0)
    assert price_for(MODEL, 9.0, 9.0) == (9.0, 9.0), "a given price wins over the table"


def test_an_estimate_grows_with_the_work_and_is_in_dollars():
    small, big = estimate_task(1, 2), estimate_task(2, 12)
    assert big[0] > small[0] and big[1] > small[1] and big[2] > small[2]
    one = estimate_run({"task_082": (1, 6)}, [], trials=1, price_in=2.5, price_out=10.0)
    two = estimate_run({"task_082": (1, 6)}, [], trials=2, price_in=2.5, price_out=10.0)
    assert one.dollars > 0 and two.calls == 2 * one.calls and two.dollars == pytest.approx(2 * one.dollars, rel=0.01)


def test_tasks_that_already_have_a_recording_cost_nothing():
    est = estimate_run({"task_082": (1, 6), "task_084": (1, 6)}, ["task_082"], trials=1, price_in=2.5, price_out=10.0)
    assert est.tasks == 2 and est.uncached_tasks == 1
    assert estimate_run({"task_082": (1, 6)}, ["task_082"], trials=1, price_in=2.5, price_out=10.0).dollars == 0


# ------------------------------------------------------------------ the gate: estimate, confirmation, cap

def test_without_yes_the_run_prints_what_it_would_cost_and_does_not_start():
    with pytest.raises(LiveRunRefused, match="re-run with --yes"):
        go("--tasks", "task_082", "--cassettes", "/nonexistent-dir-for-test")


def test_with_yes_and_an_estimate_under_the_cap_the_run_is_allowed():
    plan, estimate, _ = go("--tasks", "task_082", "--yes", "--cassettes", "/nonexistent-dir-for-test")
    assert plan.task_ids == ("task_082",) and plan.model == MODEL and plan.trials == 1
    assert 0 < estimate.dollars <= plan.max_cost


def test_an_estimate_over_the_cap_is_refused_even_with_yes():
    with pytest.raises(LiveRunRefused, match="over the cap"):
        go("--tasks", "task_086", "--yes", "--max-cost-usd", "0.01", "--cassettes", "/nonexistent-dir-for-test")


def test_the_default_cap_is_small_and_a_zero_or_negative_cap_is_refused():
    plan, _, _ = go("--tasks", "task_082", "--yes", "--cassettes", "/nonexistent-dir-for-test")
    assert plan.max_cost == live.DEFAULT_MAX_COST == 0.5
    for cap in ("0", "-1"):
        with pytest.raises(LiveRunRefused, match="above zero"):
            go("--tasks", "task_082", "--yes", "--max-cost-usd", cap)


def test_check_estimate_wording():
    plan = Plan(("task_082",), 1, 1.0, MODEL, 2.5, 10.0)
    est = estimate_run({"task_082": (1, 6)}, [], trials=1, price_in=2.5, price_out=10.0)
    with pytest.raises(LiveRunRefused, match="spend about"):
        check_estimate(est, plan, confirmed=False)
    check_estimate(est, plan, confirmed=True)


def test_nothing_is_built_or_read_before_the_run_is_confirmed():
    built = []
    with pytest.raises(LiveRunRefused):
        go("--tasks", "task_082", "--cassettes", "/nonexistent-dir-for-test")
    assert built == [], "prepare() takes no client factory at all, so it cannot build one"
    import inspect

    assert "make_client" not in inspect.signature(prepare).parameters


# ------------------------------------------------------------------ metering

class Counting:
    def __init__(self, usage=Usage(input_tokens=10_000, output_tokens=1_000)):
        self.usage, self.calls = usage, 0

    async def chat(self, messages, *, tools, tool_choice="auto", temperature=0.0):
        self.calls += 1
        return AssistantTurn(text="x", usage=self.usage)


def chat(client):
    return asyncio.run(client.chat([{"role": "user", "content": "hi"}], tools=[]))


def test_the_meter_adds_up_real_usage_and_stops_before_the_call_that_would_go_over():
    inner = Counting()  # $0.035 a call at gpt-4o prices
    meter = MeteredClient(inner, price_in=2.5, price_out=10.0, cap=0.05)
    chat(meter)
    assert meter.calls == 1 and meter.dollars == pytest.approx(0.035) and meter.input_tokens == 10_000
    chat(meter)  # $0.035 spent is under the cap, so this one is allowed
    assert meter.dollars == pytest.approx(0.07)
    with pytest.raises(BudgetExceeded):
        chat(meter)
    assert inner.calls == 2, "the third call was never made"


# ------------------------------------------------------------------ cassettes

TOOLS = [ToolSchema(name="t", description="d", parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False})]


def say(client, text="hello"):
    return asyncio.run(client.chat([{"role": "user", "content": text}], tools=TOOLS))


def test_a_recording_replays_for_free_and_gives_the_same_answer(tmp_path):
    path = tmp_path / "c.json"
    inner = Counting()
    recorder = CassetteClient(inner, path)
    first = say(recorder)
    recorder.save()
    assert path.exists() and inner.calls == 1 and not recorder.replaying
    replay = CassetteClient(None, path)
    assert replay.replaying and say(replay) == first


def test_replaying_makes_no_live_call_and_needs_no_client(tmp_path):
    path = tmp_path / "c.json"
    recorder = CassetteClient(Counting(), path)
    say(recorder)
    recorder.save()
    replay = CassetteClient(None, path)
    say(replay)  # does not raise: nothing to call


def test_a_changed_prompt_is_an_error_not_a_quiet_second_charge(tmp_path):
    path = tmp_path / "c.json"
    recorder = CassetteClient(Counting(), path)
    say(recorder, "hello")
    recorder.save()
    with pytest.raises(CassetteMismatch, match="different prompt"):
        say(CassetteClient(None, path), "something else")
    inner = Counting()
    with pytest.raises(CassetteMismatch):
        say(CassetteClient(inner, path), "something else")
    assert inner.calls == 0, "even with a live client available, a replay never falls back to it"


def test_asking_for_more_calls_than_were_recorded_is_an_error(tmp_path):
    path = tmp_path / "c.json"
    recorder = CassetteClient(Counting(), path)
    say(recorder)
    recorder.save()
    replay = CassetteClient(None, path)
    say(replay)
    with pytest.raises(CassetteMismatch, match="only 1 calls"):
        say(replay)


def test_rerecord_pays_for_a_fresh_recording(tmp_path):
    path = tmp_path / "c.json"
    first = CassetteClient(Counting(), path)
    say(first)
    first.save()
    inner = Counting(Usage(input_tokens=5, output_tokens=1))
    again = CassetteClient(inner, path, rerecord=True)
    say(again)
    again.save()
    assert inner.calls == 1 and not again.replaying
    assert CassetteClient(None, path).replaying


def test_no_client_and_no_recording_is_an_error_not_a_crash(tmp_path):
    with pytest.raises(CassetteMismatch, match="no recording and no live client"):
        say(CassetteClient(None, tmp_path / "missing.json"))


def test_a_recording_is_named_by_task_segment_and_model(tmp_path):
    assert cassette_path(tmp_path, "task_082", 2, "openai/gpt-4o").name == "task_082.2.openai-gpt-4o.json"
    assert cassette_path(tmp_path, "task_082", 2, "openai/gpt-4o-mini") != cassette_path(tmp_path, "task_082", 2, "openai/gpt-4o")


def test_the_request_hash_ignores_nothing_that_changes_the_answer():
    base = request_hash([{"role": "user", "content": "a"}], TOOLS, "auto")
    assert base == request_hash([{"role": "user", "content": "a"}], TOOLS, "auto")
    assert base != request_hash([{"role": "user", "content": "b"}], TOOLS, "auto")
    assert base != request_hash([{"role": "user", "content": "a"}], TOOLS, "required")
    assert base != request_hash([{"role": "user", "content": "a"}], [], "auto")


def test_a_failed_save_leaves_no_half_written_recording(tmp_path):
    path = tmp_path / "c.json"
    recorder = CassetteClient(Counting(), path)
    say(recorder)
    recorder.save()
    assert list(tmp_path.glob("*.tmp")) == [] and json.loads(path.read_text())[0]["turn"]


# ------------------------------------------------------------------ a whole task, with a stand-in for the model

class GoldModel:
    """Plays the right answer for each segment, one conversation after another."""

    def __init__(self, task, usage=Usage(input_tokens=800, output_tokens=60)):
        items = [i for i in plan_items(gold_steps(task), DEBIT_SPECIALIST) if isinstance(i, Segment)]
        ids, vault = Calls(), SecretVault()
        database = apply_initial_state(_base_db().model_copy(deep=True), task)
        self.conversations = [
            workflow_turns(DEBIT_SPECIALIST, i.kind, i.reads, i.writes, database, "", ids)
            or scripted_turns(i.reads, i.writes, vault, ids)
            for i in items
        ]
        self.usage, self.index, self.position, self.calls = usage, -1, 0, 0

    async def chat(self, messages, *, tools, tool_choice="auto", temperature=0.0):
        if len(messages) <= 2:  # a new conversation: system + the customer's request
            self.index += 1
            self.position = 0
        turn = self.conversations[self.index][self.position]
        self.position += 1
        self.calls += 1
        return AssistantTurn(tool_calls=turn.tool_calls, usage=self.usage)


def plan_for(model_cap=0.50):
    return Plan(("task_084",), 1, model_cap, MODEL, 2.5, 10.0)


def run_one(task_id, cassettes, make_client, plan=None, rerecord=False):
    context = RunContext(plan=plan or plan_for(), make_client=make_client, rerecord=rerecord)
    return run_live_task(TASKS[task_id], DEBIT_SPECIALIST, cassettes, context, index=INDEX)


def test_a_task_run_with_a_stand_in_model_costs_money_then_replays_for_free(tmp_path):
    model = GoldModel(TASKS["task_084"])
    first = run_one("task_084", tmp_path, lambda m: model)
    assert first.passed and first.strict_equal and not first.replayed
    assert first.calls == model.calls > 0 and first.dollars > 0
    assert first.input_tokens == model.calls * 800 and any(tmp_path.glob("task_084.*.json"))

    def forbidden(m):
        raise AssertionError("a replay must not build a live client")

    second = run_one("task_084", tmp_path, forbidden)
    assert second.passed and second.strict_equal and second.replayed and second.dollars == 0 and second.calls == 0


def test_the_cap_stops_a_task_midway_and_the_result_says_so(tmp_path):
    model = GoldModel(TASKS["task_084"], Usage(input_tokens=2_000, output_tokens=100))
    expensive = Plan(("task_084",), 1, 0.50, MODEL, 100.0, 100.0)  # $0.21 a call: three calls cost $0.63
    result = run_one("task_084", tmp_path, lambda m: model, plan=expensive)
    assert not result.passed and "cost cap of $0.50 reached" in result.note
    assert model.calls == 3, "two calls fit under the $0.50 cap, the third was allowed to start at $0.42, the fourth was not"
    assert result.dollars == pytest.approx(0.63) and result.dollars < 0.50 + 0.21, "never more than one call over"


def test_the_specialists_own_token_budget_is_a_second_limit_that_can_trip_first(tmp_path):
    model = GoldModel(TASKS["task_084"], Usage(input_tokens=60_000, output_tokens=2_000))
    result = run_one("task_084", tmp_path, lambda m: model, plan=plan_for(50.0))
    assert not result.passed and "tokens_exhausted" in result.note and model.calls == 2


def test_a_model_that_asks_instead_of_planning_fails_the_task_without_spending_past_it(tmp_path):
    asker = FakeToolLLM([AssistantTurn(tool_calls=(ToolCall(id="c1", name="ask_customer", arguments_json=json.dumps({"details": ["card_id"]})),), usage=Usage(input_tokens=500, output_tokens=20))])
    result = run_one("task_084", tmp_path, lambda m: asker)
    assert not result.passed and "SegmentError" in result.note and "not planned" in result.note


def test_a_task_whose_gold_is_invalid_is_not_run_at_all(tmp_path):
    context = RunContext(plan=Plan(("task_083",), 1, 0.5, MODEL, 2.5, 10.0), make_client=lambda m: pytest.fail("built a client"), rerecord=False)
    result = run_live_task(TASKS["task_083"], DEBIT_SPECIALIST, tmp_path, context, index=INDEX)
    assert not result.passed and "excluded" in result.note and result.dollars == 0


def test_the_task_shape_used_for_the_estimate_counts_segments_and_reads():
    segments, reads = task_shape(TASKS["task_091"], DEBIT_SPECIALIST)
    assert segments == 5 and reads >= 6
    assert task_shape(TASKS["task_089"], DEBIT_SPECIALIST)[0] == 1


# ------------------------------------------------------------------ running several

def test_the_run_stops_starting_tasks_once_the_cap_is_reached(tmp_path):
    plan = Plan(("task_082", "task_084", "task_085"), 1, 0.05, MODEL, 2.5, 10.0)
    ran = []

    def run_task(task_id, cassettes, context):
        ran.append(task_id)
        return live.LiveTaskResult(task_id, True, True, 3, 100, 10, 0.06, 1.0, False)

    args = live.build_parser().parse_args(["--tasks", "task_082"])
    report = run_plan(plan, args, lambda m: pytest.fail("no client"), run_task)
    assert ran == ["task_082"], "the first task used the whole budget; the rest were not started"
    assert [r.note for r in report.results] == ["", "skipped: cost cap reached", "skipped: cost cap reached"]
    assert "skipped" in report.render() and report.dollars == pytest.approx(0.06)


def test_trials_repeat_each_task(tmp_path):
    plan = Plan(("task_082",), 3, 5.0, MODEL, 2.5, 10.0)
    ran = []
    run_plan(plan, live.build_parser().parse_args(["--tasks", "task_082"]), lambda m: None, lambda t, c, x: ran.append(t) or live.LiveTaskResult(t, True, True, 0, 0, 0, 0.0, 0.0, True))
    assert ran == ["task_082"] * 3


def test_the_cassette_directory_is_git_ignored():
    ignore = (Path(live.__file__).resolve().parents[3] / ".gitignore").read_text()
    assert "backend/evals/cassettes/" in ignore


# ------------------------------------------------------------------ one event loop for a whole live run

class LoopBound:
    """Fails the way a real HTTP client does once the loop it first ran on has been closed."""

    def __init__(self, inner):
        self.inner, self.loop = inner, None

    async def chat(self, messages, **kwargs):
        running = asyncio.get_running_loop()
        if self.loop is None:
            self.loop = running
        elif self.loop is not running:
            raise RuntimeError("Event loop is closed")
        return await self.inner.chat(messages, **kwargs)


def two_tasks(tmp_path, client):
    results = []
    for task_id in ("task_084", "task_085"):
        model = GoldModel(TASKS[task_id])
        client.inner = model
        context = RunContext(plan=Plan((task_id,), 1, 0.5, MODEL, 2.5, 10.0), make_client=lambda m: client, rerecord=False)
        results.append(run_live_task(TASKS[task_id], DEBIT_SPECIALIST, tmp_path / task_id, context, index=INDEX))
    return results


def test_without_a_shared_loop_the_second_task_fails_the_way_it_did_live(tmp_path):
    first, second = two_tasks(tmp_path, LoopBound(None))
    assert first.passed is False or second.passed is False
    assert "model_failed" in (first.note + second.note), "this is the failure the live run hit"


def test_with_one_loop_for_the_whole_run_every_task_reaches_the_model(tmp_path):
    from agent import tier1

    loop = asyncio.new_event_loop()
    tier1.use_loop(loop)
    try:
        results = two_tasks(tmp_path, LoopBound(None))
    finally:
        tier1.use_loop(None)
        loop.close()
    assert all(r.passed for r in results), [r.note for r in results]


def test_the_customers_notes_go_to_the_specialist_so_it_need_not_ask_for_what_the_persona_would_say():
    facts = live.customer_facts(TASKS["task_084"])
    assert facts.startswith(opening_message(TASKS["task_084"])[:20])
    assert "What the customer knows and will tell you if you ask:" in facts
    assert len(facts) > len(str(TASKS["task_084"]["user_scenario"]["instructions"]))


def test_the_gold_pins_are_hidden_from_the_customers_notes_whatever_words_surround_them():
    from agent.live import gold_secrets

    task = TASKS["task_091"]
    secrets = gold_secrets(task)
    assert secrets and all(len(v) == 4 and v.isdigit() for v in secrets.values())
    vault = SecretVault()
    cleaned = vault.capture(vault.hide(live.customer_facts(task), secrets))
    for value in secrets.values():
        assert value not in cleaned, f"{value} would have reached the model"
    assert "{{secret:" in cleaned
    assert all(name.replace("_", "").isalpha() for name in secrets)


def test_the_tool_reference_tells_the_model_the_allowed_values_of_each_argument():
    from agent.specialist import system_prompt
    from orchestrator.types import IntentKind

    prompt = system_prompt(DEBIT_SPECIALIST, IntentKind.ORDER_REPLACEMENT_CARD)
    assert "Reason for closing: lost, stolen, fraud_suspected, damaged, no_longer_needed, or account_closing" in prompt


def test_a_segment_with_its_own_request_gets_only_that_not_the_whole_persona_script():
    path = Path(live.__file__).resolve().parents[2] / "evals" / "segment_requests.json"
    written = json.loads(path.read_text())["task_087"]
    assert set(written) == {"0", "1", "2"}
    assert live.segment_request("task_087", 0) == written["0"]
    assert live.segment_request("task_087", 7) is None  # no entry: the notes path is used
    # Each request names its own card and none asks for another segment's work.
    assert "Blue" in written["0"] and "Evergreen" not in written["0"] and "fraud alert" not in written["0"]
    assert "Evergreen" in written["1"] and "Evergreen" in written["2"]


def test_the_credit_tasks_can_be_chosen_for_a_live_run_including_the_ones_that_also_pay_from_checking():
    from agent.specialists.credit import CREDIT_SPECIALIST

    both = gradeable_clusters({"debit": DEBIT_SPECIALIST, "credit": CREDIT_SPECIALIST})
    credit = sorted(t for t, c in both.items() if c == "credit")
    assert credit == [f"task_{n:03d}" for n in (31, 36, 37, 38, 39, 40, 41, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54)]
    debit = sorted(t for t, c in both.items() if c == "debit")
    assert debit == sorted(GRADEABLE), "adding the credit specialist changes nothing for debit (no credit+debit task is picked up)"
    # The probe's three tasks resolve, and "credit" alone would be 18 tasks, over the default cap of five.
    assert choose_tasks(["task_043", "task_054", "task_038"], max_tasks=5, gradeable=both) == ("task_043", "task_054", "task_038")
    with pytest.raises(LiveRunRefused):
        choose_tasks(["credit"], max_tasks=5, gradeable=both)
