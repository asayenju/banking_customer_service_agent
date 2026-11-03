import asyncio
import json

import pytest

from agent.llm_tools import AssistantTurn, FakeToolLLM, ToolCall, Usage
from agent.loop import TERMINALS, Budget, StopReason, run_agent
from agent.toolbox import ScopedToolBox, SecretVault
from banking_tools import BankingSession, Fidelity
from orchestrator.context import SessionRequest, scoped

CARD = "dbc_lj82d4f1a9_bluest"
USER = "lj82d4f1a9"
ACCOUNT = "chk_lj82d4f1a9"
READ = "get_all_user_accounts_by_user_id_3847"
FREEZE = "freeze_debit_card_3892"
UNFREEZE = "unfreeze_debit_card_3893"
BLOCK = "set_debit_card_recurring_block_7382"
ALLOWED = {READ, FREEZE, UNFREEZE, BLOCK, "reset_debit_card_pin_6284"}


def tool(name, **arguments):
    return ToolCall(id=f"c-{name}", name=name, arguments_json=json.dumps(arguments))


def turn(*calls, tokens=100):
    return AssistantTurn(tool_calls=tuple(calls), usage=Usage(input_tokens=tokens, output_tokens=10))


def finish(message="all done"):
    return turn(tool("finish", message=message))


def setup(script, vault=None):
    shared = BankingSession(fidelity=Fidelity.TAU2)
    box = ScopedToolBox(scoped(shared)(SessionRequest("i")), user_id=USER, allowed_tools=ALLOWED, vault=vault)
    return FakeToolLLM(script), box, shared


def run(script, *, budget=Budget(), deadline=None, clock=None, vault=None, terminals=("finish", "ask_customer", "escalate")):
    llm, box, shared = setup(script, vault)
    kwargs = {"clock": clock} if clock else {}
    result = asyncio.run(
        run_agent(llm, box, system="You help.", request="freeze my card", budget=budget, deadline=deadline, terminals=terminals, **kwargs)
    )
    return result, llm, shared


# ------------------------------------------------------------------ ending well

def test_a_run_reads_then_writes_then_finishes_and_says_so():
    result, llm, shared = run(
        [turn(tool(READ)), turn(tool(FREEZE, card_id=CARD)), finish("Your card is frozen.")]
    )
    assert result.finished and result.reason is StopReason.TERMINAL
    assert result.terminal == "finish" and result.arguments == {"message": "Your card is frozen."}
    assert (result.steps, result.tool_calls, result.writes, result.errors) == (3, 2, 1, 0)
    assert [(r.name, r.ok, r.write) for r in result.log] == [(READ, True, False), (FREEZE, True, True)]
    assert result.usage == Usage(input_tokens=300, output_tokens=30)
    assert [e.tool for e in shared.audit_log] == [FREEZE]


def test_every_turn_requires_a_tool_call_and_offers_the_terminals_alongside_the_catalog_tools():
    _, llm, _ = run([finish()])
    assert llm.calls[0]["tool_choice"] == "required"
    offered = llm.calls[0]["tools"]
    assert {"finish", "ask_customer", "escalate"} <= set(offered) and READ in offered
    assert "propose_plan" not in offered


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("ask_customer", {"details": ["card_id"]}),
        ("escalate", {"reason": "customer asked"}),
        ("propose_plan", {"steps": [{"tool": FREEZE, "arguments_json": "{}", "description": "Freeze"}]}),
    ],
)
def test_each_terminal_ends_the_run_with_its_arguments(name, arguments):
    result, _, shared = run([turn(tool(name, **arguments))], terminals=("finish", "ask_customer", "escalate", "propose_plan"))
    assert result.finished and result.terminal == name and result.arguments == arguments
    assert shared.audit_log == ()


def test_a_terminal_with_the_wrong_arguments_is_a_counted_error_not_the_end():
    result, _, _ = run([turn(tool("finish", wrong="x")), finish()])
    assert result.finished and result.errors == 1 and result.steps == 2


def test_the_schemas_of_the_terminals_are_strict():
    from agent.toolbox import is_strict_valid

    for schema in TERMINALS.values():
        assert is_strict_valid(schema.parameters), schema.name


def test_an_unknown_terminal_or_a_name_clash_is_a_setup_error():
    with pytest.raises(ValueError, match="unknown terminal"):
        run([finish()], terminals=("finish", "made_up"))


# ------------------------------------------------------------------ budgets

def test_the_step_budget_stops_a_model_that_never_finishes():
    result, llm, _ = run([turn(tool(READ)) for _ in range(20)], budget=Budget(max_steps=3, max_tool_calls=99))
    assert result.reason is StopReason.STEPS and result.steps == 3 and len(llm.calls) == 3


def test_the_tool_call_budget_stops_a_run_before_the_next_call_is_made():
    result, _, shared = run([turn(tool(READ)) for _ in range(20)], budget=Budget(max_tool_calls=2, max_steps=99))
    assert result.reason is StopReason.TOOL_CALLS and result.tool_calls == 2


def test_the_write_budget_stops_the_next_write_but_not_a_read():
    script = [turn(tool(FREEZE, card_id=CARD)), turn(tool(UNFREEZE, card_id=CARD)), turn(tool(FREEZE, card_id=CARD))]
    result, _, shared = run(script, budget=Budget(max_writes=2, max_steps=99))
    assert result.reason is StopReason.WRITES and result.writes == 2
    assert [e.tool for e in shared.audit_log] == [FREEZE, UNFREEZE], "the third write never ran"


def test_a_failed_write_still_counts_toward_the_write_budget():
    script = [turn(tool(FREEZE, card_id="no_such_card")) for _ in range(10)]
    result, _, _ = run(script, budget=Budget(max_writes=3, max_errors=99, max_steps=99, max_tool_calls=99))
    assert result.reason is StopReason.WRITES and result.writes == 3 and result.errors == 3


def test_the_token_budget_stops_a_run_that_has_read_too_much():
    script = [turn(tool(READ), tokens=600) for _ in range(10)]
    result, llm, _ = run(script, budget=Budget(max_input_tokens=1000, max_steps=99, max_tool_calls=99))
    assert result.reason is StopReason.TOKENS and result.usage.input_tokens == 1200 and len(llm.calls) == 2


def test_too_many_tool_errors_end_the_run():
    script = [turn(tool("not_offered_tool_0000")) for _ in range(10)]
    result, _, _ = run(script, budget=Budget(max_errors=3, max_steps=99, max_tool_calls=99))
    assert result.reason is StopReason.ERRORS and result.errors == 3


def test_every_stop_is_a_code_value_with_no_model_text_in_it():
    result, _, _ = run([turn(tool(READ)) for _ in range(5)], budget=Budget(max_steps=1))
    assert result.reason in set(StopReason) and result.terminal is None and result.arguments == {}


# ------------------------------------------------------------------ malformed model output

@pytest.mark.parametrize(
    "bad",
    [
        ToolCall(id="c", name="made_up_tool_0000", arguments_json="{}"),
        ToolCall(id="c", name=FREEZE, arguments_json="not json"),
        ToolCall(id="c", name=FREEZE, arguments_json="[1]"),
        ToolCall(id="c", name=FREEZE, arguments_json="{}"),
        ToolCall(id="c", name=FREEZE, arguments_json='{"card_id": 5}'),
    ],
)
def test_a_malformed_tool_call_comes_back_as_a_counted_tool_error_and_the_run_goes_on(bad):
    result, llm, shared = run([turn(bad), finish()])
    assert result.finished and result.errors == 1 and result.tool_calls == 1
    assert result.log[0].ok is False
    second_call_history = llm.calls[1]["messages"]
    assert second_call_history[-1]["role"] == "tool" and second_call_history[-1]["content"].startswith("Error")
    assert shared.audit_log == () or all(not e.ok for e in shared.audit_log)


def test_a_reply_with_no_tool_call_is_nudged_then_counted():
    result, llm, _ = run([AssistantTurn(text="I will freeze it"), finish()])
    assert result.finished and result.errors == 1
    assert "call finish" in llm.calls[1]["messages"][-1]["content"]


def test_a_model_that_only_chats_is_stopped():
    result, _, _ = run([AssistantTurn(text="hmm")] * 10, budget=Budget(max_errors=3, max_steps=99))
    assert result.reason is StopReason.ERRORS


def test_an_exception_from_the_model_ends_the_run_without_raising_or_leaking_its_message():
    result, _, _ = run([RuntimeError("card 4111111111111111 down")])
    assert result.reason is StopReason.MODEL_FAILED and "4111" not in repr(result)


# ------------------------------------------------------------------ time

class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def test_no_write_starts_when_less_than_the_margin_is_left():
    clock = Clock(now=96.0)  # 4 seconds left, margin is 5
    script = [turn(tool(READ)), turn(tool(FREEZE, card_id=CARD)), finish()]
    result, _, shared = run(script, deadline=100.0, clock=clock, budget=Budget(write_margin=5.0))
    assert result.reason is StopReason.DEADLINE
    assert shared.audit_log == (), "the write was not started"
    assert [r.name for r in result.log] == [READ], "reads are still fine inside the margin"


def test_a_write_with_enough_time_left_runs():
    result, _, shared = run(
        [turn(tool(FREEZE, card_id=CARD)), finish()], deadline=100.0, clock=Clock(now=90.0), budget=Budget(write_margin=5.0)
    )
    assert result.finished and [e.tool for e in shared.audit_log] == [FREEZE]


def test_a_run_that_is_already_out_of_time_does_not_call_the_model():
    result, llm, _ = run([finish()], deadline=10.0, clock=Clock(now=10.0))
    assert result.reason is StopReason.DEADLINE and llm.calls == []


def test_a_model_call_that_outlasts_the_deadline_is_cut_off():
    class Slow(FakeToolLLM):
        async def chat(self, *a, **k):
            await asyncio.sleep(5)
            return finish()

    shared = BankingSession(fidelity=Fidelity.TAU2)
    box = ScopedToolBox(scoped(shared)(SessionRequest("i")), user_id=USER, allowed_tools=ALLOWED)

    async def go():
        loop = asyncio.get_running_loop()
        return await run_agent(Slow([]), box, system="s", request="r", deadline=loop.time() + 0.05)

    assert asyncio.run(go()).reason is StopReason.DEADLINE


def test_without_a_deadline_time_is_not_a_limit():
    result, _, _ = run([turn(tool(FREEZE, card_id=CARD)), finish()], deadline=None)
    assert result.finished


# ------------------------------------------------------------------ secrets through the loop

def test_a_pin_reaches_the_tool_but_never_the_model_across_the_whole_loop():
    vault = SecretVault()
    request = vault.capture("reset my pin, the new pin is 7351")
    llm, box, shared = setup(
        [
            turn(tool("reset_debit_card_pin_6284", card_id=CARD, last_4_digits="4821", new_pin="{{secret:pin}}")),
            finish("done"),
        ],
        vault,
    )
    result = asyncio.run(run_agent(llm, box, system="s", request=request))
    assert result.finished
    assert "7351" not in json.dumps(llm.calls)
    assert [e.tool for e in shared.audit_log] == ["reset_debit_card_pin_6284"]
