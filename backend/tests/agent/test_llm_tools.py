import asyncio
import os
import types
from typing import Any

import pytest

from agent import llm_tools
from agent.llm_tools import (
    AssistantTurn,
    FakeToolLLM,
    LiteLLMToolClient,
    ToolCall,
    ToolSchema,
    Usage,
    assistant_message,
    tool_message,
)
from orchestrator import config
from orchestrator.llm import FakeLLMClient, LiteLLMClient

FREEZE = ToolSchema(
    name="freeze_debit_card_3892",
    description="Freeze a debit card.",
    parameters={"type": "object", "properties": {"card_id": {"type": "string"}}, "required": ["card_id"]},
)
LOOKUP = ToolSchema(name="get_user_information_by_id", description="Look up a user.", parameters={"type": "object", "properties": {}})


def chat(client, messages=None, tools=(FREEZE,), **kwargs):
    return asyncio.run(client.chat(messages or [{"role": "user", "content": "hi"}], tools=tools, **kwargs))


# ------------------------------------------------------------------ the fake

def test_the_fake_replays_turns_in_order_and_records_what_it_was_offered():
    call = ToolCall(id="c1", name="freeze_debit_card_3892", arguments_json='{"card_id": "x"}')
    fake = FakeToolLLM([AssistantTurn(tool_calls=(call,)), AssistantTurn(text="done")])
    first = chat(fake, tools=(FREEZE, LOOKUP))
    second = chat(fake, tools=(LOOKUP,), tool_choice="required", temperature=0.2)
    assert first.tool_calls == (call,) and second.text == "done"
    assert fake.offered_tools == [["freeze_debit_card_3892", "get_user_information_by_id"], ["get_user_information_by_id"]]
    assert fake.calls[1]["tool_choice"] == "required" and fake.calls[1]["temperature"] == 0.2
    assert fake.remaining == 0


def test_the_fake_fails_loudly_on_an_unscripted_call():
    fake = FakeToolLLM([AssistantTurn(text="only one")])
    chat(fake)
    with pytest.raises(AssertionError, match="more often than scripted"):
        chat(fake)


def test_the_fake_raises_a_scripted_exception():
    fake = FakeToolLLM([TimeoutError("slow")])
    with pytest.raises(TimeoutError):
        chat(fake)


def test_a_test_can_assert_a_tool_was_never_offered():
    fake = FakeToolLLM([AssistantTurn(text="ok")])
    chat(fake, tools=(LOOKUP,))
    assert "freeze_debit_card_3892" not in [name for call in fake.offered_tools for name in call]


def test_the_fake_copies_the_messages_so_later_edits_do_not_rewrite_history():
    fake = FakeToolLLM([AssistantTurn(text="a")])
    history = [{"role": "user", "content": "first"}]
    chat(fake, messages=history)
    history[0]["content"] = "changed"
    assert fake.calls[0]["messages"][0]["content"] == "first"


# ------------------------------------------------------------------ message helpers

def test_a_turn_with_tool_calls_becomes_an_assistant_message_the_model_can_continue_from():
    turn = AssistantTurn(tool_calls=(ToolCall(id="c1", name="t", arguments_json="{}"),))
    message = assistant_message(turn)
    assert message["role"] == "assistant" and message["content"] == ""
    assert message["tool_calls"] == [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]
    assert "tool_calls" not in assistant_message(AssistantTurn(text="hello"))
    assert tool_message("c1", "ok") == {"role": "tool", "tool_call_id": "c1", "content": "ok"}


# ------------------------------------------------------------------ the real client, with the network stubbed

def stub_litellm(monkeypatch, message, usage=None):
    seen: dict[str, Any] = {}

    async def acompletion(**kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=usage)

    monkeypatch.setattr("litellm.acompletion", acompletion)
    return seen


def tool_call(id, name, arguments):
    return types.SimpleNamespace(id=id, function=types.SimpleNamespace(name=name, arguments=arguments))


def test_the_client_offers_tools_one_at_a_time_and_uses_the_agent_model_and_key(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "openai/test-agent")
    seen = stub_litellm(monkeypatch, types.SimpleNamespace(content="hi", tool_calls=None))
    turn = chat(LiteLLMToolClient(api_key="sk-test"), tools=(FREEZE,), tool_choice="required")
    assert turn.text == "hi" and turn.tool_calls == ()
    assert seen["model"] == "openai/test-agent" and seen["api_key"] == "sk-test"
    assert seen["parallel_tool_calls"] is False and seen["tool_choice"] == "required"
    assert seen["temperature"] == 0.0
    (offered,) = seen["tools"]
    assert offered == {
        "type": "function",
        "function": {"name": "freeze_debit_card_3892", "description": "Freeze a debit card.", "parameters": FREEZE.parameters},
    }


def test_tool_calls_come_back_with_their_arguments_as_unparsed_json(monkeypatch):
    message = types.SimpleNamespace(content=None, tool_calls=[tool_call("c1", "freeze_debit_card_3892", '{"card_id": "x"}'), tool_call("c2", "t", None)])
    stub_litellm(monkeypatch, message, types.SimpleNamespace(prompt_tokens=120, completion_tokens=15))
    turn = chat(LiteLLMToolClient(api_key="k"))
    assert turn.text is None
    assert [(c.id, c.name, c.arguments_json) for c in turn.tool_calls] == [
        ("c1", "freeze_debit_card_3892", '{"card_id": "x"}'),
        ("c2", "t", "{}"),
    ]
    assert turn.usage == Usage(input_tokens=120, output_tokens=15)


def test_with_no_tools_none_of_the_tool_settings_are_sent(monkeypatch):
    seen = stub_litellm(monkeypatch, types.SimpleNamespace(content="plain", tool_calls=None))
    chat(LiteLLMToolClient(api_key="k"), tools=())
    assert "tools" not in seen and "tool_choice" not in seen and "parallel_tool_calls" not in seen


def test_a_missing_key_is_a_plain_error_naming_the_variable(monkeypatch):
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    monkeypatch.setattr(config, "load_env", lambda: None)
    with pytest.raises(RuntimeError, match="OPENAI_KEY"):
        LiteLLMToolClient()


# ------------------------------------------------------------------ configuration

def test_the_agent_model_defaults_to_gpt_4o_and_is_set_apart_from_the_decomposers(monkeypatch):
    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.delenv("DECOMPOSER_MODEL", raising=False)
    assert config.agent_model() == "openai/gpt-4o" and config.decomposer_model() == "openai/gpt-4o-mini"
    monkeypatch.setenv("AGENT_MODEL", "openai/other")
    assert config.agent_model() == "openai/other" and config.decomposer_model() == "openai/gpt-4o-mini"


def test_the_decomposer_and_jev_clients_are_untouched():
    assert not hasattr(LiteLLMClient, "chat") and not hasattr(FakeLLMClient, "chat")
    assert asyncio.run(FakeLLMClient(["x"]).complete([], schema={})) == "x"
    assert "agent" not in LiteLLMClient.__module__


def test_importing_the_tool_client_neither_loads_litellm_nor_needs_a_key():
    """In a clean interpreter: the module imports, and litellm is only loaded on first use."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; "
        "import agent.llm_tools; "
        "assert 'litellm' not in sys.modules, 'litellm was imported eagerly'; "
        "print('lazy')"
    )
    env = {"PYTHONPATH": f"{root / 'src'}:{root.parent / 'tau2-bench/src'}", "PATH": os.environ.get("PATH", "")}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert done.returncode == 0 and "lazy" in done.stdout, done.stderr


# ------------------------------------------------------------------ live (opt-in)

@pytest.mark.skipif(not os.environ.get("RUN_LIVE"), reason="live model call; set RUN_LIVE=1 and OPENAI_KEY")
def test_live_the_model_returns_a_valid_tool_call():
    client = LiteLLMToolClient(model=os.environ.get("AGENT_LIVE_MODEL", "openai/gpt-4o-mini"))
    turn = asyncio.run(
        client.chat(
            [{"role": "user", "content": "Freeze the debit card with id dbc_test_1."}],
            tools=[FREEZE],
            tool_choice="required",
        )
    )
    (call,) = turn.tool_calls
    assert call.name == "freeze_debit_card_3892"
    import json

    assert json.loads(call.arguments_json) == {"card_id": "dbc_test_1"}
    assert turn.usage.input_tokens > 0
