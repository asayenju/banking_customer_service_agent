import asyncio
import types
from typing import Any

import pytest

from orchestrator import llm as llm_module
from orchestrator.llm import FakeLLMClient, LiteLLMClient


def test_fake_records_calls_and_replays_in_order() -> None:
    asyncio.run(_fake_records())


async def _fake_records() -> None:
    fake = FakeLLMClient(["one", RuntimeError("boom")])
    assert await fake.complete([{"role": "user", "content": "hi"}], schema={"a": 1}) == "one"
    assert fake.calls[0]["temperature"] == 0.0
    assert fake.calls[0]["schema"] == {"a": 1}
    with pytest.raises(RuntimeError):
        await fake.complete([], schema={})


def test_litellm_client_sends_temperature_zero_schema_and_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        seen.update(kwargs)
        message = types.SimpleNamespace(content='{"ok": true}')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    import litellm

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    client = LiteLLMClient("openai/test-model", api_key="k")
    out = asyncio.run(
        client.complete([{"role": "user", "content": "x"}], schema={"type": "object"})
    )

    assert out == '{"ok": true}'
    assert seen["temperature"] == 0.0
    assert seen["model"] == "openai/test-model"
    assert seen["api_key"] == "k"
    assert seen["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert seen["response_format"]["json_schema"]["strict"] is True


def test_missing_key_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    monkeypatch.setattr(llm_module, "require", lambda name: (_ for _ in ()).throw(RuntimeError(name)))
    with pytest.raises(RuntimeError, match="OPENAI_KEY"):
        LiteLLMClient("openai/x")
