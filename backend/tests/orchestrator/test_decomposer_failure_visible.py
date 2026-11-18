"""When the decomposer gives up and a person takes over, the log says why (exception types only, never text)."""

import asyncio
import logging

from orchestrator.llm import FakeLLMClient
from orchestrator.llm_decomposer import LLMDecomposer


def test_a_failed_decomposition_is_logged_with_the_exception_types_and_no_message_text(caplog):
    secret = "my pin is 7351 and my card is 4111111111111111"
    client = FakeLLMClient([TimeoutError(secret), "not json", ValueError(secret)])
    decomposer = LLMDecomposer(client, max_attempts=3)
    with caplog.at_level(logging.WARNING, logger="orchestrator.llm_decomposer"):
        result = asyncio.run(decomposer.interpret("close my card", []))
    assert result.decomposition.intents[0].id == "unprocessed"
    line = " ".join(r.getMessage() for r in caplog.records)
    assert "decomposer failed on turn 1 after 3 attempts" in line
    assert "TimeoutError" in line and "ValidationError" in line and "ValueError" in line
    assert "7351" not in line and "4111" not in line
    assert len(decomposer.last_failure) == 3
