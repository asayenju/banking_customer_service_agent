"""Every test is offline unless the person running it opts in with RUN_LIVE=1.

All real model calls go through ``litellm.acompletion``. It is replaced for the whole
test run with a function that refuses, so a test that forgets to fake the network fails at
once instead of quietly spending money. Tests that exercise the real client stub the
function themselves (``monkeypatch.setattr("litellm.acompletion", ...)``), which replaces
this guard for that test only.
"""

import os

import pytest


class LiveCallInOfflineTest(AssertionError):
    pass


@pytest.fixture(autouse=True)
def no_live_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("RUN_LIVE"):
        return
    import litellm

    async def refuse(*args: object, **kwargs: object) -> object:
        raise LiveCallInOfflineTest(
            "a test tried to call a real model; fake it, or run with RUN_LIVE=1 to allow live tests"
        )

    monkeypatch.setattr(litellm, "acompletion", refuse)
