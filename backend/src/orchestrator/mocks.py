"""Stand-ins for the direct tool recipes and the sub-agents, which do not exist yet.

A ScriptedRoute behaves however its script says, per intent id, so a test can
make one intent succeed, another stall and a third raise, and then assert what
the orchestrator did about it. It also records what it was asked to do and how
many runs overlapped, which is how tests see parallelism directly.
"""

import asyncio
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Optional

from orchestrator.executor import OutcomeStatus, RouteOutcome
from orchestrator.types import Intent


class Behavior(StrEnum):
    SUCCESS = "success"
    SLOW = "slow"
    TIMEOUT = "timeout"
    MALFORMED = "malformed"
    PARTIAL = "partial"
    RAISES = "raises"


class ScriptedRoute:
    def __init__(
        self,
        behaviors: Optional[Mapping[str, Behavior]] = None,
        *,
        default: Behavior = Behavior.SUCCESS,
        delay: float = 0.01,
    ) -> None:
        self._behaviors = dict(behaviors or {})
        self._default = default
        self._delay = delay
        self.calls: list[str] = []
        self.peak_concurrency = 0
        self._active = 0

    async def run(self, intent: Intent) -> RouteOutcome:
        self.calls.append(intent.id)
        self._active += 1
        self.peak_concurrency = max(self.peak_concurrency, self._active)
        try:
            return await self._behave(intent, self._behaviors.get(intent.id, self._default))
        finally:
            self._active -= 1

    async def _behave(self, intent: Intent, behavior: Behavior) -> RouteOutcome:
        done = RouteOutcome(
            intent_id=intent.id, status=OutcomeStatus.DONE, text=f"Done: {intent.summary}"
        )
        match behavior:
            case Behavior.SUCCESS:
                return done
            case Behavior.SLOW:
                await asyncio.sleep(self._delay)
                return done
            case Behavior.TIMEOUT:
                await asyncio.sleep(3600)
                return done
            case Behavior.MALFORMED:
                garbage: Any = "not an outcome"
                return garbage  # type: ignore[no-any-return]
            case Behavior.PARTIAL:
                return RouteOutcome(
                    intent_id=intent.id,
                    status=OutcomeStatus.PARTIAL,
                    text="did the first step",
                    error="second step not completed",
                )
            case Behavior.RAISES:
                raise RuntimeError("boom with card 4111111111111111")
