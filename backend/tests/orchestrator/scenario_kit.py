"""Shared setup for tests that drive a whole conversation through the Orchestrator.

Everything here is scripted, so nothing makes a network call: a scripted
decomposer says what each message means, a scripted scorer says how sure it is,
and scripted routes stand in for the tools and sub-agents.
"""

import asyncio

from orchestrator.auth import CustomerRecord, ScriptedDirectory
from orchestrator.decomposer import Interpretation, ScriptedDecomposer
from orchestrator.mocks import ScriptedRoute
from orchestrator.orchestrator import Orchestrator
from orchestrator.routes import Route
from orchestrator.scoring import IntentScore
from orchestrator.trace import TraceKind
from orchestrator.types import Decomposition, Intent, IntentKind

K = IntentKind
T = TraceKind
RECORD = CustomerRecord(
    user_id="u_kenji01",
    date_of_birth="03/14/1990",
    email="kenji.tanaka@example.com",
    phone_number="206-555-0293",
    address="12 Pine St, Seattle, WA 98101",
)
GOOD_CLAIMS = {"email": "kenji.tanaka@example.com", "phone_number": "206-555-0293"}
WRONG_CLAIMS = {"user_id": "u_kenji01", "email": "wrong@example.com"}
MESSAGE = "please do the thing"


def I(id, kind=K.GENERAL_QUESTION, slots=None, missing=(), depends_on=(), summary=None):  # noqa: E741
    return Intent(
        id=id,
        kind=kind,
        summary=summary or f"summary of {id}",
        slots=slots or {},
        missing_slots=missing,
        depends_on=depends_on,
    )


def interp(*intents, confirmed=(), declined=(), claims=None, flags=(), conflicts=()):
    return Interpretation(
        decomposition=Decomposition(
            intents=tuple(intents), flags=tuple(flags), conflicts=tuple(conflicts)
        ),
        confirmed=tuple(confirmed),
        declined=tuple(declined),
        claims=claims or {},
    )


def S(id, confidence):
    return IntentScore.flat(id, confidence)


class SequenceScorer:
    """Hands out one list of scores per call and remembers which intents it was asked about."""

    def __init__(self, *calls):
        self._calls = list(calls)
        self.asked = []

    async def score(self, message, decomposition):
        self.asked.append([i.id for i in decomposition.intents])
        if not self._calls:
            raise AssertionError("scorer called more often than scripted")
        return self._calls.pop(0)


class World:
    """One customer's conversation, wired to scripted parts."""

    def __init__(self, script, scores=(), *, direct=None, sub=None, scorer=None, route_timeout=0.2):
        self.decomposer = ScriptedDecomposer(script)
        self.scorer = scorer or SequenceScorer(*scores)
        self.direct = direct or ScriptedRoute()
        self.sub = sub or ScriptedRoute()
        self.orch = Orchestrator(
            decomposer=self.decomposer,
            scorer=self.scorer,
            directory=ScriptedDirectory(RECORD),
            runners={Route.DIRECT: self.direct, Route.SUB_AGENT: self.sub},
            route_timeout=route_timeout,
        )
        self._n = 0

    def turn(self, message=MESSAGE, message_id=None):
        self._n += 1
        return asyncio.run(self.orch.handle_turn(message_id or f"m{self._n}", message))


def trace_kinds(result, intent_id):
    return [e.kind for e in result.trace.events if e.intent_id == intent_id]


def decisions(result):
    return {d.intent_id: d for d in result.decisions}
