"""The HTTP face of the system: one conversation per id, each with every specialist behind the orchestrator.

``POST /conversations/{id}/messages`` sends one customer message through ``Orchestrator.handle_turn`` and returns
the reply. A conversation is created on its first message, with its own database copy (``agent.assembly``). The
factory that builds one is ``app.state.factory``: the real one needs the keys in the repo-root ``.env``, and tests
replace it with scripted parts.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()
app.state.conversations = {}


def _live() -> Any:  # pragma: no cover - needs keys
    from agent.assembly import live_conversation

    return live_conversation()


app.state.factory = _live

# The demo page can start a conversation as one of a few sample customers: the bank's data for a tau2 task, so there
# are real accounts and cards to talk about. A conversation without a scenario is built by ``factory`` as before.
SCENARIOS = {"wallet": "task_078", "dispute": "task_084"}
WIDGET = Path(__file__).resolve().parents[2] / "frontend" / "widget.html"


def _live_scenario(task_id: str) -> Any:  # pragma: no cover - needs keys
    from agent.assembly import live_conversation
    from banking_tools import BankingSession
    from banking_tools.manifest import load_tasks

    task = next(t for t in load_tasks() if t["id"] == task_id)
    return live_conversation(BankingSession.from_task(task))


app.state.scenario_factory = _live_scenario


class MessageIn(BaseModel):
    # A message id handled once is never handled again: a double submit returns the stored reply.
    message_id: str
    message: str
    # Only read on a conversation's first message: which sample customer the demo page is playing.
    scenario: Optional[str] = None


class ReplyOut(BaseModel):
    reply: str
    questions: list[str]
    escalated: bool
    failed: bool


@app.get("/")
def read_root() -> dict[str, str]:
    return {"service": "banking customer service agent"}


@app.get("/demo")
def demo() -> FileResponse:
    return FileResponse(WIDGET, media_type="text/html")


@app.post("/conversations/{conversation_id}/messages")
async def send_message(conversation_id: str, body: MessageIn) -> ReplyOut:
    conversations: dict[str, Any] = app.state.conversations
    factory: Callable[[], Any] = app.state.factory
    if conversation_id not in conversations:
        if body.scenario is None:
            conversations[conversation_id] = factory()
        elif body.scenario in SCENARIOS:
            conversations[conversation_id] = app.state.scenario_factory(SCENARIOS[body.scenario])
        else:
            raise HTTPException(status_code=422, detail="unknown scenario")
    result = await conversations[conversation_id].orchestrator.handle_turn(body.message_id, body.message)
    return ReplyOut(
        reply=result.reply.text,
        questions=[q.text or q.slot for q in result.reply.questions],
        escalated=bool(result.reply.escalated_intent_ids),
        failed=bool(result.reply.failed_intent_ids),
    )
