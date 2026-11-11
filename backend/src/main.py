"""The HTTP face of the system: one conversation per id, each with every specialist behind the orchestrator.

``POST /conversations/{id}/messages`` sends one customer message through ``Orchestrator.handle_turn`` and returns
the reply. A conversation is created on its first message, with its own database copy (``agent.assembly``). The
factory that builds one is ``app.state.factory``: the real one needs the keys in the repo-root ``.env``, and tests
replace it with scripted parts.
"""

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
app.state.conversations = {}


def _live() -> Any:  # pragma: no cover - needs keys
    from agent.assembly import live_conversation

    return live_conversation()


app.state.factory = _live


class MessageIn(BaseModel):
    # A message id handled once is never handled again: a double submit returns the stored reply.
    message_id: str
    message: str


class ReplyOut(BaseModel):
    reply: str
    questions: list[str]
    escalated: bool
    failed: bool


@app.get("/")
def read_root() -> dict[str, str]:
    return {"service": "banking customer service agent"}


@app.post("/conversations/{conversation_id}/messages")
async def send_message(conversation_id: str, body: MessageIn) -> ReplyOut:
    conversations: dict[str, Any] = app.state.conversations
    factory: Callable[[], Any] = app.state.factory
    if conversation_id not in conversations:
        conversations[conversation_id] = factory()
    result = await conversations[conversation_id].orchestrator.handle_turn(body.message_id, body.message)
    return ReplyOut(
        reply=result.reply.text,
        questions=[q.text or q.slot for q in result.reply.questions],
        escalated=bool(result.reply.escalated_intent_ids),
        failed=bool(result.reply.failed_intent_ids),
    )
