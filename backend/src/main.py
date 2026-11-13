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

# The demo page can start a conversation as any of the test customers: the bank's data for a tau2 task, so there
# are real accounts and cards to talk about. A conversation without a scenario is built by ``factory`` as before.
WIDGET = Path(__file__).resolve().parents[2] / "frontend" / "widget.html"
_BOILERPLATE = "You are playing the role of a customer contacting a customer service representative agent."


def _live_scenario(task_id: str) -> Any:  # pragma: no cover - needs keys
    from agent.assembly import live_conversation
    from banking_tools import BankingSession
    from banking_tools.manifest import load_tasks

    task = next(t for t in load_tasks() if t["id"] == task_id)
    return live_conversation(BankingSession.from_task(task))


def personas() -> list[dict[str, Any]]:
    """Every test customer: who they are (from the task's own verification record), what they want, and the area."""
    import re

    from banking_tools.manifest import build_manifest, load_tasks
    from knowledge.report import opening_message

    tasks = load_tasks()
    entries = {e.task_id: e for e in build_manifest(tasks)}
    found: list[dict[str, Any]] = []
    for task in tasks:
        actions = (task.get("evaluation_criteria") or {}).get("actions") or []
        who = next((a["arguments"] for a in actions if a["name"] == "log_verification"), None)
        if who is None:
            continue  # no customer to verify: the task is the customer's own side or a read
        if not re.fullmatch(r"\d{3}-\d{3}-\d{4}", str(who.get("phone_number") or "")):
            # A few tasks log placeholder text instead of the customer's details: read the record itself.
            from banking_tools import BankingSession

            users = BankingSession.from_task(task).database.users.data
            said = str((task.get("user_scenario") or {}).get("instructions") or "").lower()
            record = users.get(str(who.get("user_id"))) or next(
                (u for u in users.values() if str(u.get("email", "")).lower() in said or str(u.get("phone_number", "")) in said), {}
            )
            who = {**who, "name": record.get("name"), "phone_number": record.get("phone_number"), "email": record.get("email"),
                   "date_of_birth": record.get("date_of_birth"), "address": record.get("address")}
        text = (task.get("user_scenario") or {}).get("instructions")
        about = text.replace(_BOILERPLATE, "").strip() if isinstance(text, str) else ""
        about = re.sub(r"\*\*Your character:\*\*\s*", "", about)
        about = " ".join(re.split(r"(?<=[.!?])\s+", about.split("\n")[0])[:2])[:220]
        entry = entries[task["id"]]
        found.append(
            {
                "id": task["id"],
                "name": who.get("name"),
                "phone": who.get("phone_number"),
                "email": who.get("email"),
                "born": who.get("date_of_birth"),
                "address": who.get("address"),
                "about": about,
                "opening": opening_message(task).replace(_BOILERPLATE, "").strip()[:400],
                "areas": list(entry.clusters) or (["human transfer"] if entry.human_transfer else ["other"]),
            }
        )
    return found


app.state.scenario_factory = _live_scenario



class MessageIn(BaseModel):
    # A message id handled once is never handled again: a double submit returns the stored reply.
    message_id: str
    message: str
    # Only read on a conversation's first message: the task id of the test customer the demo page is playing.
    scenario: Optional[str] = None


class ReplyOut(BaseModel):
    reply: str
    questions: list[str]
    escalated: bool
    failed: bool


@app.get("/")
def read_root() -> dict[str, str]:
    return {"service": "banking customer service agent"}


@app.get("/personas")
def list_personas() -> list[dict[str, Any]]:
    if not hasattr(app.state, "personas"):
        app.state.personas = personas()
    result: list[dict[str, Any]] = app.state.personas
    return result


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
        elif body.scenario in {p["id"] for p in list_personas()}:
            conversations[conversation_id] = app.state.scenario_factory(body.scenario)
        else:
            raise HTTPException(status_code=422, detail="unknown scenario")
    result = await conversations[conversation_id].orchestrator.handle_turn(body.message_id, body.message)
    return ReplyOut(
        reply=result.reply.text,
        questions=[q.text or q.slot for q in result.reply.questions],
        escalated=bool(result.reply.escalated_intent_ids),
        failed=bool(result.reply.failed_intent_ids),
    )
