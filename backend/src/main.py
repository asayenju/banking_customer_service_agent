"""The HTTP face of the system: one conversation per id, each with every specialist behind the orchestrator.

``POST /conversations/{id}/messages`` sends one customer message through ``Orchestrator.handle_turn`` and returns
the reply. A conversation is created on its first message, with its own database copy (``agent.assembly``). The
factory that builds one is ``app.state.factory``: the real one needs the keys in the repo-root ``.env``, and tests
replace it with scripted parts.
"""

import re
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


def _verified(task: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Who the task's customer is, from its own verification record (the bank's record when that holds placeholders)."""
    import re

    actions = (task.get("evaluation_criteria") or {}).get("actions") or []
    who = next((a["arguments"] for a in actions if a["name"] == "log_verification"), None)
    if who is None:
        return None  # no customer to verify: the task is the customer's own side or a read
    if not re.fullmatch(r"\d{3}-\d{3}-\d{4}", str(who.get("phone_number") or "")):
        # A few tasks log placeholder text instead of the customer's details: read the record itself.
        from banking_tools import BankingSession

        users = BankingSession.from_task(task).database.users.data
        said = str((task.get("user_scenario") or {}).get("instructions") or "").lower()
        record: dict[str, Any] = users.get(str(who.get("user_id"))) or next(
            (u for u in users.values() if str(u.get("email", "")).lower() in said or str(u.get("phone_number", "")) in said), {}
        )
        who = {**who, "name": record.get("name"), "phone_number": record.get("phone_number"), "email": record.get("email"),
               "date_of_birth": record.get("date_of_birth"), "address": record.get("address")}
    return who


def personas() -> list[dict[str, Any]]:
    """The test customers, listed by name and area only. Who they are comes from ``persona`` when one is opened;
    the list carries no situation and no user id."""
    from banking_tools.manifest import build_manifest, load_tasks

    tasks = load_tasks()
    entries = {e.task_id: e for e in build_manifest(tasks)}
    found: list[dict[str, Any]] = []
    for task in tasks:
        who = _verified(task)
        if who is None:
            continue
        entry = entries[task["id"]]
        found.append(
            {
                "id": task["id"],
                "name": who.get("name"),
                "areas": list(entry.clusters) or (["human transfer"] if entry.human_transfer else ["other"]),
            }
        )
    return found


_CUT = re.compile(r"\*{0,2}Verification info|## Conversation Flow|Here are your behavior|Conversation Flow", re.IGNORECASE)
_OPENING = re.compile(r"\*{0,2}Opening[^:\n]*:\*{0,2}\s*[\"“]([^\"”]+)[\"”]", re.IGNORECASE)
_LABEL = re.compile(r"\*\*Your (?:character|situation):\*\*\s*")
_USER_ID = re.compile(r"[^.!?\n]*\buser[ _-]?id\b[^.!?\n]*[.!?]?\s*", re.IGNORECASE)


def _card_digits(task: dict[str, Any]) -> list[str]:
    """The card last-four digits the reference answers use. The records do not hold them for credit cards, so a tester
    playing the customer needs them to answer "what are the last four digits on the card?"."""
    import json

    found: list[str] = []
    for action in (task.get("evaluation_criteria") or {}).get("actions") or []:
        args = action.get("arguments") or {}
        if action.get("name") == "call_discoverable_agent_tool":
            try:
                args = json.loads(args.get("arguments") or "{}")
            except (TypeError, ValueError):
                args = {}
        for key in ("card_last_4_digits", "last_4_digits"):
            value = str(args.get(key) or "")
            if value.isdigit() and len(value) == 4 and value not in found:
                found.append(value)
    return found


def persona(task_id: str) -> Optional[dict[str, Any]]:
    """One test customer in full: who they are, their situation and the details they verify with. Never the user id."""
    from banking_tools.manifest import load_tasks

    task = next((t for t in load_tasks() if t["id"] == task_id), None)
    who = _verified(task) if task is not None else None
    if task is None or who is None:
        return None
    full = str((task.get("user_scenario") or {}).get("instructions") or "")
    opening = _OPENING.search(full)
    text = full.replace(_BOILERPLATE, "").strip()
    cut = _CUT.search(text)
    text = _USER_ID.sub("", text[: cut.start()] if cut else text)
    for secret in {str(who.get("user_id") or "")} - {""}:
        text = text.replace(secret, "")
    parts = [_LABEL.sub("", p).strip() for p in text.split("\n\n") if p.strip()]
    return {
        "id": task_id,
        "name": who.get("name"),
        "character": parts[0][:400] if parts else "",
        "situation": " ".join(parts[1:])[:700],
        "phone": who.get("phone_number"),
        "email": who.get("email"),
        "born": who.get("date_of_birth"),
        "address": who.get("address"),
        "opening": opening.group(1).strip() if opening else "",
        "card_digits": _card_digits(task),
    }


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


@app.get("/personas/{task_id}")
def one_persona(task_id: str) -> dict[str, Any]:
    found = persona(task_id)
    if found is None:
        raise HTTPException(status_code=404, detail="unknown test customer")
    return found


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
