"""The HTTP face of the system: one conversation per id, each with every specialist behind the orchestrator.

``POST /conversations/{id}/messages`` sends one customer message through ``Orchestrator.handle_turn`` and returns
the reply. A conversation is created on its first message, with its own database copy (``agent.assembly``). The
factory that builds one is ``app.state.factory``: the real one needs the keys in the repo-root ``.env``, and tests
replace it with scripted parts.
"""

import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import persona_text
from orchestrator import observe

app = FastAPI()
# The demo page may be hosted on another domain (Vercel): only that page's origin (and a local server) may call the API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://REDACTED.example.invalid", "http://localhost:8765", "http://127.0.0.1:8765"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.state.conversations = {}


def _live() -> Any:  # pragma: no cover - needs keys
    from agent.assembly import live_conversation

    return live_conversation()


app.state.factory = _live

# The demo page can start a conversation as any of the test customers: the bank's data for a tau2 task, so there
# are real accounts and cards to talk about. A conversation without a scenario is built by ``factory`` as before.
WIDGET = Path(__file__).resolve().parents[2] / "frontend" / "widget.html"


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


# The accounts tasks that are distinct customers; the other accounts tasks repeat one of these with other amounts.
ACCOUNTS_DEMO = frozenset({"task_055", "task_056", "task_060", "task_062", "task_064", "task_067", "task_076"})


# The credit tasks that are distinct customers; the other credit tasks repeat one of these (same flow, more transactions or
# the same reason for closing), so the demo lists only these.
CREDIT_DEMO = frozenset(
    {
        "task_031", "task_036", "task_038", "task_041",  # disputes: one charge, review first, several plus a new card, many across cards
        "task_043", "task_044", "task_048", "task_049",  # closing: annual fee and a balance, a competitor, many cards, a customer who argues
        "task_050", "task_051", "task_054",  # limit: approved, denied for utilization, and with a dispute and a replacement
    }
)


# The debit tasks that are distinct customers; the dropped ones repeat a flow with more transactions or another card.
DEBIT_DEMO = frozenset(
    {
        "task_077", "task_078", "task_079", "task_080", "task_081",  # wallet: maybe lost, lost (replacement limit), stolen (pending refund), five cards and one found, asks for a person
        "task_082", "task_083", "task_084",  # disputes: lost card plus unauthorized, four kinds of dispute, existing dispute limit
        "task_087", "task_088", "task_089", "task_090", "task_092",  # declines: three declined, one declined, teen account, PIN lock with fraud, PIN lock sent to a person
    }
)


def personas() -> list[dict[str, Any]]:
    """The test customers, listed by name and area only. Who they are comes from ``persona`` when one is opened;
    the list carries no situation and no user id."""
    from banking_tools.manifest import build_manifest, load_tasks

    from agent.credit_report import credit_task_ids

    tasks = load_tasks()
    entries = {e.task_id: e for e in build_manifest(tasks)}
    credit = set(credit_task_ids(tasks))
    found: list[dict[str, Any]] = []
    for task in tasks:
        who = _verified(task)
        if who is None:
            continue
        entry = entries[task["id"]]
        if list(entry.clusters) == ["accounts"] and task["id"] not in ACCOUNTS_DEMO:
            continue
        if "debit" in entry.clusters and task["id"] not in DEBIT_DEMO:
            continue
        if task["id"] in credit and task["id"] not in CREDIT_DEMO:
            continue
        found.append(
            {
                "id": task["id"],
                "name": who.get("name"),
                "areas": list(entry.clusters) or (["human transfer"] if entry.human_transfer else ["other"]),
            }
        )
    return found


_OPENING = re.compile(r"\*{0,2}(?:Opening|Start naturally)[^:\n]*:\*{0,2}[^\"“\n]*[\"“]([^\"”]+)[\"”]", re.IGNORECASE)
# A customer whose script gives no quoted first line, only what to start by saying: that, in the customer's own voice.
OPENINGS: dict[str, str] = {
    "task_005": "Hi, I'd like to check the balance on my Gold Rewards card, please.",
}


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
    said = opening.group(1).strip() if opening else OPENINGS.get(task_id, "")
    shown = persona_text.sections(full, {str(who.get("user_id") or "")})
    by_title = {sec["title"]: sec for sec in shown}
    return {
        "id": task_id,
        "name": who.get("name"),
        # The brief in full, as plain text: a section per block (who they are, what they want, each card's disputes...).
        "sections": shown,
        "character": by_title.get("Who they are", {}).get("text", ""),
        "situation": by_title.get("Their situation", {}).get("text", ""),
        "phone": who.get("phone_number"),
        "email": who.get("email"),
        "born": who.get("date_of_birth"),
        "address": who.get("address"),
        "opening": said,
        "card_digits": _card_digits(task),
    }


app.state.scenario_factory = _live_scenario



class MessageIn(BaseModel):
    # A message id handled once is never handled again: a double submit returns the stored reply.
    message_id: str
    message: str
    # Only read on a conversation's first message: the task id of the test customer the demo page is playing.
    scenario: Optional[str] = None
    # The id of a button this conversation offered. Pressing one means exactly what was offered; the page sends no more.
    pick: Optional[str] = None


class AskOut(BaseModel):
    slot: str
    # "yes_no" when the answer is a plain yes or no; the chat offers Yes and No for it and nothing otherwise.
    kind: str


class ChoiceOut(BaseModel):
    id: str
    # The line of the reply it belongs to, and the word on the button.
    label: str
    action: str


class ReplyOut(BaseModel):
    reply: str
    questions: list[str]
    asks: list[AskOut]
    choices: list[ChoiceOut]
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


# For testing: what each turn decided, ran and where it failed. No message text, values or amounts (see orchestrator/observe.py).
app.state.recorder = observe.Recorder()


@app.get("/debug/conversations/{conversation_id}/trace")
def conversation_trace(conversation_id: str) -> list[dict[str, Any]]:
    return app.state.recorder.conversation(conversation_id)


@app.get("/debug/summary")
def debug_summary() -> dict[str, Any]:
    summary: dict[str, Any] = app.state.recorder.summary()
    return summary


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
    notes = observe.begin()
    started = time.monotonic()
    result = await conversations[conversation_id].orchestrator.handle_turn(body.message_id, body.message, body.pick)
    app.state.recorder.record(conversation_id, body.message_id, result, notes, started, [d.intent_id for d in result.decisions])
    return ReplyOut(
        reply=result.reply.text,
        questions=[q.text or q.slot for q in result.reply.questions],
        asks=[AskOut(slot=q.slot, kind=q.kind) for q in result.reply.questions],
        choices=[ChoiceOut(id=c.id, label=c.label, action=c.action) for c in result.reply.choices],
        escalated=bool(result.reply.escalated_intent_ids),
        failed=bool(result.reply.failed_intent_ids),
    )
