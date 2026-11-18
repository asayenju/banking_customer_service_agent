"""A test customer's written brief, as the demo page shows it to the person playing them.

The brief is the task's instructions: markdown, with a heading per block (``**Your goal:**``, ``**GOLD REWARDS CARD
DISPUTES (4 transactions):**``), lists, and a script for how the customer talks. The page shows all of it as plain text
in sections, so the tester sees every detail they need to answer with (which charges, why, what the customer knows) and
no markdown marks. Nothing is cut short, and an id of the customer's record is never part of it.
"""

import re
from typing import Any

from orchestrator.plain import plain_text

_BOILERPLATE = "You are playing the role of a customer contacting a customer service representative agent."
_FLOW = re.compile(r"^\s*(?:#+\s*)?\**Conversation Flow\**:?\s*$", re.IGNORECASE | re.MULTILINE)
_HEADED = re.compile(r"^\s*\*\*(.+?)\*\*\s*:?\s*(.*)$")
_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$")
_USER_ID = re.compile(r"[^.!?\n]*\buser[ _-]?id\b[^.!?\n]*[.!?]?\s*", re.IGNORECASE)
_TITLES = {
    "your character": "Who they are",
    "your situation": "Their situation",
    "your goal": "What they want",
    "information you know and should provide when asked": "What they know",
}
# The verification details are shown on their own, with the lines that fill them in.
_SKIPPED = re.compile(r"^verification info", re.IGNORECASE)


def _title(raw: str) -> str:
    cleaned = plain_text(raw).strip().rstrip(":").strip()
    mapped = _TITLES.get(cleaned.lower())
    if mapped:
        return mapped
    main, bracket, rest = cleaned.partition("(")
    if not main.strip().isupper():
        return cleaned
    # "GOLD REWARDS CARD DISPUTES (4 transactions)" reads as a heading, not a shout.
    return main.strip().title() + (" " if bracket else "") + bracket + rest


def _plain(text: str) -> str:
    """Plain text, with any bold mark the markdown left unclosed ("**If the agent corrects you...") taken out too."""
    return plain_text(text).replace("**", "").replace("__", "").strip()


def _block(lines: list[str]) -> dict[str, Any]:
    items = [_plain(m.group(1)) for line in lines if (m := _ITEM.match(line))]
    prose = [_plain(line) for line in lines if line.strip() and not _ITEM.match(line)]
    return {"text": " ".join(p for p in prose if p), "items": [i for i in items if i]}


def sections(instructions: str, secrets: set[str] = frozenset()) -> list[dict[str, Any]]:  # type: ignore[assignment]
    """[{title, text, items}] for each block of the brief, then "How they talk" for the conversation script."""
    text = instructions.replace(_BOILERPLATE, "").strip()
    for secret in secrets - {""}:
        text = text.replace(secret, "")
    text = _USER_ID.sub("", text)
    flow_at = _FLOW.search(text)
    head, flow = (text[: flow_at.start()], text[flow_at.end() :]) if flow_at else (text, "")
    found: list[dict[str, Any]] = []
    for chunk in re.split(r"\n\s*\n", head):
        lines = [line for line in chunk.splitlines() if line.strip()]
        if not lines:
            continue
        match = _HEADED.match(lines[0])
        if match and match.group(1).strip():
            title, rest = _title(match.group(1)), match.group(2)
            body = _block(([rest] if rest.strip() else []) + lines[1:])
        else:
            title, body = "", _block(lines)
        if _SKIPPED.match(title) or not (body["text"] or body["items"]):
            continue
        if not title and found:  # unheaded lines belong to the section above them
            last = found[-1]
            last["text"] = (last["text"] + " " + body["text"]).strip()
            last["items"] += body["items"]
            continue
        found.append({"title": title or "About them", **body})
    steps = [
        _plain(m.group(1) if (m := _ITEM.match(line)) else line)
        for line in flow.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    steps = [s for s in steps if s]
    if steps:
        found.append({"title": "How they talk", "text": "", "items": steps})
    return found
