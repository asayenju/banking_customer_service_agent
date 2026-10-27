"""Catches greetings and gibberish before any model sees them, and answers with a menu.

A bare "hi" or a keyboard mash scores low with Jev, and a low score escalates,
which would send a human a message that a short menu resolves. So this check
runs first, in plain code, and Jev is never asked about these messages.

It is a cheap first filter, not a classifier. A message is unclear only when
every word in it is filler or noise ("I need help" is, "I need help with my
card" is not), when it has no letters at all, or when it is mostly written in a
non-Latin script. Anything it lets through still goes to the decomposer, which
can return UNCLEAR_OR_GREETING itself. Known gap: non-English text in the Latin
script (Spanish, French) is not detected here.

Apply it to a fresh message only, never to the answer to a question we just
asked: "12345" is nonsense as an opener and a fine reply to "which account?".
"""

import re
import unicodedata
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field

from orchestrator.decision import Action, IntentDecision
from orchestrator.types import Decomposition, Intent, IntentKind

MAX_MENUS = 3

HELP_MENU = (
    "I can help with:\n"
    "- Cards: freeze or unfreeze, activate, replace, change or reset a PIN, or close a card\n"
    "- Disputes: report a charge you don't recognise\n"
    "- Money: transfer between your accounts or pay a credit card\n"
    "- Accounts: open or close one, check balances and history\n"
    "- Fees, interest and limits: ask about a charge or request a credit limit increase\n"
    "- Questions about our products and policies\n"
    "- Talking to a human agent\n"
    "\n"
    "For anything on your accounts I'll need to verify your identity first. "
    "What would you like to do?"
)

_FILLER = frozenset(
    """
    hi hello hey hiya howdy yo greetings good morning afternoon evening day there
    please pls thanks thank you thx ok okay help assist assistance support
    i i'm im i'd i'll need needs want would like to can could may me a an the
    some something anything question questions ask have got is this um uh hmm again
    """.split()
)
_KEYBOARD_RUNS = ("asdf", "qwer", "zxcv", "hjkl", "uiop", "sdfg", "dfgh")
_VOWELS = "aeiouy"
_TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")


def _is_noise(token: str) -> bool:
    if len(token) >= 4 and len(set(token)) == 1:
        return True
    if len(token) >= 3 and not any(c in _VOWELS for c in token):
        return True
    return any(run in token for run in _KEYBOARD_RUNS)


def _mostly_non_latin(letters: list[str]) -> bool:
    non_latin = sum(1 for ch in letters if not unicodedata.name(ch, "").startswith("LATIN"))
    return non_latin * 2 > len(letters)


def is_unclear(message: str) -> bool:
    letters = [ch for ch in message if ch.isalpha()]
    if not letters or _mostly_non_latin(letters):
        return True
    tokens = _TOKEN.findall(message.lower())
    return all(t in _FILLER or _is_noise(t) for t in tokens)


def precheck(message: str) -> Optional[Decomposition]:
    """An unclear message as a one-intent decomposition, or None to carry on."""
    if not is_unclear(message):
        return None
    # The summary is fixed: it must not echo a message that may hold a card number.
    intent = Intent(
        id="unclear-1", kind=IntentKind.UNCLEAR_OR_GREETING, summary="unclear or greeting"
    )
    return Decomposition(intents=(intent,))


def offer_help() -> str:
    """The read-only help tool: one fixed menu, the same for every customer."""
    return HELP_MENU


class HelpState(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    menus_shown: Annotated[int, Field(ge=0)] = 0


def handle_unclear(intent: Intent, state: HelpState) -> tuple[IntentDecision, HelpState]:
    """Show the menu for the first three unclear turns, then escalate. A turn with
    a real intent resets the count by replacing the state with ``HelpState()``."""
    if state.menus_shown >= MAX_MENUS:
        decision = IntentDecision(
            intent_id=intent.id,
            action=Action.ESCALATE,
            reason=f"still unclear after {MAX_MENUS} help menus",
        )
        return decision, state
    decision = IntentDecision(
        intent_id=intent.id, action=Action.RUN, reason="showing the help menu"
    )
    return decision, HelpState(menus_shown=state.menus_shown + 1)
