"""What Jev may be told about the situation, and the extra questions it may be asked (all off unless switched on).

State is facts in code, as types and flags only: whether the caller is verified, which detail the assistant just asked
for, whether a plan is waiting for a yes, how often this request has been asked about. Never a value, id, amount, date
or digit. Questions beyond "is the kind right" are the Choice over every kind (to find the runner-up and to see when two
kinds are really one reading) and signals asked in the same call: "asks for a person", "answers the question on
screen", and frustration on a 0 to 4 scale. Signals are logged and compared with the code's own checks, and decide nothing.

Switches (environment): ``JEV_STATE=facts``, ``JEV_CHOICE=1``, ``JEV_SIGNALS=1``.
"""

import os
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from orchestrator.capabilities import CAPABILITY
from orchestrator.decision import risk_of
from orchestrator.routes import route_for
from orchestrator.types import IntentKind

READING = "reading"
SIGNAL_HUMAN = "asks_for_a_person"
SIGNAL_ANSWER = "answers_the_question_on_screen"
SIGNAL_FRUSTRATION = "frustration"
SIGNAL_NAMES = (SIGNAL_HUMAN, SIGNAL_ANSWER, SIGNAL_FRUSTRATION)


def facts_enabled() -> bool:
    return os.environ.get("JEV_STATE", "") == "facts"


def choice_enabled() -> bool:
    return os.environ.get("JEV_CHOICE", "") == "1"


def signals_enabled() -> bool:
    return os.environ.get("JEV_SIGNALS", "") == "1"


def build(
    *,
    verified: bool,
    question_slot: Optional[str],
    plan_waiting: bool,
    other_open_kinds: Sequence[IntentKind],
    asks_so_far: int,
) -> tuple[tuple[str, str], ...]:
    """The facts, as (name, value) pairs of words and yes/no only."""
    facts = [
        ("caller_identity_verified", "yes" if verified else "no"),
        ("a_plan_is_waiting_for_the_customers_yes", "yes" if plan_waiting else "no"),
        ("times_this_request_was_already_asked_about", {0: "none", 1: "once"}.get(asks_so_far, "several times")),
    ]
    if question_slot:
        facts.append(("detail_the_assistant_just_asked_for", question_slot.replace("_", " ")))
    if other_open_kinds:
        facts.append(("other_open_requests", ", ".join(sorted({k.value.lower().replace("_", " ") for k in other_open_kinds}))))
    return tuple(facts)


def same_reading(a: IntentKind, b: IntentKind) -> bool:
    """Two kinds that run the same way at the same risk are one reading to the customer: history versus lookup is not a
    difference worth asking about. Never true for anything that changes something."""
    if a is b:
        return True
    return risk_of(a).value == "read" and risk_of(b).value == "read" and route_for(a) is route_for(b)


def choice_question() -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": "Which of these is the customer asking the assistant to do? Pick the closest.",
        "criteria": {kind.value: text for kind, text in CAPABILITY.items()},
    }


def signal_questions() -> dict[str, dict[str, Any]]:
    return {
        SIGNAL_HUMAN: {"type": "noul", "instructions": "Is the customer asking to speak with a person instead of the assistant?"},
        SIGNAL_ANSWER: {
            "type": "noul",
            "instructions": "Is the customer's latest message an answer to the question the assistant just asked, rather than a new request?",
        },
        SIGNAL_FRUSTRATION: {
            "type": "score",
            "instructions": "How frustrated does the customer sound?",
            "criteria": ["not at all", "slightly", "somewhat", "clearly frustrated", "angry"],
        },
    }


def runner_up(kind: IntentKind, probabilities: Mapping[str, float]) -> tuple[float, Optional[str]]:
    """(the probability that the kind is right, counting siblings that run the same way; the next most likely different
    reading, if there is one)."""
    mass = 0.0
    best: Optional[tuple[float, str]] = None
    for name, p in probabilities.items():
        try:
            other = IntentKind(name)
        except ValueError:
            continue
        if same_reading(kind, other):
            mass += p
        elif best is None or p > best[0]:
            best = (p, name)
    return min(mass, 1.0), (best[1] if best is not None else None)
