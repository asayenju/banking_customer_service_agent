"""What kind of message each intent came from, and what that allows.

The model labels every intent with how the customer is talking: asking the bank to
do something, asking a question, shopping, describing a problem, complaining. Only
an ACTION_REQUEST may stay a write. The label is a second opinion that code
enforces; it never authorizes anything (``decision`` still gates every write).

A write intent that is not an action request is dropped when other intents remain.
If every intent is such a write, they become the read kind that fits the message,
so "nothing was asked of the bank" is an ordinary result and not a failure. A
reply that cannot be read as any of these goes to a person (OTHER).
"""

import re
from collections.abc import Iterable, Mapping
from enum import StrEnum

from orchestrator.evidence import has_trigger_words
from orchestrator.plan import READ_ONLY_KINDS
from orchestrator.types import IntentKind


class CommunicationType(StrEnum):
    ACTION_REQUEST = "ACTION_REQUEST"
    INFORMATION_REQUEST = "INFORMATION_REQUEST"
    PRODUCT_SHOPPING = "PRODUCT_SHOPPING"
    PROBLEM_STATEMENT = "PROBLEM_STATEMENT"
    COMPLAINT = "COMPLAINT"
    UNCLEAR = "UNCLEAR"


_READ_FOR: Mapping[CommunicationType, IntentKind] = {
    CommunicationType.INFORMATION_REQUEST: IntentKind.GENERAL_QUESTION,
    CommunicationType.PRODUCT_SHOPPING: IntentKind.GENERAL_QUESTION,
    CommunicationType.PROBLEM_STATEMENT: IntentKind.ACCOUNT_BALANCE_OR_HISTORY,
    CommunicationType.COMPLAINT: IntentKind.ACCOUNT_BALANCE_OR_HISTORY,
    CommunicationType.UNCLEAR: IntentKind.OTHER,
}


# Telling the bank "I lost my card" is the report itself, so it needs no further ask,
# but only when the quote says so: "my cards are locked" is a symptom.
_ALSO_FINE: Mapping[IntentKind, frozenset[CommunicationType]] = {
    IntentKind.REPORT_FRAUD_OR_LOSS: frozenset({CommunicationType.PROBLEM_STATEMENT}),
    # "My cash back looks wrong" is asking for the review: for fees, interest and rewards the review is the service. The
    # fix itself is still shown as a plan and runs only after the customer's yes. Only when the quote names the topic.
    IntentKind.FEE_OR_INTEREST_ADJUSTMENT: frozenset({CommunicationType.PROBLEM_STATEMENT, CommunicationType.COMPLAINT}),
}
# What a fee, interest or rewards problem is about: without one of these words a problem statement stays a read.
_FEE_TOPIC = re.compile(r"\b(?:fees?|interest|apy|rewards?|cash ?back|points|atm|charged)\b", re.IGNORECASE)


def _stated_problem_with_review(kind: IntentKind, t: CommunicationType, evidence: str) -> bool:
    if kind is IntentKind.FEE_OR_INTEREST_ADJUSTMENT:
        return t in _ALSO_FINE[kind] and _FEE_TOPIC.search(evidence) is not None
    return t in _ALSO_FINE.get(kind, frozenset()) and has_trigger_words(kind, evidence)


def unrequested_writes(
    intents: Iterable[tuple[str, IntentKind, CommunicationType, str]],
) -> tuple[frozenset[str], Mapping[str, IntentKind]]:
    """Ids to drop, and ids whose kind becomes a read, for writes nobody asked for."""
    rows = list(intents)
    unrequested = [
        (i, t)
        for i, kind, t, evidence in rows
        if kind not in READ_ONLY_KINDS
        and kind is not IntentKind.OTHER
        and t is not CommunicationType.ACTION_REQUEST
        and not _stated_problem_with_review(kind, t, evidence)
    ]
    if not unrequested:
        return frozenset(), {}
    if len(unrequested) == len(rows):
        return frozenset(), {i: _READ_FOR[t] for i, t in unrequested}
    return frozenset(i for i, _ in unrequested), {}
