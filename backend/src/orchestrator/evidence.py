"""Check that the words a model quoted really support the intent it chose.

The prompt asks for the customer's own words as "evidence" before the kind. The
model sometimes quotes a phrase that is not in the message at all, or one that
cannot ask for the kind it picked ("I really need help getting these sorted out"
for a hand-off). An intent with unsupported evidence is dropped, but only when
another intent remains, so the customer is never left with nothing. REQUEST_HUMAN,
UPDATE_PROFILE and REPORT_FRAUD_OR_LOSS are also checked against a word list, because
they need the customer's own words; the lists are broad on purpose, as a false match
only keeps an intent.
"""

import re
from collections.abc import Iterable

from orchestrator.grounding import PERSON_WORDS
from orchestrator.types import IntentKind

_PROFILE_WORDS = re.compile(
    r"\b(e-?mail|phone|mobile|cell|number|address|contact)\b", re.IGNORECASE
)
_LOSS_WORDS = re.compile(
    r"\b(lost|loss|stolen|steal|stole|theft|fraud\w*|unauthori[sz]ed|scam\w*|hack\w*|"
    r"compromis\w*|suspicious|identity|recogni[sz]e|not mine|"
    r"(didn|did not|never)\W*(t\s+)?(make|made|authori[sz]e\w*|approve\w*))\b",
    re.IGNORECASE,
)
_TRIGGERS = {
    IntentKind.REQUEST_HUMAN: PERSON_WORDS,
    IntentKind.UPDATE_PROFILE: _PROFILE_WORDS,
    IntentKind.REPORT_FRAUD_OR_LOSS: _LOSS_WORDS,
}
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def _normal(text: str) -> str:
    return " ".join(text.translate(_QUOTES).casefold().split())


_WORDS = re.compile(r"[a-z0-9$%']{3,}")
# A model often copies a quote with one word changed ("want" for "looking"). That is
# still the customer's request; a quote of words they never wrote is not.
_MIN_OVERLAP = 0.8


def _in_message(message: str, quote: str) -> bool:
    if not quote:
        return False
    if quote in message:
        return True
    words = _WORDS.findall(quote)
    seen = set(_WORDS.findall(message))
    return bool(words) and sum(w in seen for w in words) / len(words) >= _MIN_OVERLAP


def has_trigger_words(kind: IntentKind, evidence: str) -> bool:
    """True when the kind needs no particular words, or the quote has one of them."""
    trigger = _TRIGGERS.get(kind)
    return trigger is None or trigger.search(evidence) is not None


def _supported(message: str, kind: IntentKind, evidence: str) -> bool:
    return _in_message(message, _normal(evidence)) and has_trigger_words(kind, evidence)


def unsupported_ids(
    message: str, intents: Iterable[tuple[str, IntentKind, str]]
) -> frozenset[str]:
    """Ids of intents whose evidence is not in the message or cannot ask for the kind."""
    rows = list(intents)
    text = _normal(message)
    bad = frozenset(i for i, kind, evidence in rows if not _supported(text, kind, evidence))
    return frozenset() if len(bad) == len(rows) else bad
