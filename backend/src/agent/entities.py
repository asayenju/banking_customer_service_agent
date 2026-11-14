"""Which of the customer's debit cards do their words mean? One resolver for every path that names cards.

Built on ``cards.named_cards`` (all/every, card ids, last four digits, account names, a pasted list line) and
adding what a reply to a list needs: "the first two", "the second one", "those", and a report of what could not
be placed. It only ever returns the customer's own cards from the list it is given, and never picks one for them:
a number that is not theirs, or a closed card, comes back in ``unknown`` or ``closed`` so the reply can say so.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent.cards import Card, named_cards

_ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4, "1st": 0, "2nd": 1, "3rd": 2, "4th": 3, "5th": 4}
_NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "2": 2, "3": 3, "4": 4, "5": 5}
_THOSE = re.compile(r"\b(?:those|these|them|those ones|the ones above|the ones you listed)\b", re.IGNORECASE)
_FOUR = re.compile(r"(?<!\d)\d{4}(?!\d)")


@dataclass(frozen=True)
class Resolution:
    cards: tuple[Card, ...] = ()
    every: bool = False
    # Four-digit numbers the customer said that are not one of their cards.
    unknown: tuple[str, ...] = ()
    # Cards the customer named that cannot be used (closed).
    closed: tuple[Card, ...] = ()
    # Four-digit numbers that fit more than one card on offer: never guessed at.
    ambiguous: tuple[str, ...] = ()

    @property
    def found(self) -> bool:
        return bool(self.cards)


def ordinal_picks(said: str, shown: Sequence[Card]) -> list[Card]:  # any list the customer was shown, in order
    """"the first two", "the second one", "the last one", counted along the list the customer was shown."""
    words = re.findall(r"[a-z0-9]+", said)
    if "last" in words and shown:
        return [shown[-1]]
    for i, word in enumerate(words):
        if word == "first" and i + 1 < len(words) and words[i + 1] in _NUMBER_WORDS:
            return list(shown[: _NUMBER_WORDS[words[i + 1]]])
    if re.fullmatch(r"\s*(?:number\s*|option\s*|#)?([1-9])\.?\s*", said):
        n = int(re.sub(r"\D", "", said)) - 1
        return [shown[n]] if n < len(shown) else []
    picks = [shown[_ORDINALS[w]] for w in words if w in _ORDINALS and _ORDINALS[w] < len(shown)]
    return [c for n, c in enumerate(picks) if c not in picks[:n]]


def resolve_cards(text: str, shown: Sequence[Card], db: Any, *, owned: Sequence[Card] = ()) -> Resolution:
    """The cards ``text`` names among ``shown`` (the cards on offer, in the order they were listed).

    ``owned`` adds the customer's other cards (closed ones) so a closed card they name is reported, not guessed at.
    """
    said = (text or "").strip()
    if not said:
        return Resolution()
    found, every = named_cards(said, shown, db)
    if every:
        return Resolution(cards=tuple(shown), every=True)
    if not found:
        found = ordinal_picks(said.lower(), shown)
    if not found and _THOSE.search(said) and shown:
        return Resolution(cards=tuple(shown), every=True)
    on_offer = {c.get("last_4_digits") for c in shown}
    shared = tuple(n for n in dict.fromkeys(_FOUR.findall(said)) if sum(c.get("last_4_digits") == n for c in shown) > 1)
    held = {c.get("last_4_digits"): c for c in owned}
    closed: list[Card] = []
    unknown: list[str] = []
    for four in _FOUR.findall(said):
        if four in on_offer:
            continue
        if four in held and held[four] not in closed:
            closed.append(held[four])
        elif four not in held and four not in unknown:
            unknown.append(four)
    kept = tuple(c for c in found if c.get("last_4_digits") not in shared)
    return Resolution(cards=kept, every=False, unknown=tuple(unknown), closed=tuple(closed), ambiguous=shared)
