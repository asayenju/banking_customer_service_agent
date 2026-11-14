"""A question is never shown twice the same way.

When the customer's answer did not settle a question, asking it again word for word is what made the loops (the same
"Which card do you mean?" four times, then a hand-off). The ladder changes strategy each time the same question is
about to be shown for the same request:

  1. first time: as written;
  2. second time: say so and narrow it to numbered options (a list in the question becomes 1, 2, 3...);
  3. third time: offer a person, and a last way to answer;
  (the orchestrator's ask limit then hands over, so a person comes only after two failed repairs).

Wording is fixed templates written here, never by a model or by Jev. Confirmations and the identity question are not
repaired: a yes is asked as it is, and the identity prompt is already a list of choices.
"""

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence

from orchestrator.auth import IDENTITY_SLOT
from orchestrator.decision import CONFIRM_SLOT, Action, IntentDecision
from orchestrator.types import Intent, Question

_OPTION = re.compile(r"^\s*-\s+(.*\S)\s*$")
SORRY = "Sorry, I didn't catch which you meant."


def numbered(text: str) -> str:
    """The question narrowed to numbered choices, or reworded when it has no list."""
    lines = text.splitlines()
    options = [m.group(1) for line in lines if (m := _OPTION.match(line))]
    stem = [line for line in lines if not _OPTION.match(line) and line.strip() and not line.lower().startswith("say '")]
    if len(options) >= 2:
        listing = "\n".join(f"{n}. {o}" for n, o in enumerate(options, 1))
        return f"{SORRY} Please reply with just a number:\n{listing}"
    first = stem[0] if stem else text
    return f"{SORRY} Could you say it another way? {first}"


def offer_person(text: str) -> str:
    return (
        "I'm still not sure I have this right, and I don't want to guess. You can tell me in your own words once "
        "more (the last four digits of the card work best), or say 'person' and I'll connect you with someone."
    )


class RepairLadder:
    """Remembers what was shown for each request, and rewords a question that is about to repeat."""

    def __init__(self) -> None:
        self._base: dict[str, Counter[str]] = {}
        self._shown: dict[str, set[str]] = {}

    def apply(
        self,
        intents: Mapping[str, Intent],
        decisions: Sequence[IntentDecision],
        wording: Callable[[Mapping[str, Intent], Question], str],
    ) -> list[IntentDecision]:
        out: list[IntentDecision] = []
        for decision in decisions:
            if decision.action is not Action.ASK:
                out.append(decision)
                continue
            fixed = tuple(self._repair(intents, q, wording) for q in decision.questions)
            out.append(decision.model_copy(update={"questions": fixed}))
        return out

    def forget(self, intent_id: str) -> None:
        self._base.pop(intent_id, None)
        self._shown.pop(intent_id, None)

    def _repair(self, intents: Mapping[str, Intent], q: Question, wording: Callable[[Mapping[str, Intent], Question], str]) -> Question:
        if q.slot in (IDENTITY_SLOT, CONFIRM_SLOT):
            return q
        base = q.text or wording(intents, q)
        counts = self._base.setdefault(q.intent_id, Counter())
        shown = self._shown.setdefault(q.intent_id, set())
        times = counts[base]
        counts[base] += 1
        text = base if times == 0 else numbered(base) if times == 1 else offer_person(base)
        while text in shown and times:
            text += " (I still need an answer to this.)"
        shown.add(text)
        return q.model_copy(update={"text": text})
