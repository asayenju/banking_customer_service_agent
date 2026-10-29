"""Drop a hand-off the customer did not ask for.

A model sometimes adds REQUEST_HUMAN to a long or urgent message that never
mentions a person. A person is the cautious outcome, so this check leans toward
keeping the intent: it is dropped only when the message holds no word that could
ask for one AND another intent remains, so the customer is never left with
nothing. It is separate from ``validate`` because that module promises never to
change kinds or remove intents.
"""

import re

from orchestrator.types import Decomposition, IntentKind

# Deliberately broad. A false match only keeps a hand-off; a miss would drop one.
_PERSON = re.compile(
    r"\b(human|person|people|someone|somebody|anybody|anyone|agent|representative|rep|"
    r"manager|supervisor|specialist|advisor|adviser|operator|employee|staff|"
    r"(talk|speak|chat|connect|transfer|put)\s+(me\s+)?(to|with|through)|call\s+me|phone\s+me)\b",
    re.IGNORECASE,
)


def drop_unasked_handoff(message: str, decomposition: Decomposition) -> Decomposition:
    if _PERSON.search(message):
        return decomposition
    drop = {i.id for i in decomposition.intents if i.kind is IntentKind.REQUEST_HUMAN}
    if not drop or len(drop) == len(decomposition.intents):
        return decomposition
    intents = tuple(
        i.model_copy(update={"depends_on": tuple(d for d in i.depends_on if d not in drop)})
        for i in decomposition.intents
        if i.id not in drop
    )
    conflicts = tuple(
        c for c in decomposition.conflicts if not drop.intersection(c.intent_ids)
    )
    return decomposition.model_copy(update={"intents": intents, "conflicts": conflicts})
