"""Turns an intent's confidence into proceed / ask / escalate.

The thresholds are placeholders until shadow-mode data comparing Jev to the
rule-based decider exists (docs/prd-orchestrator-routing.md), so they live here
as the only place to change. A band is necessary, not sufficient: an
irreversible action still needs the customer's explicit yes at any confidence.
"""

from enum import StrEnum

PROCEED_AT = 0.85
ASK_AT = 0.65
# A write whose kind the scorer is not sure of is confirmed with the customer first.
# It was 0.95 (on 96 labelled action intents Jev's kind answer at 0.95 caught 60% of the
# wrong ones while 2% of the right ones were asked; at 0.85 it caught 40%). A hedged but
# real report ("I think I might have lost my wallet") scored 0.92 every time and was asked
# about in a loop, so the bar is 0.80. That is below PROCEED_AT, which already needs the kind
# answer at 0.85, so the extra gate for writes no longer binds: the plan the customer must
# say yes to (every irreversible step) is what protects them.
WRITE_KIND_ASK_AT = 0.80


class Risk(StrEnum):
    """What a wrong run costs. The bar for acting follows it: a read changes nothing, a freeze can be undone, anything
    else needs the customer's yes to an exact plan, and a loss report must not wait on a hesitant score because the
    named cards are frozen first."""

    READ = "read"
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"
    URGENT = "urgent"


# (proceed at, ask at) per class. Placeholders until the decision log (docs/prd-decision-layer.md) is calibrated.
# Below the ask line the customer is asked what they mean, not handed to a person.
BANDS: dict[Risk, tuple[float, float]] = {
    Risk.READ: (0.70, 0.40),
    Risk.REVERSIBLE: (0.80, 0.55),
    Risk.IRREVERSIBLE: (PROCEED_AT, ASK_AT),
    Risk.URGENT: (0.70, 0.40),
}


class Band(StrEnum):
    PROCEED = "proceed"
    ASK = "ask"
    ESCALATE = "escalate"


def band(confidence: float, risk: Risk = Risk.IRREVERSIBLE) -> Band:
    # Written so that NaN, which fails every comparison, also escalates.
    if not 0.0 <= confidence <= 1.0:
        return Band.ESCALATE
    proceed, ask = BANDS[risk]
    if confidence >= proceed:
        return Band.PROCEED
    if confidence >= ask:
        return Band.ASK
    return Band.ESCALATE
