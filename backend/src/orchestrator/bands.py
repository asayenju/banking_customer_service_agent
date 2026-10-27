"""Turns an intent's confidence into proceed / ask / escalate.

The thresholds are placeholders until shadow-mode data comparing Jev to the
rule-based decider exists (docs/prd-orchestrator-routing.md), so they live here
as the only place to change. A band is necessary, not sufficient: an
irreversible action still needs the customer's explicit yes at any confidence.
"""

from enum import StrEnum

PROCEED_AT = 0.85
ASK_AT = 0.65


class Band(StrEnum):
    PROCEED = "proceed"
    ASK = "ask"
    ESCALATE = "escalate"


def band(confidence: float) -> Band:
    # Written so that NaN, which fails every comparison, also escalates.
    if not 0.0 <= confidence <= 1.0:
        return Band.ESCALATE
    if confidence >= PROCEED_AT:
        return Band.PROCEED
    if confidence >= ASK_AT:
        return Band.ASK
    return Band.ESCALATE
