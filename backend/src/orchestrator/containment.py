"""Containment: freeze what a customer reports lost before a plan is made.

A freeze is reversible and needs no yes (``decision.NO_YES_NEEDED``), so the orchestrator makes it as soon as
the customer is verified and has named the cards, and only then asks the plan-making route to work out the
closes and replacements, which do need a yes. Each card is a ``FREEZE_CARD`` intent run through the same direct
recipe as any freeze request: nothing new may be written here.

The orchestrator holds no database, so a ``Containment`` (built where the database is) says which cards.
It returns nothing when it cannot tell, and the plan-making route then asks which cards were meant.
"""

from typing import Protocol

from orchestrator.types import Intent


class Containment(Protocol):
    def freezes(self, intent: Intent, user_id: str, text: str = "") -> tuple[Intent, ...]:
        """One FREEZE_CARD intent per card to freeze now (the verified customer's own, named, ACTIVE). ``text`` is what
        the customer wrote about the report, for the names a decomposer may not have filled in."""
        ...
