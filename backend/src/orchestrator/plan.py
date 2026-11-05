"""Which intents run together, in what order, and which have to wait.

Takes the intents that were cleared to run this turn and lays them out in
waves. Everything in a wave may run at the same time; a wave starts only when
the one before it has finished.

  - An intent comes after the intents it depends on.
  - A credit limit request goes before a dispute or a replacement in the same turn: the bank
    refuses to approve an increase while a dispute is open or a replacement is on its way, so
    filing those first would make the increase impossible. The order is added unless the customer
    already said the opposite (a cycle is never created).
  - Urgent intents (a fraud or loss report, a card freeze) go first. If any is
    ready, the wave holds only urgent intents, so protection is never queued
    behind a balance lookup.
  - Two writes run together only when each names the resource it touches and
    the two do not overlap. A write with no resource id conflicts with every
    other write, so a missing detail serializes rather than races. Reads never
    conflict with anything.
  - An intent whose dependency is not running waits, and so does everything
    downstream of it.
  - A dependency cycle rejects the whole plan: it means the decomposition is
    corrupt, and guessing an order from it would be worse than handing off.

Known limit: resources are matched by the ids in the intents' slots. A dispute
that names only a transaction id and a freeze that names only a card id look
unrelated here even when the transaction was on that card. Relations like that
have to come from the decomposer, via ``Decomposition.conflicts``, which this
planner honours.
"""

import itertools
from collections.abc import Iterable, Sequence
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from orchestrator.types import Conflict, Intent, IntentKind

URGENT_KINDS = frozenset({IntentKind.REPORT_FRAUD_OR_LOSS, IntentKind.FREEZE_CARD})

# Kinds that only read. Everything else is treated as a write, so a new kind is
# serialized until someone decides otherwise.
READ_ONLY_KINDS = frozenset(
    {
        IntentKind.ACCOUNT_BALANCE_OR_HISTORY,
        IntentKind.LOOKUP_INFO,
        IntentKind.GENERAL_QUESTION,
        IntentKind.UNCLEAR_OR_GREETING,
    }
)


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class Waiting(_Model):
    intent_id: Annotated[str, Field(min_length=1)]
    # The direct dependencies that are not done and not running.
    blocked_by: tuple[str, ...]


class Plan(_Model):
    waves: tuple[tuple[str, ...], ...]
    waiting: tuple[Waiting, ...]


class PlanRejected(_Model):
    reason: str
    intent_ids: tuple[str, ...]


def _resources(intent: Intent) -> frozenset[str]:
    return frozenset(
        value.strip().casefold()
        for name, value in intent.slots.items()
        if name.endswith("_id") and name != "user_id" and value.strip()
    )


def _flagged_pairs(conflicts: Iterable[Conflict]) -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    for conflict in conflicts:
        for a, b in itertools.combinations(conflict.intent_ids, 2):
            pairs.add(frozenset({a, b}))
    return pairs


def _conflict(a: Intent, b: Intent, flagged: set[frozenset[str]]) -> bool:
    if frozenset({a.id, b.id}) in flagged:
        return True
    if a.kind in READ_ONLY_KINDS or b.kind in READ_ONLY_KINDS:
        return False
    mine, theirs = _resources(a), _resources(b)
    if not mine or not theirs:
        return True
    return bool(mine & theirs)


# Work that blocks a credit limit approval for as long as it is open.
BLOCKS_A_LIMIT_INCREASE = frozenset({IntentKind.DISPUTE_TRANSACTION, IntentKind.ORDER_REPLACEMENT_CARD})


# A new account has to exist before the one it replaces is closed: a savings account needs a checking
# account that has been open 14 days, and a business account is refused once any account has been closed
# (docs/accounts-policy-decisions.md, section 7). Closing first would make the opening impossible.
MUST_FOLLOW_AN_OPENING = frozenset({IntentKind.CLOSE_ACCOUNT})


def _order_first(
    deps: dict[str, tuple[str, ...]],
    intents: Sequence[Intent],
    firsts: Iterable[Intent],
    laters: Iterable[Intent],
) -> None:
    """Make every later intent wait for every first intent, except where the customer already said the
    opposite: an order that would create a cycle is never added."""
    ids = set(deps)

    def waits_for(start: str) -> set[str]:
        seen: set[str] = set()
        stack = [d for d in deps[start] if d in ids]
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(d for d in deps[current] if d in ids)
        return seen

    firsts = list(firsts)
    for later in laters:
        for first in firsts:
            if first.id != later.id and first.id not in deps[later.id] and later.id not in waits_for(first.id):
                deps[later.id] = (*deps[later.id], first.id)


def dependencies(intents: Sequence[Intent]) -> dict[str, tuple[str, ...]]:
    """What each intent waits for: what the decomposer said, plus the limit-first and open-before-close rules."""
    deps: dict[str, tuple[str, ...]] = {i.id: tuple(i.depends_on) for i in intents}
    _order_first(
        deps,
        intents,
        (i for i in intents if i.kind is IntentKind.CREDIT_LIMIT_REQUEST),
        (i for i in intents if i.kind in BLOCKS_A_LIMIT_INCREASE),
    )
    _order_first(
        deps,
        intents,
        (i for i in intents if i.kind is IntentKind.OPEN_ACCOUNT),
        (i for i in intents if i.kind in MUST_FOLLOW_AN_OPENING),
    )
    return deps


def closes_wait_for_openings(intents: Sequence[Intent]) -> list[Intent]:
    """The same intents, with each account closing made to depend on the openings in the turn.

    The wave order alone puts the opening first. Writing it into ``depends_on`` as well means a closing is
    skipped when its opening did not go through (it needs an answer, was refused or failed), so the
    customer is never left having closed the old account without the new one. Credit limits get only
    the order: a denied limit request is a normal outcome and the dispute still goes ahead."""
    effective = dependencies(intents)
    return [
        i.model_copy(update={"depends_on": effective[i.id]}) if i.kind in MUST_FOLLOW_AN_OPENING else i
        for i in intents
    ]


def _on_a_cycle(intents: Sequence[Intent]) -> tuple[str, ...]:
    """Ids left over once everything that can be ordered has been, minus anything
    that merely sits downstream of a cycle. Empty when there is no cycle."""
    ids = {i.id for i in intents}
    effective = dependencies(intents)
    deps = {i.id: {d for d in effective[i.id] if d in ids} for i in intents}
    resolved: set[str] = set()
    progress = True
    while progress:
        progress = False
        for intent_id, needs in deps.items():
            if intent_id not in resolved and needs <= resolved:
                resolved.add(intent_id)
                progress = True
    stuck = {i for i in deps if i not in resolved}
    trimming = True
    while trimming:
        trimming = False
        for intent_id in sorted(stuck):
            # Counts the intent itself, so a self-dependency stays a cycle.
            if not any(intent_id in deps[other] for other in stuck):
                stuck.discard(intent_id)
                trimming = True
    return tuple(i.id for i in intents if i.id in stuck)


def plan(
    intents: Sequence[Intent],
    *,
    done: frozenset[str] = frozenset(),
    conflicts: Sequence[Conflict] = (),
) -> Plan | PlanRejected:
    cycle = _on_a_cycle(intents)
    if cycle:
        return PlanRejected(reason="dependency cycle", intent_ids=cycle)

    ids = {i.id for i in intents}
    needs = dependencies(intents)
    blocked: dict[str, tuple[str, ...]] = {}
    changed = True
    while changed:
        changed = False
        for intent in intents:
            if intent.id in blocked:
                continue
            unmet = tuple(
                d for d in needs[intent.id] if d not in done and (d not in ids or d in blocked)
            )
            if unmet:
                blocked[intent.id] = unmet
                changed = True
    waiting = tuple(Waiting(intent_id=i.id, blocked_by=blocked[i.id]) for i in intents if i.id in blocked)

    flagged = _flagged_pairs(conflicts)
    remaining = [i for i in intents if i.id not in blocked]
    finished = set(done)
    waves: list[tuple[str, ...]] = []
    while remaining:
        ready = [i for i in remaining if all(d in finished for d in needs[i.id])]
        if not ready:
            # Unreachable once cycles are rejected; here so a bug can never hang a turn.
            return PlanRejected(
                reason="could not order the remaining intents",
                intent_ids=tuple(i.id for i in remaining),
            )
        candidates = [i for i in ready if i.kind in URGENT_KINDS] or ready
        wave: list[Intent] = []
        for candidate in candidates:
            if not any(_conflict(candidate, member, flagged) for member in wave):
                wave.append(candidate)
        wave_ids = {i.id for i in wave}
        waves.append(tuple(i.id for i in wave))
        finished |= wave_ids
        remaining = [i for i in remaining if i.id not in wave_ids]
    return Plan(waves=tuple(waves), waiting=waiting)
