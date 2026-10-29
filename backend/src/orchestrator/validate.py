"""Pure checks on what the model said, run after parsing and before anything acts on it.

No network, no model. ``validate`` returns a cleaned ``Decomposition``:

- required details per kind are added to ``missing_slots`` when absent;
- freeze and unfreeze of the same card are recorded as a conflict;
- intents are put in a canonical order (position in the message, then id);
- a request the message clearly contains but no intent covers is flagged
  (``possible_dropped_intent``), as are phrases that try to steer the system
  (``injection_suspected``).

A dependency cycle cannot be repaired, so it raises ``InvalidDecomposition`` and
the decomposer treats the output as bad. Kinds are never changed here, and
nothing in the message can change them: flags only add caution.
"""

import re
from collections.abc import Iterable

from orchestrator.types import Conflict, Decomposition, Flag, Intent, IntentKind

K = IntentKind


class InvalidDecomposition(ValueError):
    pass


# Details an intent needs before a tool can run. Kinds not listed need none.
REQUIRED_SLOTS: dict[IntentKind, tuple[str, ...]] = {
    K.FREEZE_CARD: ("card_id",),
    K.UNFREEZE_CARD: ("card_id",),
    K.ACTIVATE_CARD: ("card_id",),
    K.ORDER_REPLACEMENT_CARD: ("card_id",),
    K.CHANGE_OR_RESET_PIN: ("card_id",),
    K.CLOSE_CARD: ("card_id",),
    K.CARD_CONTROLS: ("card_id",),
    K.DISPUTE_TRANSACTION: ("transaction_id",),
    K.TRANSFER_FUNDS: ("from_account_id", "to_account_id", "amount"),
    K.PAY_CREDIT_CARD: ("account_id", "amount"),
    K.CLOSE_ACCOUNT: ("account_id",),
}

# Phrases that clearly ask for something, and the kinds that would cover them.
_CUES: tuple[tuple[re.Pattern[str], frozenset[IntentKind]], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), frozenset(kinds))
    for pattern, kinds in (
        (r"\b(freeze|lock|block)\b", {K.FREEZE_CARD, K.REPORT_FRAUD_OR_LOSS, K.CARD_CONTROLS}),
        (r"\b(unfreeze|unlock|unblock)\b", {K.UNFREEZE_CARD, K.CARD_CONTROLS}),
        (r"\b(dispute|chargeback)\b", {K.DISPUTE_TRANSACTION, K.REPORT_FRAUD_OR_LOSS}),
        (r"\b(close|cancel)\b", {K.CLOSE_CARD, K.CLOSE_ACCOUNT, K.FREEZE_CARD, K.REPORT_FRAUD_OR_LOSS}),
        (r"\b(transfer|move)\b", {K.TRANSFER_FUNDS, K.PAY_CREDIT_CARD, K.REQUEST_HUMAN}),
        (r"\bbalance\b", {K.ACCOUNT_BALANCE_OR_HISTORY, K.LOOKUP_INFO, K.GENERAL_QUESTION}),
        (r"\bpin\b", {K.CHANGE_OR_RESET_PIN}),
        (r"\b(replace|replacement)\b", {K.ORDER_REPLACEMENT_CARD}),
        (r"\bactivate\b", {K.ACTIVATE_CARD}),
        (r"credit limit", {K.CREDIT_LIMIT_REQUEST}),
        (r"\b(human|real person|representative)\b", {K.REQUEST_HUMAN}),
    )
)

_INJECTION = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|earlier|all)\b.{0,30}\b(instructions?|rules?|prompt)",
        r"\bsystem prompt\b",
        r"\byou are now\b",
        r"\b(developer|admin|debug|god) mode\b",
        r"\b(mark|set|treat)\b.{0,20}\b(me|this|caller)\b.{0,20}\bverified\b",
        r"\bskip\b.{0,20}\b(confirmation|verification|verifying)\b",
        r"\bsupervisor override\b",
        r"\bjailbreak\b",
    )
)


def _cycle(intents: Iterable[Intent]) -> tuple[str, ...]:
    """Ids on a dependency cycle, or empty when there is none."""
    deps = {i.id: i.depends_on for i in intents}
    done: set[str] = set()

    def visit(node: str, path: tuple[str, ...]) -> tuple[str, ...]:
        if node in path:
            return path[path.index(node) :]
        if node in done:
            return ()
        for nxt in deps.get(node, ()):
            found = visit(nxt, (*path, node))
            if found:
                return found
        done.add(node)
        return ()

    for start in deps:
        found = visit(start, ())
        if found:
            return found
    return ()


def _order(intents: tuple[Intent, ...]) -> tuple[Intent, ...]:
    def key(i: Intent) -> tuple[int, int, str]:
        if i.source_span is None:
            return (1, 0, i.id)
        return (0, i.source_span[0], i.id)

    return tuple(sorted(intents, key=key))


_NOT_AN_ID = re.compile(r"^\s*[$€£]|^\s*\d+(\.\d+)?\s*(dollars?|usd|bucks)\b", re.IGNORECASE)


def _without_fake_ids(slots: dict[str, str]) -> dict[str, str]:
    """Drop ``*_id`` slots holding an amount: a model sometimes files "$200" as an id."""
    return {k: v for k, v in slots.items() if not (k.endswith("_id") and _NOT_AN_ID.match(v))}


def _with_required_slots(intent: Intent) -> Intent:
    slots = _without_fake_ids(intent.slots)
    missing = list(intent.missing_slots)
    for slot in REQUIRED_SLOTS.get(intent.kind, ()):
        if not slots.get(slot, "").strip() and slot not in missing:
            missing.append(slot)
    if slots == intent.slots and tuple(missing) == intent.missing_slots:
        return intent
    return intent.model_copy(update={"slots": slots, "missing_slots": tuple(missing)})


def _opposing_card_actions(intents: tuple[Intent, ...]) -> list[Conflict]:
    found: list[Conflict] = []
    freezes = [i for i in intents if i.kind is K.FREEZE_CARD]
    unfreezes = [i for i in intents if i.kind is K.UNFREEZE_CARD]
    for a in freezes:
        for b in unfreezes:
            card_a, card_b = a.slots.get("card_id", ""), b.slots.get("card_id", "")
            if card_a == card_b:  # same card, or neither names one
                found.append(
                    Conflict(intent_ids=(a.id, b.id), reason="freeze and unfreeze of the same card")
                )
    return found


def _flags(message: str, intents: tuple[Intent, ...], existing: tuple[Flag, ...]) -> tuple[Flag, ...]:
    flags = list(existing)
    kinds = {i.kind for i in intents}
    if Flag.INJECTION_SUSPECTED not in flags and any(p.search(message) for p in _INJECTION):
        flags.append(Flag.INJECTION_SUSPECTED)
    if Flag.POSSIBLE_DROPPED_INTENT not in flags and any(
        cue.search(message) and not (kinds & accepted) for cue, accepted in _CUES
    ):
        flags.append(Flag.POSSIBLE_DROPPED_INTENT)
    return tuple(flags)


def validate(message: str, decomposition: Decomposition) -> Decomposition:
    cycle = _cycle(decomposition.intents)
    if cycle:
        raise InvalidDecomposition(f"dependency cycle: {' -> '.join(cycle)}")

    intents = _order(tuple(_with_required_slots(i) for i in decomposition.intents))
    known = {frozenset(c.intent_ids) for c in decomposition.conflicts}
    conflicts = list(decomposition.conflicts)
    for conflict in _opposing_card_actions(intents):
        if frozenset(conflict.intent_ids) not in known:
            conflicts.append(conflict)

    return Decomposition(
        intents=intents,
        conflicts=tuple(conflicts),
        flags=_flags(message, intents, decomposition.flags),
        schema_version=decomposition.schema_version,
    )
