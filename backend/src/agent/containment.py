"""Which cards a lost-wallet report names, so the orchestrator can freeze them first.

The decomposer's details for a report are free text ("my Bluest and Light Green cards", a last four, an id).
Each part is settled against the verified customer's own cards in code: by id, by last four digits (when only one
card has them) or by the account's name when that names exactly one card. Another customer's card never matches,
a card that is not ACTIVE is left alone (a frozen one is already safe, a closed one cannot be frozen), and a
part that matches nothing is ignored: the composer then asks which cards were in the wallet.
"""

import re
from collections.abc import Mapping
from typing import Any

from agent.cards import Card, owned_cards, resolve
from orchestrator.types import Intent, IntentKind

_SLOTS = ("card_id", "card_ids", "cards", "debit_cards", "card", "account_id", "accounts", "account_names")
_SPLIT = re.compile(r"\s*(?:,|;|/|&|\band\b|\bplus\b)\s*", re.IGNORECASE)
_NOISE = {"account", "accounts", "card", "cards", "debit", "my", "the", "both", "all", "of"}


def _words(text: str) -> frozenset[str]:
    return frozenset(w for w in re.sub(r"[^a-z0-9]+", " ", text.lower()).split() if w not in _NOISE)


def named_cards(db: Any, user_id: str, slots: Mapping[str, str]) -> list[Card]:
    """The customer's own cards the details name, each once, in the order named."""
    everyone = owned_cards(db, user_id, statuses=("ACTIVE", "FROZEN", "PENDING"))
    found: list[Card] = []
    for slot in _SLOTS:
        for part in _SPLIT.split(slots.get(slot) or ""):
            part = part.strip()
            if not part:
                continue
            card = resolve(part, everyone)
            if card is None:
                wanted = _words(part)
                same = [
                    c
                    for c in everyone
                    if wanted and _words(str(db.accounts.data.get(str(c.get("account_id")), {}).get("level") or "")) == wanted
                ]
                card = same[0] if len(same) == 1 else None
            if card is not None and card not in found:
                found.append(card)
    return found


class WalletContainment:
    def __init__(self, database: Any) -> None:
        self._db = database

    def freezes(self, intent: Intent, user_id: str) -> tuple[Intent, ...]:
        if intent.kind is not IntentKind.REPORT_FRAUD_OR_LOSS:
            return ()
        return tuple(
            Intent(
                id=f"{intent.id}:freeze:{card['card_id']}",
                kind=IntentKind.FREEZE_CARD,
                summary="freeze a card from your lost wallet",
                slots={"card_id": str(card["card_id"])},
            )
            for card in named_cards(self._db, user_id, intent.slots)
            if card.get("status") == "ACTIVE"
        )
