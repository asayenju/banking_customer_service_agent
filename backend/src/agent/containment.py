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

_SLOTS = (
    "card_id", "card_ids", "cards", "debit_cards", "card", "card_name", "card_names", "card_type",
    "account_id", "accounts", "account_name", "account_names",
)
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


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def named_in_text(db: Any, user_id: str, text: str) -> list[Card]:
    """The customer's own cards the message itself names, found by code: an account's name ("Light Blue",
    "Green Fee-Free") or a card's last four digits. The longest names are matched first and removed from the text,
    so "Green" is not read out of "Light Green". The decomposer's details are a model's guess at this; the
    customer's words are not."""
    everyone = owned_cards(db, user_id, statuses=("ACTIVE", "FROZEN", "PENDING"))
    by_name: dict[str, list[Card]] = {}
    for card in everyone:
        level = str(db.accounts.data.get(str(card.get("account_id")), {}).get("level") or "")
        name = _plain(re.sub(r"(?i)\baccount\b", "", level))
        if name:
            by_name.setdefault(name, []).append(card)
    remaining = f" {_plain(text)} "
    found: list[Card] = []
    for name in sorted(by_name, key=len, reverse=True):
        phrase = f" {name} "
        if phrase in remaining and len(by_name[name]) == 1:
            found.append(by_name[name][0])
            remaining = remaining.replace(phrase, " ")
    for card in everyone:
        last_four = str(card.get("last_4_digits") or "")
        if last_four and f" {last_four} " in remaining and card not in found:
            found.append(card)
    return found


class WalletContainment:
    def __init__(self, database: Any) -> None:
        self._db = database

    def freezes(self, intent: Intent, user_id: str, text: str = "") -> tuple[Intent, ...]:
        if intent.kind is not IntentKind.REPORT_FRAUD_OR_LOSS:
            return ()
        named = named_cards(self._db, user_id, intent.slots)
        named += [c for c in named_in_text(self._db, user_id, f"{intent.slots.get('said', '')} {text}") if c not in named]
        return tuple(
            Intent(
                id=f"{intent.id}:freeze:{card['card_id']}",
                kind=IntentKind.FREEZE_CARD,
                summary="freeze a card from your lost wallet",
                slots={"card_id": str(card["card_id"])},
            )
            for card in named
            if card.get("status") == "ACTIVE"
        )
