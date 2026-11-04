"""Every credit card type in the bank's data has a tier, or is listed as deliberately without one."""

import json
from pathlib import Path

from agent.workflows.credit_tiers import (
    ENTRY,
    MID,
    PREMIUM,
    TIER_OF_CARD_TYPE,
    UNMAPPED_CARD_TYPES,
    tier_of_card_type,
)
from banking_tools.session import _base_db

DOCS = Path(__file__).resolve().parents[3] / "tau2-bench/data/tau2/domains/banking_knowledge/documents"


def test_every_card_type_in_the_data_is_mapped_or_deliberately_unmapped():
    types = {a["card_type"] for a in _base_db().credit_card_accounts.data.values()}
    assert types - set(TIER_OF_CARD_TYPE) == UNMAPPED_CARD_TYPES & types
    assert not (set(TIER_OF_CARD_TYPE) & UNMAPPED_CARD_TYPES)


def test_the_table_is_the_one_in_the_replacement_documents():
    text = (DOCS / "doc_credit_cards_credit_card_replacements_002.json").read_text()
    content = json.loads(text)["content"]
    for card_type, tier in TIER_OF_CARD_TYPE.items():
        line = next(l for l in content.splitlines() if card_type in l)
        assert {ENTRY: "Entry-tier", MID: "Mid-tier", PREMIUM: "Premium-tier"}[tier] in line, card_type


def test_an_unlisted_card_type_gets_no_tier():
    assert tier_of_card_type("Crypto-Cash Back") is None
    assert tier_of_card_type("Made Up Card") is None
    assert tier_of_card_type("Gold Rewards Card") == PREMIUM
