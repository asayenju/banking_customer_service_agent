"""Slice 1 of the fees specialist: every value in the policy tables is the value the document says.

The tables in ``agent/workflows/fee_products.py`` are hand-curated (Decision 3 of the PRD), so a
value cannot be trusted unless a test reads the document it names. If a document changes, or a
value was copied wrong, a test here fails.
"""

import json
import re
from pathlib import Path

import pytest

from agent.workflows import fee_products as P

DOCS = Path(__file__).resolve().parents[3] / "tau2-bench" / "data" / "tau2" / "domains" / "banking_knowledge" / "documents"


def text(doc: str) -> str:
    return json.loads((DOCS / f"doc_{doc}.json").read_text())["content"]


def pct(value: float) -> str:
    return f"{value:g}"


def mentions(body: str, *needles: str) -> bool:
    flat = body.replace("’", "'")
    return all(n.lower() in flat.lower() for n in needles)


def test_every_named_document_exists() -> None:
    names = set(P.CARD_BONUS_DOC.values())
    names |= {doc for fees in P.ATM_FEES.values() for rule in (fees.out_of_network, fees.foreign) for doc in rule.docs}
    names |= {doc for _, doc in P.CHECKING_BOOST.values()}
    names |= {doc for _, doc in P.SAVINGS_BASE.values()}
    assert not [n for n in names if not (DOCS / f"doc_{n}.json").exists()]


def test_there_are_exactly_the_eighteen_pairings_of_document_012() -> None:
    body = text("bank_accounts_bank_accounts_(general)_012")
    listed = re.findall(r"^\d+\. (.+?) \(checking\) \+ (.+?) \(savings\)$", body, flags=re.M)
    assert len(listed) == 18
    assert set(P.CHECKING_BOOST) == {(c, s) for c, s in listed}


@pytest.mark.parametrize("pair", sorted(P.CHECKING_BOOST))
def test_each_checking_boost_is_in_its_document(pair: tuple[str, str]) -> None:
    value, doc = P.CHECKING_BOOST[pair]
    checking, savings = pair
    body = text(doc)
    savings_word = savings.removesuffix(" Account")
    pattern = re.compile(rf"{re.escape(savings_word)}[^\n]*?\+?{re.escape(pct(value))}%|{re.escape(pct(value))}%[^\n]*?{re.escape(savings_word)}", re.I)
    assert pattern.search(body), f"{checking} + {savings}: {value}% not found in {doc}"


@pytest.mark.parametrize("level", sorted(P.CARD_BONUS))
def test_each_card_bonus_table_matches_its_document(level: str) -> None:
    body = text(P.CARD_BONUS_DOC[level])
    # The Bronze table is spread over two documents; Silver's Platinum and Diamond rows are in document 009.
    extra = {"Bronze Account": ["savings_accounts_bronze_account_006"], "Silver Account": ["savings_accounts_silver_account_009"]}
    for doc in extra.get(level, []):
        body += "\n" + text(doc)
    if level == "Gold Account":
        body += "\n" + text("savings_accounts_gold_account_013")
    for card, value in P.CARD_BONUS[level].items():
        word = card.replace("Crypto-Cash Back Card", "Crypto-Cash Back").replace(" Card", "")
        found = re.search(rf"{re.escape(word)}[^\n]*?\+\s?{re.escape(pct(value))}\s?%", body, re.I)
        assert found, f"{level}: {card} +{value}% not found in {P.CARD_BONUS_DOC[level]}"


def test_each_card_bonus_table_has_all_eight_cards() -> None:
    for level, table in P.CARD_BONUS.items():
        assert set(table) == set(P.CARD_NAMES), level


@pytest.mark.parametrize("level", sorted(P.SAVINGS_BASE))
def test_base_apy_tiers_are_in_their_document(level: str) -> None:
    tiers, doc = P.SAVINGS_BASE[level]
    body = text(doc) + text(doc.replace("_002", "_001").replace("_013", "_001").replace("_003", "_001"))
    for _, apy in tiers:
        assert f"{apy:g}%" in body or f"{apy:.1f}%" in body, f"{level}: {apy}% not in {doc}"


def test_silver_and_silver_plus_tier_thresholds() -> None:
    assert P.SAVINGS_BASE["Silver Account"][0][1][0] == 10000.0 and "$10,000" in text("savings_accounts_silver_account_003")
    assert P.SAVINGS_BASE["Silver Plus Account"][0][1][0] == 15000.0 and "$15,000" in text("savings_accounts_silver_plus_account_001")


def test_gold_card_bonus_is_the_same_figure_as_the_gold_relationship_bonus() -> None:
    """Document 045 counts the Gold card's 0.025 among the card bonuses that do not stack."""
    assert P.CARD_BONUS["Gold Account"]["Gold Rewards Card"] == P.RELATIONSHIP_BONUS
    assert "Gold Rewards Card: +0.025% APY bonus" in text("bank_accounts_bank_accounts_(general)_045")


def test_stacking_documents_say_only_the_highest_applies() -> None:
    assert "Only the highest applicable credit card" in text("bank_accounts_bank_accounts_(general)_045")
    assert "Only the highest applicable checking account" in text("bank_accounts_bank_accounts_(general)_046")


ATM_EXPECT = {
    # level: (document fragment that must appear for the out-of-network rule, and for the foreign rule)
    "Blue Account": (("1%", "max $3"), ("3%", "$5")),
    "Bluest Account": (("$2",), ("$0",)),
    "Dark Green Account": (("1%", "min $1"), ("2.5%", "max")),
    "Evergreen Account": (("1%", "$2.5"), ("2%", "min $3")),
    "Green Account": (("$3",), ("3%", "$5")),
    "Light Blue Account": (("2 free", "$2.5"), ("2 free", "$4")),
    "Light Green Account": (("4 free", "$1.5"), ("$2.00", "$3.50", "$5.00")),
    "Purple Account": (("2.50",), ("$0",)),
}


@pytest.mark.parametrize("level", sorted(P.ATM_FEES))
def test_each_atm_schedule_is_in_its_documents(level: str) -> None:
    fees = P.ATM_FEES[level]
    out_needles, foreign_needles = ATM_EXPECT[level]
    out_body = " ".join(text(d) for d in fees.out_of_network.docs)
    foreign_body = " ".join(text(d) for d in fees.foreign.docs)
    assert mentions(out_body, *out_needles), f"{level} out-of-network {out_needles}"
    assert mentions(foreign_body, *foreign_needles), f"{level} foreign {foreign_needles}"


def test_atm_rule_fee_arithmetic() -> None:
    blue = P.ATM_FEES["Blue Account"]
    assert blue.out_of_network.fee(300.0) == 3.0 and blue.out_of_network.fee(150.0) == 1.5
    assert blue.foreign.fee(100.0) == 5.0 and blue.foreign.fee(500.0) == 15.0
    light_green = P.ATM_FEES["Light Green Account"].foreign
    assert light_green.fee(100.0) == 2.0 and light_green.fee(100.01) == 3.5 and light_green.fee(300.0) == 3.5
    assert light_green.fee(350.0) == 5.0
    dark = P.ATM_FEES["Dark Green Account"]
    assert dark.out_of_network.fee(50.0) == 1.5 and dark.foreign.fee(250.0) == 6.0 and dark.foreign.fee(80.0) == 2.0
    ever = P.ATM_FEES["Evergreen Account"]
    assert ever.foreign.fee(100.0) == 3.0 and ever.out_of_network.fee(400.0) == 2.5
    assert P.ATM_FEES["Bluest Account"].foreign.fee(400.0) == 0.0


def test_classes_without_a_table_are_not_guessed() -> None:
    assert "Gold Years Account" not in P.ATM_FEES and "Green Fee-Free Account" not in P.ATM_FEES
