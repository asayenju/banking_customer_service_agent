"""The brief of a test customer is shown whole and as plain text, so the person playing them can answer."""

import main
import persona_text


def test_every_test_customer_brief_is_plain_whole_and_free_of_their_record_id():
    for person in main.personas():
        brief = main.persona(person["id"])
        assert brief["sections"], person["id"]
        text = str(brief["sections"])
        assert "**" not in text and "__" not in text, f"markdown marks left in {person['id']}"
        assert "user_id" not in text.lower() and "user id" not in text.lower(), person["id"]
        assert all(s["text"] or s["items"] for s in brief["sections"]), f"an empty section in {person['id']}"


def test_nothing_is_cut_off_the_long_brief_lists_every_charge_to_dispute():
    """Claire Dubois: 16 disputes. The page used to keep the first 700 characters, which stopped inside the first card."""
    brief = main.persona("task_041")
    titles = [s["title"] for s in brief["sections"]]
    assert titles[:4] == ["Who they are", "Their situation", "What they want", "What they know"]
    cards = [s for s in brief["sections"] if s["title"].endswith("Disputes (4 transactions)")]
    assert [s["title"] for s in cards] == [
        "Gold Rewards Card Disputes (4 transactions)",
        "Crypto-Cash Back Card Disputes (4 transactions)",
        "Ecocard Disputes (4 transactions)",
        "Business Gold Rewards Card Disputes (4 transactions)",
    ]
    assert sum(len(s["items"]) for s in cards) == 16
    assert any("Delta Airlines $487.50 on 10/08/2025 - FRAUD" in i for i in cards[0]["items"])
    assert any("Staples $623.45 on 10/14/2025" in i for i in cards[3]["items"])
    assert titles[-1] == "How they talk"


def test_the_brief_keeps_its_lists_and_drops_markdown_and_boilerplate():
    brief = (
        "You are playing the role of a customer contacting a customer service representative agent.\n\n"
        "**Your character:** You are **Ann**, a nurse.\n\n**Your goal:** Dispute one charge.\n\n"
        "**CARD DISPUTES (2 transactions):**\n1. **Delta $5.00 on 10/08/2025** - FRAUD.\n2. Amazon $9.00 - DUPLICATE.\n\n"
        "**Verification info:**\n- phone 1\n\n## Conversation Flow:\n\n1. **Opening:** \"Hi\"\n2. Say thanks."
    )
    out = persona_text.sections(brief, set())
    assert [s["title"] for s in out] == ["Who they are", "What they want", "Card Disputes (2 transactions)", "How they talk"]
    assert out[0]["text"] == "You are Ann, a nurse." and "playing the role" not in str(out)
    assert out[2]["items"] == ["Delta $5.00 on 10/08/2025 - FRAUD.", "Amazon $9.00 - DUPLICATE."]
    assert out[3]["items"] == ['Opening: "Hi"', "Say thanks."]
    assert "phone 1" not in str(out), "the verification block is shown separately by the page"
    assert "user id" not in str(persona_text.sections("**Your goal:** Hi.\n\nThe user id is abc123xyz.", {"abc123xyz"})).lower()
