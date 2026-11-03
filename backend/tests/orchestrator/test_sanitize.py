import pytest

from orchestrator.sanitize import redact


@pytest.mark.parametrize(
    "text, gone",
    [
        ("my pin is 4821", "4821"),
        ("PIN: 4821 please", "4821"),
        ("new pin 123456", "123456"),
        ("the cvv is 123", "123"),
        ("CVV2=987", "987"),
        ("security code 4455", "4455"),
        ("4821 is my pin", "4821"),
        ("card 4111 1111 1111 1111 was stolen", "4111"),
        ("card 4111-1111-1111-1111", "4111"),
        ("number 4111111111111111", "4111"),
        ("please set my new pin to 7351", "7351"),
        ("the pin should be 7351", "7351"),
        ("change the pin to 7351 today", "7351"),
        ("my pin will be changed to 7351", "7351"),
        ("PIN to 7351", "7351"),
        ("set the cvv to 739", "739"),
    ],
)
def test_secrets_are_removed(text: str, gone: str) -> None:
    out = redact(text)
    assert gone not in out
    assert "***" in out


@pytest.mark.parametrize(
    "text",
    [
        "dispute the $200.00 charge from 2024-05-12",
        "freeze dbc_lj82d4f1a9_bluest",
        "I have 3 cards and paid 1500 dollars",
        "call me at 555-0100 about my pinned card",
        "I want to pin this to the top of my statement for 2024",
        "please switch to 2025 statements and set it up to email me",
        "the code for the branch is posted on the wall",
        "",
    ],
)
def test_harmless_text_is_unchanged(text: str) -> None:
    assert redact(text) == text
