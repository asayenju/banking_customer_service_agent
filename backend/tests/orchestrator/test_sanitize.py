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
        "",
    ],
)
def test_harmless_text_is_unchanged(text: str) -> None:
    assert redact(text) == text
