"""Strips secrets from text before it is logged or sent to any outside service.

PINs and CVVs are caught by the words around them ("my pin is 4821", "cvv: 123"),
card numbers by their shape (13 to 19 digits, spaces or dashes allowed). Amounts,
dates and ids such as ``dbc_lj82d4f1a9_bluest`` are left alone.
"""

import re
from collections.abc import Callable

from banking_tools.session import REDACTED

_SECRET_WORD = r"(?:pin|cvv2?|cvc|security\s+code)"
# Words that can sit between "pin" and the digits: "pin is", "pin to", "pin should be".
_LINKS = r"(?:(?:is|was|are|be|to|as|number|code|should|will|set|changed)\b|=|:)"
_WORD_THEN_DIGITS = re.compile(
    rf"\b({_SECRET_WORD})\b((?:\W{{0,12}}?{_LINKS}){{0,4}}\W{{0,12}}?)(\d{{3,8}})\b",
    re.IGNORECASE,
)
_DIGITS_THEN_WORD = re.compile(
    rf"\b(\d{{3,8}})\b(\s*(?:is|as)?\s*(?:my|the)?\s*{_SECRET_WORD}\b)", re.IGNORECASE
)
_CARD_NUMBER = re.compile(r"(?<![\w.])\d(?:[ -]?\d){12,18}(?![\w])")


def replace_secrets(text: str, replace: Callable[[str, str], str]) -> str:
    """Swap every PIN, CVV and card number for ``replace(kind, value)``.

    ``kind`` is "pin", "cvv" or "card_number"; ``value`` is the digits that were found.
    ``redact`` blanks them. The secret vault instead swaps each for a reference the model
    can pass around without ever seeing the value.
    """

    def word_then_digits(m: re.Match[str]) -> str:
        kind = "pin" if m.group(1).lower() == "pin" else "cvv"
        return f"{m.group(1)}{m.group(2)}{replace(kind, m.group(3))}"

    def digits_then_word(m: re.Match[str]) -> str:
        kind = "pin" if "pin" in m.group(2).lower() else "cvv"
        return f"{replace(kind, m.group(1))}{m.group(2)}"

    text = _WORD_THEN_DIGITS.sub(word_then_digits, text)
    text = _DIGITS_THEN_WORD.sub(digits_then_word, text)
    return _CARD_NUMBER.sub(lambda m: replace("card_number", m.group(0)), text)


def redact(text: str) -> str:
    return replace_secrets(text, lambda kind, value: REDACTED)
