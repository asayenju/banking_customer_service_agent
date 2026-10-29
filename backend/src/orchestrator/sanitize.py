"""Strips secrets from text before it is logged or sent to any outside service.

PINs and CVVs are caught by the words around them ("my pin is 4821", "cvv: 123"),
card numbers by their shape (13 to 19 digits, spaces or dashes allowed). Amounts,
dates and ids such as ``dbc_lj82d4f1a9_bluest`` are left alone.
"""

import re

from banking_tools.session import REDACTED

_SECRET_WORD = r"(?:pin|cvv2?|cvc|security\s+code)"
_WORD_THEN_DIGITS = re.compile(
    rf"\b({_SECRET_WORD})\b(\W{{0,20}}?(?:is|was|=|:|number|code)?\W{{0,6}}?)(\d{{3,8}})\b",
    re.IGNORECASE,
)
_DIGITS_THEN_WORD = re.compile(
    rf"\b(\d{{3,8}})\b(\s*(?:is|as)?\s*(?:my|the)?\s*{_SECRET_WORD}\b)", re.IGNORECASE
)
_CARD_NUMBER = re.compile(r"(?<![\w.])\d(?:[ -]?\d){12,18}(?![\w])")


def redact(text: str) -> str:
    text = _WORD_THEN_DIGITS.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    text = _DIGITS_THEN_WORD.sub(lambda m: f"{REDACTED}{m.group(2)}", text)
    return _CARD_NUMBER.sub(REDACTED, text)
