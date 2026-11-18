"""What a customer reads is plain text.

A model writes markdown out of habit (asterisks for bold, hash marks for headings, backticks). The chat window shows
text as it is, so the customer would see the asterisks. This turns a model's answer into what a person should read:
the words, with the lists kept. Citations in square brackets are left alone: they are a separate decision.
"""

import re

_BOLD = re.compile(r"(\*\*|__)(.+?)\1", re.DOTALL)
_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BULLET = re.compile(r"^(\s*)[*•]\s+", re.MULTILINE)
_BLANKS = re.compile(r"\n{3,}")


def plain_text(text: str) -> str:
    cleaned = _BOLD.sub(r"\2", text)
    cleaned = _ITALIC.sub(r"\1", cleaned)
    cleaned = _HEADING.sub("", cleaned)
    cleaned = _BULLET.sub(r"\1- ", cleaned)
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return _BLANKS.sub("\n\n", cleaned).strip()
