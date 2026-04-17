from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[。．.!！?？,，;；:：]+$")


def normalise_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalise_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", normalise_unicode(text)).strip()


def strip_trailing_punctuation(text: str) -> str:
    return _TRAILING_PUNCT_RE.sub("", text.strip())


def canonical_text(text: str) -> str:
    text = normalise_unicode(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_for_exact_match(text: str) -> str:
    return strip_trailing_punctuation(normalise_whitespace(text)).replace("−", "-")
