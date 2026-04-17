from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

from src.competition.schema import PuzzleExample


_UNIT_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*$")
_EQUATION_RE = re.compile(r"[=+\-*/()]")


def _tokenize(text: str) -> list[str]:
    text = text.strip()
    return text.split() if " " in text else list(text)


def _jaccard(lhs: set[str], rhs: set[str]) -> float:
    if not lhs and not rhs:
        return 1.0
    return len(lhs & rhs) / max(1, len(lhs | rhs))


def _safe_int(text: str) -> int | None:
    try:
        if text.lower().startswith(("0x", "-0x", "+0x")):
            return int(text, 16)
        if text.lower().startswith(("0b", "-0b", "+0b")):
            return int(text, 2)
        if text.lower().startswith(("0o", "-0o", "+0o")):
            return int(text, 8)
        return int(text)
    except ValueError:
        return None


def _same_multiset(lhs: str, rhs: str) -> bool:
    return Counter(lhs) == Counter(rhs)


def _is_reverse(lhs: str, rhs: str) -> bool:
    return lhs[::-1] == rhs


def _is_token_reverse(lhs: str, rhs: str) -> bool:
    return " ".join(_tokenize(lhs)[::-1]) == rhs


def _consistent_char_map(lhs: str, rhs: str) -> bool:
    if len(lhs) != len(rhs):
        return False
    if not any(char.isalpha() for char in lhs + rhs):
        return False
    forward: dict[str, str] = {}
    backward: dict[str, str] = {}
    for src, dst in zip(lhs, rhs):
        if forward.setdefault(src, dst) != dst:
            return False
        if backward.setdefault(dst, src) != src:
            return False
    return any(src != dst for src, dst in zip(lhs, rhs))


def _possible_base_conversion(lhs: str, rhs: str) -> bool:
    source = _safe_int(lhs)
    target = _safe_int(rhs)
    if source is None:
        return False
    return rhs.lower() in {format(source, "b"), format(source, "o"), format(source, "x")} or target == source


def _possible_unit_conversion(lhs: str, rhs: str) -> bool:
    left_match = _UNIT_RE.match(lhs)
    right_match = _UNIT_RE.match(rhs)
    return bool(left_match and right_match and left_match.group(2).lower() != right_match.group(2).lower())


def _possible_counting(lhs: str, rhs: str) -> bool:
    right_value = _safe_int(rhs)
    if right_value is None:
        return False
    token_count = len(_tokenize(lhs))
    options = {
        len(lhs),
        len(lhs.replace(" ", "")),
        token_count,
        sum(char.isdigit() for char in lhs),
        sum(char.isalpha() for char in lhs),
        len(set(lhs.replace(" ", ""))),
    }
    return right_value in options


@dataclass(slots=True)
class PairFeatures:
    input_length: int
    output_length: int
    length_delta: int
    char_jaccard: float
    token_jaccard: float
    sequence_similarity: float
    same_char_multiset: bool
    is_reverse: bool
    is_token_reverse: bool
    consistent_char_map: bool
    possible_base_conversion: bool
    possible_unit_conversion: bool
    possible_counting: bool
    equation_like: bool
    numeric_input: bool
    numeric_output: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_pair_features(input_text: str, output_text: str) -> PairFeatures:
    input_tokens = _tokenize(input_text)
    output_tokens = _tokenize(output_text)
    input_chars = set(input_text)
    output_chars = set(output_text)
    return PairFeatures(
        input_length=len(input_text),
        output_length=len(output_text),
        length_delta=len(output_text) - len(input_text),
        char_jaccard=_jaccard(input_chars, output_chars),
        token_jaccard=_jaccard(set(input_tokens), set(output_tokens)),
        sequence_similarity=SequenceMatcher(None, input_text, output_text).ratio(),
        same_char_multiset=_same_multiset(input_text, output_text),
        is_reverse=_is_reverse(input_text, output_text),
        is_token_reverse=_is_token_reverse(input_text, output_text),
        consistent_char_map=_consistent_char_map(input_text, output_text),
        possible_base_conversion=_possible_base_conversion(input_text, output_text),
        possible_unit_conversion=_possible_unit_conversion(input_text, output_text),
        possible_counting=_possible_counting(input_text, output_text),
        equation_like=bool(_EQUATION_RE.search(input_text)),
        numeric_input=_safe_int(input_text) is not None,
        numeric_output=_safe_int(output_text) is not None,
    )


def extract_example_features(example: PuzzleExample) -> dict[str, Any]:
    pair_features = [extract_pair_features(pair.input, pair.output) for pair in example.train_pairs]
    if not pair_features:
        return {
            "id": example.id,
            "num_pairs": 0,
            "avg_length_delta": 0.0,
            "avg_char_jaccard": 0.0,
            "avg_token_jaccard": 0.0,
            "avg_sequence_similarity": 0.0,
            "same_multiset_rate": 0.0,
            "pair_features": [],
        }

    num_pairs = len(pair_features)
    return {
        "id": example.id,
        "num_pairs": num_pairs,
        "avg_length_delta": sum(item.length_delta for item in pair_features) / num_pairs,
        "avg_char_jaccard": sum(item.char_jaccard for item in pair_features) / num_pairs,
        "avg_token_jaccard": sum(item.token_jaccard for item in pair_features) / num_pairs,
        "avg_sequence_similarity": sum(item.sequence_similarity for item in pair_features) / num_pairs,
        "same_multiset_rate": sum(item.same_char_multiset for item in pair_features) / num_pairs,
        "pair_features": [item.to_dict() for item in pair_features],
    }
