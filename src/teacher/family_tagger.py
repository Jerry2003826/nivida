from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.competition.official_prompts import detect_official_family
from src.competition.schema import PuzzleExample
from src.teacher.feature_extractor import extract_example_features, extract_pair_features


OFFICIAL_FAMILIES = ["bit", "gravity", "unit", "cipher", "numeral", "equation"]

_UNIT_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*$")
_NUMERIC_EQUATION_RE = re.compile(r"^\s*\d+\D\d+\s*$")
_ROMAN_RE = re.compile(r"^[IVXLCDM]+$", flags=re.IGNORECASE)


@dataclass(slots=True)
class FamilyPrediction:
    official_family: str
    subtype: str
    family_tags: list[str]
    composition_key: str
    confidence: float
    scores: dict[str, float]
    evidence: dict[str, Any]


def _is_binary(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and set(stripped) <= {"0", "1"}


def _infer_from_examples(example: PuzzleExample) -> tuple[str, dict[str, float]]:
    scores = {family: 0.0 for family in OFFICIAL_FAMILIES}
    pairs = example.parsed_examples
    if not pairs:
        return "equation", scores

    if all(_is_binary(pair.input) and _is_binary(pair.output) for pair in pairs):
        scores["bit"] += 1.0
    if all(_UNIT_RE.match(pair.input) for pair in pairs):
        scores["unit"] += 1.0
    if all(pair.input.replace(".", "", 1).isdigit() and pair.output.replace(".", "", 1).isdigit() for pair in pairs):
        scores["gravity"] += 0.5
    if all(_ROMAN_RE.fullmatch(pair.output) for pair in pairs if pair.output):
        scores["numeral"] += 1.0
    if all(_NUMERIC_EQUATION_RE.match(pair.input) for pair in pairs):
        scores["equation"] += 0.8
    if any(" " in pair.input or pair.input.isalpha() for pair in pairs):
        scores["cipher"] += 0.5

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0], scores


def _classify_unit(example: PuzzleExample) -> str:
    output_has_unit = any(_UNIT_RE.match(pair.output) for pair in example.parsed_examples)
    return "unit_convert" if output_has_unit else "unit_scale"


def _classify_numeral(example: PuzzleExample) -> str:
    outputs = [pair.output for pair in example.parsed_examples]
    return "numeral_roman" if outputs and all(_ROMAN_RE.fullmatch(output) for output in outputs) else "numeral_other"


def _classify_equation(example: PuzzleExample) -> str:
    values = [pair.input for pair in example.parsed_examples]
    if example.query:
        values.append(example.query)
    numeric = [_NUMERIC_EQUATION_RE.match(value.strip()) is not None for value in values if value.strip()]
    if numeric and all(numeric):
        return "equation_numeric"
    if example.parsed_examples and all(
        pair.output == "".join(char for char in pair.input if char.isalnum())
        for pair in example.parsed_examples
    ):
        return "equation_delete"
    if example.parsed_examples and all(len(pair.output) <= len(pair.input) for pair in example.parsed_examples):
        return "equation_position"
    if example.parsed_examples and any(any(not char.isalnum() for char in pair.input) for pair in example.parsed_examples):
        return "equation_template"
    return "equation_symbolic"


def _classify_cipher(example: PuzzleExample) -> str:
    pair_features = [extract_pair_features(pair.input, pair.output) for pair in example.parsed_examples]
    if pair_features and all(feature.consistent_char_map and feature.input_length == feature.output_length for feature in pair_features):
        return "cipher_char_sub"
    if all(len(pair.input.split()) == len(pair.output.split()) and len(pair.input.split()) > 1 for pair in example.parsed_examples):
        return "cipher_token_sub"
    if any(feature.same_char_multiset for feature in pair_features):
        return "cipher_perm"
    lower_prompt = example.raw_prompt.lower()
    if "vocab" in lower_prompt or "dictionary" in lower_prompt:
        return "cipher_vocab"
    if "caesar" in lower_prompt or "shift" in lower_prompt:
        return "cipher_char_sub"
    return "cipher_vocab"


def _rotate_matches(lhs: str, rhs: str) -> bool:
    if len(lhs) != len(rhs) or not lhs:
        return False
    for shift in range(1, len(lhs)):
        if lhs[shift:] + lhs[:shift] == rhs or lhs[-shift:] + lhs[:-shift] == rhs:
            return True
    return False


def _constant_mask_family(example: PuzzleExample) -> str | None:
    pairs = example.parsed_examples
    if not pairs or not all(_is_binary(pair.input) and _is_binary(pair.output) for pair in pairs):
        return None
    values = [(int(pair.input, 2), int(pair.output, 2)) for pair in pairs]
    xor_masks = {src ^ dst for src, dst in values}
    if len(xor_masks) == 1:
        return "bit_xor_mask"
    return None


def _classify_bit(example: PuzzleExample) -> str:
    pairs = example.parsed_examples
    if pairs and all(_rotate_matches(pair.input, pair.output) for pair in pairs):
        return "bit_rotate"
    if pairs and all(pair.input[4:] + pair.input[:4] == pair.output for pair in pairs if len(pair.input) == 8):
        return "bit_nibble"
    mask_family = _constant_mask_family(example)
    if mask_family:
        return mask_family
    if len(pairs) >= 3 and any(pair.input != pair.output for pair in pairs):
        return "bit_permutation"
    return "bit_affine"


def _classify_subtype(example: PuzzleExample, family: str) -> str:
    if family == "bit":
        return _classify_bit(example)
    if family == "gravity":
        return "gravity_inverse_square"
    if family == "unit":
        return _classify_unit(example)
    if family == "cipher":
        return _classify_cipher(example)
    if family == "numeral":
        return _classify_numeral(example)
    return _classify_equation(example)


def _pattern_shape(example: PuzzleExample) -> str:
    if example.parsed_examples and all(_is_binary(pair.input) for pair in example.parsed_examples):
        return "binary"
    if example.parsed_examples and all(_UNIT_RE.match(pair.input) for pair in example.parsed_examples):
        return "measurement"
    if example.parsed_examples and all(_NUMERIC_EQUATION_RE.match(pair.input) for pair in example.parsed_examples):
        return "numeric_equation"
    if example.parsed_examples and all(pair.input.isdigit() for pair in example.parsed_examples):
        return "numeric"
    return "symbolic"


def tag_example(example: PuzzleExample) -> FamilyPrediction:
    scores = {family: 0.0 for family in OFFICIAL_FAMILIES}
    official_family = example.metadata.official_family or detect_official_family(example.raw_prompt)
    if official_family:
        scores[official_family] = 1.0
    else:
        official_family, inferred_scores = _infer_from_examples(example)
        for family, score in inferred_scores.items():
            scores[family] = max(scores[family], score)

    subtype = _classify_subtype(example, official_family)
    program_bucket = (
        example.metadata.extras.get("program_signature_bucket")
        or example.metadata.extras.get("signature_bucket")
    )
    composition_key = (
        f"{official_family}|{subtype}|{program_bucket}"
        if program_bucket
        else f"{official_family}|{subtype}|{_pattern_shape(example)}"
    )
    confidence = max(scores.values()) if any(scores.values()) else 0.0
    if confidence <= 0:
        confidence = 0.5
    return FamilyPrediction(
        official_family=official_family,
        subtype=subtype,
        family_tags=[official_family, subtype],
        composition_key=composition_key,
        confidence=min(1.0, confidence),
        scores=scores,
        evidence={
            "official_family": official_family,
            "subtype": subtype,
            "num_examples": len(example.parsed_examples),
        },
    )


def apply_family_tags(examples: list[PuzzleExample]) -> list[PuzzleExample]:
    for example in examples:
        prediction = tag_example(example)
        example.metadata.official_family = prediction.official_family
        example.metadata.subtype = prediction.subtype
        example.metadata.family_tags = prediction.family_tags
        example.metadata.family_scores = prediction.scores
        example.metadata.teacher_confidence = prediction.confidence
        example.metadata.composition_key = prediction.composition_key
        example.metadata.extras = {
            **dict(example.metadata.extras),
            "official_family": prediction.official_family,
            "subtype": prediction.subtype,
            "composition_key": prediction.composition_key,
        }
        if not example.official_instruction:
            example.official_instruction = example.raw_prompt.split("\n\n", 1)[0].strip()
    return examples


def build_family_report(examples: list[PuzzleExample]) -> dict[str, Any]:
    examples = apply_family_tags(examples)
    feature_rows = [extract_example_features(example) for example in examples]
    family_counts = Counter(example.metadata.official_family or "unknown" for example in examples)
    subtype_counts = Counter(
        f"{example.metadata.official_family}:{example.metadata.subtype}"
        for example in examples
        if example.metadata.official_family and example.metadata.subtype
    )
    mean_confidence = 0.0
    if examples:
        mean_confidence = sum(example.metadata.teacher_confidence or 0.0 for example in examples) / len(examples)
    return {
        "num_examples": len(examples),
        "family_counts": dict(sorted(family_counts.items())),
        "subtype_counts": dict(sorted(subtype_counts.items())),
        "avg_teacher_confidence": mean_confidence,
        "avg_length_delta": sum(row["avg_length_delta"] for row in feature_rows) / max(1, len(feature_rows)),
        "avg_char_jaccard": sum(row["avg_char_jaccard"] for row in feature_rows) / max(1, len(feature_rows)),
        "avg_token_jaccard": sum(row["avg_token_jaccard"] for row in feature_rows) / max(1, len(feature_rows)),
        "rows": [
            {
                "id": example.id,
                "official_family": example.metadata.official_family,
                "subtype": example.metadata.subtype,
                "family_scores": example.metadata.family_scores,
                "teacher_confidence": example.metadata.teacher_confidence,
                "avg_length_delta": features["avg_length_delta"],
                "avg_char_jaccard": features["avg_char_jaccard"],
                "avg_token_jaccard": features["avg_token_jaccard"],
            }
            for example, features in zip(examples, feature_rows)
        ],
    }
