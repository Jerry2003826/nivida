from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from src.competition.schema import PuzzleExample
from src.teacher.feature_extractor import extract_example_features, extract_pair_features


FAMILY_LABELS = [
    "bit_manipulation",
    "gravity",
    "unit_conversion",
    "cipher",
    "numeral",
    "equation",
    "reverse_reorder",
    "substitution_cipher",
    "base_conversion",
    "bit_operations",
    "arithmetic_equation",
    "count_filter_aggregation",
    "multi_step_composition",
]

OFFICIAL_PROMPT_FAMILY_PATTERNS = {
    "bit manipulation rule transforms 8-bit binary numbers": "bit_manipulation",
    "gravitational constant has been secretly changed": "gravity",
    "secret unit conversion is applied to measurements": "unit_conversion",
    "secret encryption rules are used on text": "cipher",
    "numbers are secretly converted into a different numeral system": "numeral",
    "secret set of transformation rules is applied to equations": "equation",
}


@dataclass(slots=True)
class FamilyPrediction:
    families: list[str]
    confidence: float
    scores: dict[str, float]
    evidence: dict[str, Any]


def detect_official_prompt_family(prompt: str) -> str | None:
    lower = prompt.lower()
    for pattern, family in OFFICIAL_PROMPT_FAMILY_PATTERNS.items():
        if pattern in lower:
            return family
    return None


def _score_single_pair(input_text: str, output_text: str) -> dict[str, float]:
    features = extract_pair_features(input_text, output_text)
    scores = defaultdict(float)

    if features.is_reverse or features.is_token_reverse or features.same_char_multiset:
        scores["reverse_reorder"] += 1.0 if features.is_reverse or features.is_token_reverse else 0.6
    if features.consistent_char_map and not features.is_reverse and not features.same_char_multiset:
        scores["substitution_cipher"] += 1.0
    if features.possible_base_conversion:
        scores["base_conversion"] += 1.0
    if features.possible_unit_conversion:
        scores["unit_conversion"] += 1.0
    if features.possible_counting:
        scores["count_filter_aggregation"] += 1.0
    if features.equation_like or (features.numeric_input and features.numeric_output):
        scores["arithmetic_equation"] += 0.7
    if input_text.strip() and set(input_text.strip()) <= {"0", "1"} and set(output_text.strip()) <= {"0", "1"}:
        scores["bit_operations"] += 0.8
    if len([name for name, value in scores.items() if value >= 0.8]) >= 2:
        scores["multi_step_composition"] += 0.7

    return scores


def tag_example(example: PuzzleExample) -> FamilyPrediction:
    aggregate = defaultdict(float)
    evidence_rows: list[dict[str, Any]] = []
    official_family = detect_official_prompt_family(example.raw_prompt)
    if official_family is not None:
        aggregate[official_family] += 2.0
    for pair in example.train_pairs:
        scores = _score_single_pair(pair.input, pair.output)
        aggregate.update({name: aggregate[name] + value for name, value in scores.items()})
        evidence_rows.append({"input": pair.input, "output": pair.output, "scores": dict(scores)})

    if not example.train_pairs:
        return FamilyPrediction(
            families=["multi_step_composition"],
            confidence=0.0,
            scores={"multi_step_composition": 0.0},
            evidence={"pairs": []},
        )

    normalized = {family: aggregate.get(family, 0.0) / len(example.train_pairs) for family in FAMILY_LABELS}
    ranked = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
    best_score = ranked[0][1]
    selected = [name for name, score in ranked if score >= max(0.45, best_score * 0.7) and score > 0]
    if len(selected) >= 2 and "multi_step_composition" not in selected:
        normalized["multi_step_composition"] = max(normalized["multi_step_composition"], min(0.99, best_score))
        selected.append("multi_step_composition")
    if official_family is not None:
        selected = [official_family] + [name for name in selected if name != official_family]
    if not selected:
        selected = ["multi_step_composition"]
    confidence = best_score / max(1e-6, sum(normalized.values()) or 1.0)
    return FamilyPrediction(
        families=selected,
        confidence=min(1.0, confidence),
        scores=dict(normalized),
        evidence={"pairs": evidence_rows, "features": extract_example_features(example), "official_family": official_family},
    )


def apply_family_tags(examples: list[PuzzleExample]) -> list[PuzzleExample]:
    for example in examples:
        prediction = tag_example(example)
        example.metadata.family_tags = prediction.families
        example.metadata.family_scores = prediction.scores
        example.metadata.composition_key = "+".join(sorted(prediction.families))
    return examples


def build_family_report(examples: list[PuzzleExample]) -> dict[str, Any]:
    examples = apply_family_tags(examples)
    feature_rows = [extract_example_features(example) for example in examples]
    counts = Counter(tag for example in examples for tag in example.metadata.family_tags)
    mean_confidence = 0.0
    if examples:
        mean_confidence = sum(max(example.metadata.family_scores.values(), default=0.0) for example in examples) / len(examples)
    return {
        "num_examples": len(examples),
        "family_counts": dict(counts),
        "avg_top_family_confidence": mean_confidence,
        "avg_length_delta": sum(row["avg_length_delta"] for row in feature_rows) / max(1, len(feature_rows)),
        "avg_char_jaccard": sum(row["avg_char_jaccard"] for row in feature_rows) / max(1, len(feature_rows)),
        "avg_token_jaccard": sum(row["avg_token_jaccard"] for row in feature_rows) / max(1, len(feature_rows)),
        "rows": [
            {
                "id": example.id,
                "family_tags": example.metadata.family_tags,
                "family_scores": example.metadata.family_scores,
                "avg_length_delta": features["avg_length_delta"],
                "avg_char_jaccard": features["avg_char_jaccard"],
                "avg_token_jaccard": features["avg_token_jaccard"],
            }
            for example, features in zip(examples, feature_rows)
        ],
    }
