from __future__ import annotations

import json
from typing import Any, Iterable

from src.competition.metrics import competition_numeric_match
from src.competition.schema import PuzzleExample


_FAMILY_ALIASES = {
    "bit_manipulation": "bit",
    "bit_operations": "bit",
    "unit_conversion": "unit",
    "arithmetic_equation": "equation",
}


def normalize_family_alias(family: str | None) -> str:
    if family is None:
        return "unknown"
    lowered = str(family).strip().lower()
    return _FAMILY_ALIASES.get(lowered, lowered)


def _stable_param_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable_param_value(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_stable_param_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_stable_param_value(item) for item in value)
    return value


def canonicalize_step(op_name: str, params: dict[str, Any]) -> str:
    if not params:
        return str(op_name)
    payload = json.dumps(_stable_param_value(params), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return f"{op_name}{payload}"


def canonicalize_program_signature(
    steps: Iterable[Any],
    *,
    family: str | None = None,
    subtype: str | None = None,
) -> str:
    normalized_family = normalize_family_alias(family)
    normalized_subtype = normalize_family_alias(subtype) if subtype else "unknown"
    step_tokens = [
        canonicalize_step(
            getattr(step, "op_name", getattr(step, "name", "unknown")),
            dict(getattr(step, "params", {}) or {}),
        )
        for step in steps
    ]
    return f"family={normalized_family};subtype={normalized_subtype};steps={'|'.join(step_tokens) or 'identity'}"


def annotate_example_from_candidates(example: PuzzleExample, candidates: list[Any]) -> PuzzleExample:
    top = candidates[0] if candidates else None
    second = candidates[1] if len(candidates) > 1 else None
    family = example.metadata.official_family
    subtype = example.metadata.subtype

    support_matches = 0
    solver_verifiable = False
    if top is not None and len(top.predictions) == len(example.parsed_examples):
        for prediction, pair in zip(top.predictions, example.parsed_examples):
            if prediction == pair.output or competition_numeric_match(prediction, pair.output):
                support_matches += 1
        solver_verifiable = support_matches == len(example.parsed_examples) and top.query_prediction is not None

    top_margin = 0.0
    if top is not None and second is not None:
        top_margin = float(top.score) - float(second.score)

    example.metadata.program_signature = None
    example.metadata.teacher_confidence = None if top is None else float(top.confidence)
    if top is not None:
        example.metadata.program_signature = canonicalize_program_signature(
            top.steps,
            family=family,
            subtype=subtype,
        )
    example.metadata.extras = {
        **dict(example.metadata.extras),
        "support_coverage": 0.0 if not example.parsed_examples else support_matches / len(example.parsed_examples),
        "top1_top2_margin": top_margin,
        "solver_verifiable": solver_verifiable,
        "top_candidate_score": None if top is None else float(top.score),
        "top_candidate_steps": [] if top is None else [step.op_name for step in top.steps],
    }
    return example
