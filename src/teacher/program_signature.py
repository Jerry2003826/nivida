from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from src.competition.metrics import competition_correct, competition_numeric_match
from src.competition.schema import PuzzleExample


_FAMILY_ALIASES = {
    "bit_manipulation": "bit",
    "bit_operations": "bit",
    "unit_conversion": "unit",
    "arithmetic_equation": "equation",
}

_OP_ALIASES = {
    "binary_xor_mask": "xor",
    "binary_affine_transform": "affine",
    "binary_rotate_left": "rotl",
    "binary_rotate_right": "rotr",
    "binary_permutation": "perm",
    "binary_nibble_map": "nibble",
    "swap_nibbles": "swap_nibbles",
    "reverse_bits": "reverse_bits",
    "decimal_to_roman": "dec2roman",
    "scale_measurement": "scale",
    "unit_convert": "convert",
    "gravity_distance": "inverse_square",
    "caesar_shift": "caesar",
    "fixed_substitution": "char_sub",
    "vocabulary_cipher": "vocab_sub",
    "delete_characters": "delete",
    "position_transducer": "position",
    "operator_template": "template",
    "binary_equation_rule": "eq_rule",
    "add_constant": "add",
    "multiply_constant": "mul",
    "affine_transform": "affine",
}

_SIGNATURE_IGNORED_PARAM_KEYS = {
    "template_rank_features",
}


@dataclass(slots=True)
class ProgramSignature:
    official_family: str
    subtype: str | None
    steps: list[str]
    parameters: list[dict[str, Any]]
    depth: int
    signature: str
    signature_bucket: str


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


def parameter_to_token(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, Decimal)):
        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return str(value)
        rendered = format(decimal_value.normalize(), "f")
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".") or "0"
        return rendered
    if isinstance(value, str):
        return value.replace(" ", "_")
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(parameter_to_token(item) for item in value) + "]"
    if isinstance(value, dict):
        return ",".join(
            f"{key}={parameter_to_token(val)}"
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        )
    return str(value)


def canonicalize_step(op_name: str, params: dict[str, Any]) -> str:
    alias = _OP_ALIASES.get(op_name, op_name)
    stable_params = dict(
        sorted(
            (
                (key, _stable_param_value(value))
                for key, value in (params or {}).items()
                if key not in _SIGNATURE_IGNORED_PARAM_KEYS
            ),
            key=lambda item: str(item[0]),
        )
    )
    if not stable_params:
        return alias
    if alias == "xor" and "mask" in stable_params:
        return f"{alias}:{parameter_to_token(stable_params['mask'])}"
    if alias == "scale" and "factor" in stable_params:
        return f"{alias}:{parameter_to_token(stable_params['factor'])}"
    payload = ",".join(f"{key}={parameter_to_token(val)}" for key, val in stable_params.items())
    return f"{alias}:{payload}"


def build_signature_bucket(signature: str) -> str:
    coarse_tokens = []
    for token in signature.split(">"):
        coarse_tokens.append(token.split(":", 1)[0].strip())
    return ">".join(coarse_tokens or ["identity"])


def canonicalize_candidate(candidate: Any, official_family: str, subtype: str | None = None) -> ProgramSignature:
    steps = list(getattr(candidate, "steps", []) or [])
    step_names: list[str] = []
    parameters: list[dict[str, Any]] = []
    tokens: list[str] = []
    for step in steps:
        op_name = getattr(step, "op_name", getattr(step, "name", "unknown"))
        params = dict(getattr(step, "params", {}) or {})
        step_names.append(str(op_name))
        parameters.append(params)
        tokens.append(canonicalize_step(str(op_name), params))
    signature = ">".join(tokens or ["identity"])
    return ProgramSignature(
        official_family=normalize_family_alias(official_family),
        subtype=None if subtype is None else str(subtype),
        steps=step_names,
        parameters=parameters,
        depth=len(step_names),
        signature=signature,
        signature_bucket=build_signature_bucket(signature),
    )


def canonicalize_program_signature(
    steps: Iterable[Any],
    *,
    family: str | None = None,
    subtype: str | None = None,
) -> str:
    candidate = type("SignatureCandidate", (), {"steps": list(steps)})()
    return canonicalize_candidate(candidate, family or "unknown", subtype=subtype).signature


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

    query_prediction = None if top is None else top.query_prediction
    query_solver_correct: bool | None = None
    if query_prediction is not None and example.target_answer not in (None, ""):
        query_solver_correct = competition_correct(str(query_prediction), str(example.target_answer))

    top_margin = 0.0
    if top is not None and second is not None:
        top_margin = float(top.score) - float(second.score)

    example.metadata.program_signature = None
    example.metadata.teacher_confidence = None if top is None else float(top.confidence)
    program: ProgramSignature | None = None
    if top is not None:
        program = canonicalize_candidate(top, family or "unknown", subtype=subtype)
        example.metadata.program_signature = program.signature
        example.metadata.composition_key = (
            f"{program.official_family}|{program.subtype or 'unknown'}|{program.signature_bucket}"
        )
    example.metadata.extras = {
        **dict(example.metadata.extras),
        "support_coverage": 0.0 if not example.parsed_examples else support_matches / len(example.parsed_examples),
        "top1_top2_margin": top_margin,
        "solver_verifiable": solver_verifiable,
        "query_prediction": None if query_prediction is None else str(query_prediction),
        "query_solver_correct": query_solver_correct,
        "program_signature_bucket": None if program is None else program.signature_bucket,
        "top_candidate_score": None if top is None else float(top.score),
        "top_candidate_steps": [] if top is None else [step.op_name for step in top.steps],
    }
    return example
