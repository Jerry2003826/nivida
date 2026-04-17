from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.common.io import load_jsonl, read_yaml, write_json, write_jsonl
from src.competition.official_prompts import OFFICIAL_FAMILY_INSTRUCTIONS
from src.competition.prompt_templates import build_competition_prompt
from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair
from src.teacher.op_catalog import build_default_catalog
from src.teacher.program_signature import canonicalize_program_signature


_EQUATION_RE = re.compile(r"^\s*(\d+)(\D)(\d+)\s*$")

_SUBTYPE_OPS: dict[str, dict[str, list[str]]] = {
    "bit": {
        "bit_rotate": ["binary_rotate_left", "binary_rotate_right", "reverse_bits"],
        "bit_xor_mask": ["binary_xor_mask"],
        "bit_nibble": ["swap_nibbles", "binary_nibble_map"],
        "bit_permutation": ["binary_permutation", "swap_nibbles", "reverse_bits"],
        "bit_affine": ["binary_affine_transform", "binary_invert", "binary_and_mask", "binary_or_mask"],
    },
    "gravity": {
        "gravity_inverse_square": ["gravity_distance"],
    },
    "unit": {
        "unit_scale": ["scale_measurement"],
        "unit_convert": ["unit_convert"],
    },
    "cipher": {
        "cipher_char_sub": ["caesar_shift", "fixed_substitution"],
        "cipher_token_sub": ["vocabulary_cipher"],
        "cipher_perm": ["fixed_substitution", "reverse_tokens"],
        "cipher_vocab": ["vocabulary_cipher", "fixed_substitution"],
    },
    "numeral": {
        "numeral_roman": ["decimal_to_roman"],
    },
    "equation": {
        "equation_numeric": ["binary_equation_rule", "add_constant", "multiply_constant", "affine_transform"],
        "equation_delete": ["delete_characters"],
        "equation_position": ["position_transducer"],
        "equation_template": ["operator_template"],
        "equation_symbolic": ["operator_template", "position_transducer", "delete_characters"],
    },
}


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    items = [(name, max(0.0, float(weight))) for name, weight in weights.items()]
    total = sum(weight for _, weight in items)
    if total <= 0:
        return items[0][0]
    threshold = rng.random() * total
    cumulative = 0.0
    for name, weight in items:
        cumulative += weight
        if cumulative >= threshold:
            return name
    return items[-1][0]


def _sample_input_like(seed_text: str, rng: random.Random) -> str:
    stripped = seed_text.strip()
    equation_match = _EQUATION_RE.match(stripped)
    if equation_match:
        left, operator, right = equation_match.groups()
        return f"{rng.randint(0, 10 ** len(left) - 1):0{len(left)}d}{operator}{rng.randint(0, 10 ** len(right) - 1):0{len(right)}d}"
    if stripped and set(stripped) <= {"0", "1"}:
        return "".join(rng.choice("01") for _ in stripped)
    measurement_match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*$", stripped)
    if measurement_match:
        decimals = len(measurement_match.group(1).split(".", 1)[1]) if "." in measurement_match.group(1) else 0
        value = round(rng.uniform(0.5, 50.0), decimals)
        return f"{value:.{decimals}f} {measurement_match.group(2)}"
    if stripped.isdigit():
        width = len(stripped)
        return f"{rng.randint(0, 10 ** width - 1):0{width}d}"
    alphabet = "".join(sorted(set(stripped.replace(" ", "")))) or "abcd"
    if " " in stripped:
        token_count = max(1, len(stripped.split()))
        token_width = max(3, len(stripped.split()[0]))
        return " ".join("".join(rng.choice(alphabet) for _ in range(token_width)) for _ in range(token_count))
    return "".join(rng.choice(alphabet) for _ in stripped)


def _example_signature(example: PuzzleExample) -> tuple[Any, ...]:
    return (
        tuple((pair.input, pair.output) for pair in example.parsed_examples),
        example.query,
        example.target_answer,
    )


def _max_chain_length_for_family(family: str, requested: int) -> int:
    family_cap = 3 if family in {"bit", "cipher"} else 2
    return max(1, min(requested, family_cap))


def _pick_subtype(rng: random.Random, family: str, subtype_weights: dict[str, float] | None = None) -> str:
    subtype_names = sorted(_SUBTYPE_OPS[family])
    if not subtype_weights:
        return subtype_names[rng.randrange(len(subtype_names))]
    weights = {name: float(subtype_weights.get(name, 1.0)) for name in subtype_names}
    return _weighted_choice(rng, weights)


def _build_single_op_example(
    op_chain: list[Any],
    *,
    index: int,
    family: str,
    subtype: str,
    rng: random.Random,
    hard_negative_ratio: float,
) -> PuzzleExample | None:
    if not op_chain:
        return None
    op = op_chain[0]
    try:
        seed_input, _, params = op.generate_random_instance(rng)
    except Exception:
        return None

    chain_params: list[dict[str, Any]] = [params]
    current_probe = seed_input
    try:
        current_probe = op.apply(current_probe, params)
    except Exception:
        return None

    for extra_op in op_chain[1:]:
        extra_params: dict[str, Any] | None = None
        for _ in range(32):
            try:
                _, _, candidate_params = extra_op.generate_random_instance(rng)
                extra_op.apply(current_probe, candidate_params)
                extra_params = candidate_params
                break
            except Exception:
                continue
        if extra_params is None:
            return None
        chain_params.append(extra_params)
        current_probe = extra_op.apply(current_probe, extra_params)

    support_inputs = [seed_input]
    attempts = 0
    while len(support_inputs) < 3 and attempts < 32:
        attempts += 1
        candidate = _sample_input_like(seed_input, rng)
        if candidate in support_inputs:
            continue
        try:
            current_value = candidate
            for chain_op, chain_params_item in zip(op_chain, chain_params):
                current_value = chain_op.apply(current_value, chain_params_item)
        except Exception:
            continue
        support_inputs.append(candidate)

    if len(support_inputs) < 3:
        return None

    query = _sample_input_like(seed_input, rng)
    query_target: str | None = None
    attempts = 0
    while attempts < 32:
        attempts += 1
        if query not in support_inputs:
            try:
                query_target = query
                for chain_op, chain_params_item in zip(op_chain, chain_params):
                    query_target = chain_op.apply(query_target, chain_params_item)
                break
            except Exception:
                pass
        query = _sample_input_like(seed_input, rng)
    if query_target is None:
        return None

    try:
        pairs = []
        for value in support_inputs:
            output_value = value
            for chain_op, chain_params_item in zip(op_chain, chain_params):
                output_value = chain_op.apply(output_value, chain_params_item)
            pairs.append(PuzzlePair(input=value, output=output_value))
    except Exception:
        return None

    signature = canonicalize_program_signature(
        [
            type(
                "SigStep",
                (),
                {"op_name": chain_op.name, "params": params},
            )()
            for chain_op, params in zip(op_chain, chain_params)
        ],
        family=family,
        subtype=subtype,
    )
    metadata = PuzzleMetadata(
        official_family=family,
        subtype=subtype,
        family_scores={family: 1.0},
        teacher_confidence=1.0,
        program_signature=signature,
        difficulty=0.35 + 0.15 * (len(op_chain) - 1),
        source="synthetic",
        split="synthetic",
        extras={
            "source": "synthetic",
            "solver_verifiable": True,
            "support_coverage": 1.0,
            "top1_top2_margin": 1.0,
            "chain_length": len(op_chain),
            "hard_negative": False,
            "top_candidate_steps": [op.name for op in op_chain],
        },
    )
    if rng.random() < hard_negative_ratio:
        metadata.extras["hard_negative"] = True
        metadata.extras["negative_answer"] = query_target[::-1] if len(query_target) > 1 else f"{query_target}0"

    example = PuzzleExample(
        id=f"synth_{index:05d}",
        raw_prompt="",
        official_instruction=OFFICIAL_FAMILY_INSTRUCTIONS[family],
        parsed_examples=pairs,
        query=query,
        target_answer=query_target,
        metadata=metadata,
    )
    example.raw_prompt = build_competition_prompt(example)
    return example


def generate_synthetic_examples(
    *,
    num_samples: int,
    family_weights: dict[str, float],
    subtype_weights: dict[str, float] | None = None,
    max_chain_length: int,
    hard_negative_ratio: float,
    dedupe_against_real: str | None,
    seed: int = 42,
) -> tuple[list[PuzzleExample], dict[str, Any]]:
    rng = random.Random(seed)
    ops = build_default_catalog()
    ops_by_name = {op.name: op for op in ops}

    dedupe_signatures: set[tuple[Any, ...]] = set()
    if dedupe_against_real and Path(dedupe_against_real).exists():
        for row in load_jsonl(dedupe_against_real):
            dedupe_signatures.add(_example_signature(PuzzleExample.from_dict(row)))

    examples: list[PuzzleExample] = []
    attempts = 0
    skipped_duplicates = 0
    skipped_generation_failures = 0
    generation_failure_families: Counter[str] = Counter()

    while len(examples) < num_samples and attempts < num_samples * 32:
        attempts += 1
        family = _weighted_choice(rng, family_weights)
        if family not in _SUBTYPE_OPS:
            skipped_generation_failures += 1
            generation_failure_families[family] += 1
            continue
        subtype = _pick_subtype(rng, family, subtype_weights=subtype_weights)
        candidate_names = _SUBTYPE_OPS[family][subtype]
        candidates = [ops_by_name[name] for name in candidate_names if name in ops_by_name]
        if not candidates:
            skipped_generation_failures += 1
            generation_failure_families[family] += 1
            continue

        desired_chain_length = rng.randint(1, _max_chain_length_for_family(family, max_chain_length))
        op_chain = [rng.choice(candidates)]
        while len(op_chain) < desired_chain_length:
            op_chain.append(rng.choice(candidates))

        example = _build_single_op_example(
            op_chain,
            index=len(examples),
            family=family,
            subtype=subtype,
            rng=rng,
            hard_negative_ratio=hard_negative_ratio,
        )
        if example is None:
            skipped_generation_failures += 1
            generation_failure_families[family] += 1
            continue
        signature = _example_signature(example)
        if signature in dedupe_signatures:
            skipped_duplicates += 1
            continue
        dedupe_signatures.add(signature)
        examples.append(example)

    family_balance = Counter(example.metadata.official_family for example in examples if example.metadata.official_family)
    subtype_balance = Counter(
        f"{example.metadata.official_family}:{example.metadata.subtype}"
        for example in examples
        if example.metadata.official_family and example.metadata.subtype
    )
    chain_length_distribution = Counter(int(example.metadata.extras.get("chain_length", 1)) for example in examples)
    hard_negative_count = sum(bool(example.metadata.extras.get("hard_negative")) for example in examples)
    summary = {
        "num_examples": len(examples),
        "requested_examples": num_samples,
        "family_balance": dict(sorted(family_balance.items())),
        "subtype_balance": dict(sorted(subtype_balance.items())),
        "chain_length_distribution": dict(sorted(chain_length_distribution.items())),
        "hard_negative_ratio": 0.0 if not examples else hard_negative_count / len(examples),
        "dedupe_rate": 0.0 if attempts == 0 else skipped_duplicates / attempts,
        "skipped_duplicates": skipped_duplicates,
        "skipped_generation_failures": skipped_generation_failures,
        "generation_failure_families": dict(sorted(generation_failure_families.items())),
    }
    return examples, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a subtype-aware synthetic teacher dataset.")
    parser.add_argument("--config", default="configs/synth.yaml")
    args = parser.parse_args()
    config = read_yaml(args.config)
    family_weights = config.get("family_weights") or {
        family: 1.0
        for family in config.get("families", ["bit", "cipher", "equation", "unit", "gravity", "numeral"])
    }
    examples, summary = generate_synthetic_examples(
        num_samples=int(config.get("num_samples", 128)),
        family_weights={str(key): float(value) for key, value in family_weights.items()},
        subtype_weights={str(key): float(value) for key, value in dict(config.get("subtype_weights", {})).items()} or None,
        max_chain_length=int(config.get("max_chain_length", 3)),
        hard_negative_ratio=float(config.get("hard_negative_ratio", 0.0)),
        dedupe_against_real=config.get("dedupe_against_real"),
        seed=int(config.get("seed", 42)),
    )
    output_path = config.get("output_path", "data/synthetic/synth_stage2.jsonl")
    summary_path = config.get("summary_path", str(Path(output_path).with_name("synth_summary.json")))
    write_jsonl(output_path, [example.to_dict() for example in examples])
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()
