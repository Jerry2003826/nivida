from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.common.io import load_jsonl, read_yaml, write_json, write_jsonl
from src.competition.prompt_templates import build_competition_prompt
from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair
from src.teacher.op_catalog import build_default_catalog


_EQUATION_RE = re.compile(r"^\s*(\d+)(\D)(\d+)\s*$")


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
    alphabet = "".join(sorted(set(stripped))) or "abcd"
    return "".join(rng.choice(alphabet) for _ in stripped)


def _example_signature(example: PuzzleExample) -> tuple[Any, ...]:
    return (
        tuple((pair.input, pair.output) for pair in example.train_pairs),
        example.query_input,
        example.target_answer,
    )


def _chainable_ops_for_family(family: str) -> list[str]:
    mapping = {
        "reverse_reorder": {"reverse_string", "rotate_left", "rotate_right", "sort_chars"},
        "substitution_cipher": {"caesar_shift"},
        "cipher": {"vocabulary_cipher"},
        "arithmetic_equation": {"add_constant", "multiply_constant", "affine_transform"},
        "unit_conversion": {"scale_measurement"},
        "gravity": {"gravity_distance"},
        "bit_manipulation": {"binary_invert", "reverse_bits", "swap_nibbles", "binary_rotate_left", "binary_rotate_right"},
    }
    return sorted(mapping.get(family, set()))


def _build_single_op_example(
    op_chain: list[Any],
    *,
    index: int,
    family: str,
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
        for _ in range(24):
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
    while len(support_inputs) < 3 and attempts < 24:
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

    query_input = _sample_input_like(seed_input, rng)
    attempts = 0
    while attempts < 24:
        attempts += 1
        if query_input not in support_inputs:
            try:
                query_target = query_input
                for chain_op, chain_params_item in zip(op_chain, chain_params):
                    query_target = chain_op.apply(query_target, chain_params_item)
                break
            except Exception:
                pass
        query_input = _sample_input_like(seed_input, rng)
    else:
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

    metadata = PuzzleMetadata(
        source="synthetic",
        split="synthetic",
        family_tags=[family],
        family_scores={family: 1.0},
        difficulty=0.35 + 0.15 * (len(op_chain) - 1),
        extras={
            "source": "synthetic",
            "family": family,
            "op_chain": [op.name for op in op_chain],
            "chain_length": len(op_chain),
            "hard_negative": False,
        },
    )
    if rng.random() < hard_negative_ratio:
        metadata.extras["hard_negative"] = True
        metadata.extras["negative_answer"] = query_target[::-1] if len(query_target) > 1 else f"{query_target}0"
    example = PuzzleExample(
        id=f"synth_{index:05d}",
        raw_prompt="",
        train_pairs=pairs,
        query_input=query_input,
        target_answer=query_target,
        metadata=metadata,
    )
    example.raw_prompt = build_competition_prompt(example)
    return example


def generate_synthetic_examples(
    *,
    num_samples: int,
    family_weights: dict[str, float],
    max_chain_length: int,
    hard_negative_ratio: float,
    dedupe_against_real: str | None,
    seed: int = 42,
) -> tuple[list[PuzzleExample], dict[str, Any]]:
    rng = random.Random(seed)
    ops = build_default_catalog()
    family_ops: dict[str, list[Any]] = {}
    for op in ops:
        family_ops.setdefault(getattr(op, "family", "unknown"), []).append(op)

    dedupe_signatures: set[tuple[Any, ...]] = set()
    if dedupe_against_real and Path(dedupe_against_real).exists():
        for row in load_jsonl(dedupe_against_real):
            dedupe_signatures.add(_example_signature(PuzzleExample.from_dict(row)))

    examples: list[PuzzleExample] = []
    attempts = 0
    skipped_duplicates = 0
    skipped_generation_failures = 0
    generation_failure_families: Counter[str] = Counter()
    while len(examples) < num_samples and attempts < num_samples * 24:
        attempts += 1
        family = _weighted_choice(rng, family_weights)
        chainable_names = set(_chainable_ops_for_family(family))
        candidates = [op for op in family_ops.get(family, []) if not chainable_names or op.name in chainable_names]
        if not candidates:
            candidates = family_ops.get(family, [])
        if not candidates:
            skipped_generation_failures += 1
            generation_failure_families[family] += 1
            continue
        desired_chain_length = 1 if max_chain_length <= 1 or not chainable_names else rng.randint(1, max_chain_length)
        op_chain = [rng.choice(candidates)]
        while len(op_chain) < desired_chain_length:
            chain_candidates = [op for op in candidates if op.name in chainable_names]
            if not chain_candidates:
                break
            op_chain.append(rng.choice(chain_candidates))
        example = _build_single_op_example(
            op_chain,
            index=len(examples),
            family=family,
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

    family_balance = Counter(example.metadata.family_tags[0] for example in examples if example.metadata.family_tags)
    difficulty_distribution = Counter(str(example.metadata.difficulty) for example in examples)
    chain_length_distribution = Counter(int(example.metadata.extras.get("chain_length", 1)) for example in examples)
    hard_negative_count = sum(bool(example.metadata.extras.get("hard_negative")) for example in examples)
    summary = {
        "num_examples": len(examples),
        "requested_examples": num_samples,
        "family_balance": dict(sorted(family_balance.items())),
        "difficulty_distribution": dict(sorted(difficulty_distribution.items())),
        "chain_length_distribution": dict(sorted(chain_length_distribution.items())),
        "hard_negative_ratio": 0.0 if not examples else hard_negative_count / len(examples),
        "dedupe_rate": 0.0 if attempts == 0 else skipped_duplicates / attempts,
        "skipped_duplicates": skipped_duplicates,
        "skipped_generation_failures": skipped_generation_failures,
        "generation_failure_families": dict(sorted(generation_failure_families.items())),
    }
    return examples, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic teacher dataset.")
    parser.add_argument("--config", default="configs/synth.yaml")
    args = parser.parse_args()
    config = read_yaml(args.config)
    family_weights = config.get("family_weights") or {
        family: 1.0
        for family in config.get("families", ["reverse_reorder", "substitution_cipher", "base_conversion", "arithmetic_equation"])
    }
    examples, summary = generate_synthetic_examples(
        num_samples=int(config.get("num_samples", 128)),
        family_weights={str(key): float(value) for key, value in family_weights.items()},
        max_chain_length=int(config.get("max_chain_length", 1)),
        hard_negative_ratio=float(config.get("hard_negative_ratio", 0.0)),
        dedupe_against_real=config.get("dedupe_against_real"),
        seed=int(config.get("seed", 42)),
    )
    output_path = config.get("output_path", "data/synthetic/synth.jsonl")
    summary_path = config.get("summary_path", str(Path(output_path).with_name("synth_summary.json")))
    write_jsonl(output_path, [example.to_dict() for example in examples])
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()
