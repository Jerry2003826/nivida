from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from src.common.io import load_jsonl, read_yaml, write_json
from src.competition.schema import PuzzleExample
from src.teacher.chain_search import ChainSearchEngine
from src.teacher.family_tagger import apply_family_tags
from src.teacher.program_signature import annotate_example_from_candidates


def _ensure_teacher_annotations(examples: list[PuzzleExample]) -> list[PuzzleExample]:
    apply_family_tags(examples)
    engine = ChainSearchEngine(beam_width=8, max_depth=2)
    for example in examples:
        if example.metadata.program_signature and example.metadata.teacher_confidence is not None:
            continue
        candidates = engine.solve_example(example, top_k=2)
        annotate_example_from_candidates(example, candidates)
    return examples


def _group_signature(example: PuzzleExample) -> str:
    family = example.metadata.official_family or "unknown"
    subtype = example.metadata.subtype or "unknown"
    signature = example.metadata.program_signature or "unknown"
    return f"{family}|{subtype}|{signature}"


def _build_rule_novelty_split(items: list[PuzzleExample], *, valid_ratio: float, seed: int) -> dict[str, list[str]]:
    eligible = [
        item
        for item in items
        if item.metadata.source != "synthetic"
        and bool(item.metadata.program_signature)
        and bool(item.metadata.extras.get("solver_verifiable"))
    ]
    groups: dict[str, list[PuzzleExample]] = defaultdict(list)
    for item in eligible:
        groups[_group_signature(item)].append(item)

    candidate_groups = [(name, group_items) for name, group_items in groups.items() if len(group_items) >= 3]
    rng = random.Random(seed)
    rng.shuffle(candidate_groups)

    train_ids = [item.id for item in items]
    valid_ids: list[str] = []
    held_out_groups: list[str] = []
    valid_target = max(1, int(len(eligible) * valid_ratio))

    train_id_set = set(train_ids)
    for group_name, group_items in candidate_groups:
        if len(valid_ids) >= valid_target:
            break
        held_out_groups.append(group_name)
        for item in group_items:
            if item.id in train_id_set:
                train_id_set.remove(item.id)
                valid_ids.append(item.id)

    return {
        "train_ids": sorted(train_id_set),
        "valid_ids": sorted(valid_ids),
        "held_out_groups": sorted(held_out_groups),
    }


def _build_hard_triad_split(items: list[PuzzleExample], *, valid_ratio: float, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    grouped: dict[str, list[PuzzleExample]] = defaultdict(list)
    for item in items:
        if item.metadata.official_family not in {"bit", "cipher", "equation"}:
            continue
        subtype = item.metadata.subtype or "unknown"
        grouped[subtype].append(item)

    valid_ids: list[str] = []
    train_ids = [item.id for item in items]
    train_id_set = set(train_ids)
    held_out_groups: list[str] = []
    for subtype, group_items in grouped.items():
        shuffled = group_items[:]
        rng.shuffle(shuffled)
        cut = max(1, int(len(shuffled) * valid_ratio))
        valid_slice = shuffled[:cut]
        if valid_slice:
            held_out_groups.append(subtype)
        for item in valid_slice:
            if item.id in train_id_set:
                train_id_set.remove(item.id)
                valid_ids.append(item.id)
    return {
        "train_ids": sorted(train_id_set),
        "valid_ids": sorted(valid_ids),
        "held_out_groups": sorted(held_out_groups),
    }


def build_splits(
    examples: Iterable[PuzzleExample],
    *,
    rule_novelty_valid_ratio: float,
    hard_triad_valid_ratio: float,
    seed: int,
) -> dict[str, dict[str, list[str]]]:
    items = _ensure_teacher_annotations(list(examples))
    return {
        "rule_novelty": _build_rule_novelty_split(items, valid_ratio=rule_novelty_valid_ratio, seed=seed),
        "hard_triad": _build_hard_triad_split(items, valid_ratio=hard_triad_valid_ratio, seed=seed + 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rule_novelty and hard_triad splits.")
    parser.add_argument("--input", required=True, help="Processed JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--rule-novelty-valid-ratio", type=float, default=0.15)
    parser.add_argument("--hard-triad-valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", help="Optional YAML config path.")
    args = parser.parse_args()

    if args.config:
        config = read_yaml(args.config)
        args.input = config.get("output_path", args.input)
        args.output = str(Path(config.get("split_output_dir", "data/splits")) / "splits.json")
        split_cfg = config.get("split_strategy", config.get("split", {}))
        args.rule_novelty_valid_ratio = float(split_cfg.get("rule_novelty_valid_ratio", args.rule_novelty_valid_ratio))
        args.hard_triad_valid_ratio = float(split_cfg.get("hard_triad_valid_ratio", args.hard_triad_valid_ratio))
        args.seed = int(config.get("seed", args.seed))

    rows = [PuzzleExample.from_dict(row) for row in load_jsonl(args.input)]
    payload = build_splits(
        rows,
        rule_novelty_valid_ratio=args.rule_novelty_valid_ratio,
        hard_triad_valid_ratio=args.hard_triad_valid_ratio,
        seed=args.seed,
    )
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
