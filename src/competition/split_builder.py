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
from src.teacher.program_signature import annotate_example_from_candidates, build_signature_bucket


HARD_TRIAD_FAMILIES = {"bit", "cipher", "equation"}
EASY_TRIAD_FAMILIES = {"numeral", "unit", "gravity"}


def _ensure_teacher_annotations(examples: list[PuzzleExample]) -> list[PuzzleExample]:
    apply_family_tags(examples)
    engine = ChainSearchEngine(beam_width=8, max_depth=2)
    for example in examples:
        if example.metadata.program_signature and example.metadata.teacher_confidence is not None:
            continue
        candidates = engine.solve_example(example, top_k=2)
        annotate_example_from_candidates(example, candidates)
    return examples


def _signature_bucket(example: PuzzleExample) -> str:
    return (
        str(example.metadata.extras.get("program_signature_bucket"))
        if example.metadata.extras.get("program_signature_bucket")
        else build_signature_bucket(example.metadata.program_signature)
        if example.metadata.program_signature
        else example.metadata.composition_key
        or "unknown"
    )


def _rule_group_key(example: PuzzleExample) -> str:
    family = example.metadata.official_family or "unknown"
    subtype = example.metadata.subtype or "unknown"
    if example.metadata.program_signature:
        return f"{family}|{subtype}|{example.metadata.program_signature}"
    return f"{family}|{subtype}|{_signature_bucket(example)}"


def _eligible_items(items: list[PuzzleExample], *, families: set[str] | None = None) -> list[PuzzleExample]:
    return [
        item
        for item in items
        if item.metadata.source != "synthetic"
        and (families is None or item.metadata.official_family in families)
    ]


def _iid_split(items: list[PuzzleExample], *, valid_ratio: float, seed: int) -> tuple[list[str], list[str], dict[str, int]]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * valid_ratio)) if shuffled else 0
    valid_ids = sorted(item.id for item in shuffled[:cut])
    valid_id_set = set(valid_ids)
    train_ids = sorted(item.id for item in items if item.id not in valid_id_set)
    return train_ids, valid_ids, {"iid": len(valid_ids)}


def _group_split(
    items: list[PuzzleExample],
    *,
    valid_ratio: float,
    seed: int,
    key_builder,
    family_balanced: bool = False,
) -> tuple[list[str], list[str], dict[str, int]]:
    train_id_set = {item.id for item in items}
    valid_ids: list[str] = []
    held_out_groups: dict[str, int] = {}

    if family_balanced:
        by_family: dict[str, list[PuzzleExample]] = defaultdict(list)
        for item in items:
            by_family[str(item.metadata.official_family or "unknown")].append(item)
        for family_index, family in enumerate(sorted(by_family)):
            family_items = by_family[family]
            family_groups: dict[str, list[PuzzleExample]] = defaultdict(list)
            for item in family_items:
                family_groups[str(key_builder(item))].append(item)
            rng = random.Random(seed + family_index)
            candidate_groups = list(family_groups.items())
            candidate_groups.sort(key=lambda pair: (len(pair[1]), rng.random()))

            family_valid_target = max(1, int(len(family_items) * valid_ratio)) if family_items else 0
            if len(candidate_groups) <= 1 or (candidate_groups and len(candidate_groups[0][1]) > family_valid_target):
                shuffled = list(family_items)
                rng.shuffle(shuffled)
                for item in shuffled[:family_valid_target]:
                    if item.id in train_id_set:
                        train_id_set.remove(item.id)
                        valid_ids.append(item.id)
                held_out_groups[f"{family}|iid_fallback"] = family_valid_target
                continue
            family_valid_count = 0
            for group_name, group_items in candidate_groups:
                if family_valid_count >= family_valid_target:
                    break
                held_out_groups[group_name] = len(group_items)
                for item in group_items:
                    if item.id in train_id_set:
                        train_id_set.remove(item.id)
                        valid_ids.append(item.id)
                        family_valid_count += 1
    else:
        groups: dict[str, list[PuzzleExample]] = defaultdict(list)
        for item in items:
            groups[str(key_builder(item))].append(item)
        candidate_groups = list(groups.items())
        rng = random.Random(seed)
        candidate_groups.sort(key=lambda pair: (len(pair[1]), rng.random()))
        valid_target = max(1, int(len(items) * valid_ratio)) if items else 0
        for group_name, group_items in candidate_groups:
            if len(valid_ids) >= valid_target:
                break
            held_out_groups[group_name] = len(group_items)
            for item in group_items:
                if item.id in train_id_set:
                    train_id_set.remove(item.id)
                    valid_ids.append(item.id)
    return sorted(train_id_set), sorted(valid_ids), dict(sorted(held_out_groups.items()))


def _stats(items: list[PuzzleExample], valid_ids: set[str], group_stats: dict[str, int]) -> dict[str, object]:
    family_stats: dict[str, int] = defaultdict(int)
    subtype_stats: dict[str, int] = defaultdict(int)
    signature_bucket_stats: dict[str, int] = defaultdict(int)
    for item in items:
        if item.id not in valid_ids:
            continue
        family = item.metadata.official_family or "unknown"
        subtype = item.metadata.subtype or "unknown"
        family_stats[family] += 1
        subtype_stats[f"{family}:{subtype}"] += 1
        signature_bucket_stats[_signature_bucket(item)] += 1
    return {
        "group_stats": dict(sorted(group_stats.items())),
        "family_stats": dict(sorted(family_stats.items())),
        "subtype_stats": dict(sorted(subtype_stats.items())),
        "signature_bucket_stats": dict(sorted(signature_bucket_stats.items())),
    }


def build_splits(
    examples: Iterable[PuzzleExample],
    *,
    rule_novelty_valid_ratio: float,
    hard_triad_valid_ratio: float,
    seed: int,
) -> dict[str, dict[str, object]]:
    items = _ensure_teacher_annotations(list(examples))
    rule_items = _eligible_items(items)
    hard_items = _eligible_items(items, families=HARD_TRIAD_FAMILIES)
    easy_items = _eligible_items(items, families=EASY_TRIAD_FAMILIES)

    payload: dict[str, dict[str, object]] = {}
    for split_name, subset, ratio, split_seed, key_builder in [
        ("iid", items, rule_novelty_valid_ratio, seed, None),
        ("family_holdout_legacy", rule_items, rule_novelty_valid_ratio, seed + 1, lambda item: item.metadata.official_family or "unknown"),
        ("composition_holdout_legacy", rule_items, rule_novelty_valid_ratio, seed + 2, lambda item: item.metadata.composition_key or "unknown"),
        ("rule_novelty_all", rule_items, rule_novelty_valid_ratio, seed + 3, _rule_group_key),
        ("hard_triad_rule_novelty", hard_items, hard_triad_valid_ratio, seed + 4, _rule_group_key),
        ("easy_triad_sanity", easy_items, hard_triad_valid_ratio, seed + 5, _rule_group_key),
    ]:
        if key_builder is None:
            train_ids, valid_ids, group_stats = _iid_split(subset, valid_ratio=ratio, seed=split_seed)
        else:
            train_ids, valid_ids, group_stats = _group_split(
                subset,
                valid_ratio=ratio,
                seed=split_seed,
                key_builder=key_builder,
                family_balanced=split_name in {"rule_novelty_all", "hard_triad_rule_novelty", "easy_triad_sanity"},
            )
        payload[split_name] = {
            "train_ids": train_ids,
            "valid_ids": valid_ids,
            **_stats(items, set(valid_ids), group_stats),
        }

    payload["rule_novelty"] = payload["rule_novelty_all"]
    payload["hard_triad"] = payload["hard_triad_rule_novelty"]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rule-novelty and hard-triad evaluation splits.")
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
