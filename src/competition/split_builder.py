from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

from src.common.io import load_jsonl, read_yaml, write_json
from src.competition.schema import PuzzleExample


def _random_split(items: list[PuzzleExample], valid_ratio: float, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * valid_ratio))
    return {
        "train_ids": [item.id for item in shuffled[cut:]],
        "valid_ids": [item.id for item in shuffled[:cut]],
    }


def _group_holdout(
    items: list[PuzzleExample],
    group_key: Callable[[PuzzleExample], str],
    holdout_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    groups: dict[str, list[PuzzleExample]] = defaultdict(list)
    for item in items:
        groups[group_key(item)].append(item)

    rng = random.Random(seed)
    group_names = list(groups)
    rng.shuffle(group_names)
    holdout_groups: set[str] = set()
    holdout_target = max(1, int(len(group_names) * holdout_ratio))
    for name in group_names:
        holdout_groups.add(name)
        if len(holdout_groups) >= holdout_target:
            break

    train_ids: list[str] = []
    valid_ids: list[str] = []
    for name, group_items in groups.items():
        target = valid_ids if name in holdout_groups else train_ids
        target.extend(item.id for item in group_items)

    return {"train_ids": train_ids, "valid_ids": valid_ids, "held_out_groups": sorted(holdout_groups)}


def build_splits(
    examples: Iterable[PuzzleExample],
    *,
    valid_ratio: float,
    family_holdout_ratio: float,
    composition_holdout_ratio: float,
    seed: int,
) -> dict[str, dict[str, list[str]]]:
    items = list(examples)
    return {
        "iid": _random_split(items, valid_ratio=valid_ratio, seed=seed),
        "family_holdout": _group_holdout(
            items,
            group_key=lambda item: ",".join(item.metadata.family_tags) or "unknown",
            holdout_ratio=family_holdout_ratio,
            seed=seed + 1,
        ),
        "composition_holdout": _group_holdout(
            items,
            group_key=lambda item: item.metadata.composition_key or ",".join(item.metadata.family_tags) or "unknown",
            holdout_ratio=composition_holdout_ratio,
            seed=seed + 2,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IID/family/composition holdout splits.")
    parser.add_argument("--input", required=True, help="Processed JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--family-holdout-ratio", type=float, default=0.34)
    parser.add_argument("--composition-holdout-ratio", type=float, default=0.34)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", help="Optional YAML config path.")
    args = parser.parse_args()

    if args.config:
        config = read_yaml(args.config)
        args.input = config.get("output_path", args.input)
        args.output = str(Path(config.get("split_output_dir", "data/splits")) / "splits.json")
        split_cfg = config.get("split_strategy", config.get("split", {}))
        args.valid_ratio = float(split_cfg.get("valid_ratio", args.valid_ratio))
        args.family_holdout_ratio = float(split_cfg.get("family_holdout_ratio", args.family_holdout_ratio))
        args.composition_holdout_ratio = float(split_cfg.get("composition_holdout_ratio", args.composition_holdout_ratio))
        args.seed = int(config.get("seed", args.seed))

    rows = [PuzzleExample.from_dict(row) for row in load_jsonl(args.input)]
    payload = build_splits(
        rows,
        valid_ratio=args.valid_ratio,
        family_holdout_ratio=args.family_holdout_ratio,
        composition_holdout_ratio=args.composition_holdout_ratio,
        seed=args.seed,
    )
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
