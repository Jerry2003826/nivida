from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import read_yaml, write_jsonl
from src.competition.parser import parse_competition_file
from src.competition.split_builder import build_splits
from src.teacher.chain_search import ChainSearchEngine
from src.teacher.family_tagger import apply_family_tags
from src.teacher.program_signature import annotate_example_from_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse raw data, tag families, and build splits.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--source")
    parser.add_argument("--dataset-split")
    parser.add_argument("--split-output-dir")
    args = parser.parse_args()

    config = read_yaml(args.config)
    input_path = args.input or config["input_path"]
    output_path = args.output or config["output_path"]
    source = args.source or config.get("source", "kaggle")
    dataset_split = args.dataset_split or config.get("dataset_split", config.get("split", "train"))
    split_output_dir = args.split_output_dir or config.get("split_output_dir", "data/splits")
    examples = parse_competition_file(
        input_path,
        source=source,
        split=dataset_split,
    )
    examples = apply_family_tags(examples)
    engine = ChainSearchEngine(beam_width=8, max_depth=2)
    for example in examples:
        candidates = engine.solve_example(example, top_k=2)
        annotate_example_from_candidates(example, candidates)
    write_jsonl(output_path, [example.to_dict() for example in examples])

    if config.get("build_splits", True):
        split_cfg = config.get("split_strategy", config.get("split", {}))
        split_payload = build_splits(
            examples,
            rule_novelty_valid_ratio=float(split_cfg.get("rule_novelty_valid_ratio", 0.15)),
            hard_triad_valid_ratio=float(split_cfg.get("hard_triad_valid_ratio", 0.15)),
            seed=int(config.get("seed", 42)),
        )
        output_dir = Path(split_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        from src.common.io import write_json

        write_json(output_dir / "splits.json", split_payload)


if __name__ == "__main__":
    main()
