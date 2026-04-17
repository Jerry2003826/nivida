from __future__ import annotations

import argparse
from collections import Counter

from src.common.io import read_yaml, write_json
from src.teacher.synth_generator import generate_synthetic_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic sample summary for ablation.")
    parser.add_argument("--config", default="configs/synth.yaml")
    parser.add_argument("--output", default="data/synthetic/synth_summary.json")
    args = parser.parse_args()

    config = read_yaml(args.config)
    examples, summary = generate_synthetic_examples(
        num_samples=int(config.get("num_samples", 128)),
        family_weights={str(key): float(value) for key, value in dict(config.get("family_weights", {})).items()},
        max_chain_length=int(config.get("max_chain_length", 3)),
        hard_negative_ratio=float(config.get("hard_negative_ratio", 0.0)),
        dedupe_against_real=config.get("dedupe_against_real"),
        seed=int(config.get("seed", 42)),
    )
    family_counts = Counter(example.metadata.official_family for example in examples if example.metadata.official_family)
    write_json(args.output, {"num_examples": len(examples), "family_counts": dict(family_counts), "summary": summary})


if __name__ == "__main__":
    main()
