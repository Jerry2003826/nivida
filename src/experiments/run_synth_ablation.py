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
    examples = generate_synthetic_examples(int(config.get("num_samples", 128)), seed=int(config.get("seed", 42)))
    family_counts = Counter(tag for example in examples for tag in example.metadata.family_tags)
    write_json(args.output, {"num_examples": len(examples), "family_counts": dict(family_counts)})


if __name__ == "__main__":
    main()
