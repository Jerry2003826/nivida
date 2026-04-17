from __future__ import annotations

import argparse

from src.common.io import read_json, write_json
from src.teacher.curriculum import build_curriculum
from src.teacher.hardcase_miner import mine_hard_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard cases from an evaluation artifact and bucket them into a curriculum.")
    parser.add_argument("--input", default="data/processed/baseline_eval.json")
    parser.add_argument("--output", default="data/processed/hard_cases.json")
    parser.add_argument("--max-items", type=int, default=64)
    args = parser.parse_args()

    payload = read_json(args.input)
    hard_cases = mine_hard_cases(payload, max_items=args.max_items)
    curriculum = build_curriculum(hard_cases)
    write_json(
        args.output,
        {
            "num_hard_cases": len(hard_cases),
            "rows": hard_cases,
            "curriculum": curriculum,
        },
    )


if __name__ == "__main__":
    main()
