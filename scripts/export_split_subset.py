from __future__ import annotations
# ruff: noqa: E402

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import load_jsonl, write_jsonl
from src.competition.schema import PuzzleExample
from src.student.sft_dataset_builder import export_split_subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a split-filtered canonical JSONL subset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--split-role", required=True, choices=["train", "valid"])
    args = parser.parse_args()

    examples = [PuzzleExample.from_dict(row) for row in load_jsonl(args.input)]
    rows = export_split_subset(
        examples,
        split_file=args.split_file,
        split_name=args.split_name,
        split_role=args.split_role,
    )
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
