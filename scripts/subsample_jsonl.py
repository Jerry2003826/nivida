from __future__ import annotations

import argparse
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import load_jsonl, write_jsonl


def subsample_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limit: int,
) -> Path:
    if limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    rows = load_jsonl(input_path)
    write_jsonl(output_path, rows[:limit])
    return Path(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the first N rows of a JSONL file to a new JSONL file."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", required=True, type=int)
    args = parser.parse_args()

    subsample_jsonl(args.input, args.output, limit=args.limit)


if __name__ == "__main__":
    main()
