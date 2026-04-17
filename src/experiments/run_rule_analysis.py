from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.io import load_jsonl, read_yaml, write_json
from src.competition.parser import parse_competition_file
from src.competition.schema import PuzzleExample
from src.teacher.family_tagger import build_family_report


def _load_examples(input_path: str) -> list[PuzzleExample]:
    path = Path(input_path)
    if path.suffix.lower() == ".jsonl":
        return [PuzzleExample.from_dict(row) for row in load_jsonl(path)]
    return parse_competition_file(path, source="kaggle", split="train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run family tagging analysis and export a report.")
    parser.add_argument("--config")
    parser.add_argument("--input")
    parser.add_argument("--output-dir", default="data/processed/family_report")
    args = parser.parse_args()

    if args.config:
        config = read_yaml(args.config)
        args.input = config.get("output_path", config.get("input_path", args.input))
        args.output_dir = config.get("report_output_dir", args.output_dir)

    examples = _load_examples(args.input)
    report = build_family_report(examples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "family_report.json", report)
    pd.DataFrame(report["rows"]).to_csv(output_dir / "family_rows.csv", index=False)


if __name__ == "__main__":
    main()
