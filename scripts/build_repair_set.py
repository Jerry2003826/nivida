from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import load_jsonl, read_json, write_json, write_jsonl
from src.competition.schema import PuzzleExample
from src.student.sft_dataset_builder import PROMPT_MODE_RAW_WITH_GUARD, build_sft_record
from src.teacher.error_taxonomy import HARD_TRIAD_FAMILIES, classify_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage-3 repair SFT records from benchmark failures.")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", default="data/processed/repair_train.jsonl")
    parser.add_argument("--report", default="data/processed/repair_report.json")
    parser.add_argument("--prompt-mode", default=PROMPT_MODE_RAW_WITH_GUARD)
    parser.add_argument("--trace-style", default="short_trace")
    args = parser.parse_args()

    examples = {example.id: example for example in (PuzzleExample.from_dict(row) for row in load_jsonl(args.examples))}
    benchmark_payload = read_json(args.benchmark)
    benchmark_rows = benchmark_payload.get("records", benchmark_payload.get("rows", []))

    records = []
    for row in benchmark_rows:
        if row.get("competition_correct", False):
            continue
        example = examples.get(str(row["id"]))
        if example is None:
            continue
        error_type = classify_error(
            {
                **row,
                "official_family": example.metadata.official_family,
                "subtype": example.metadata.subtype,
                "program_signature": example.metadata.program_signature,
            }
        )
        example.metadata.source = "repair"
        example.metadata.extras = {
            **dict(example.metadata.extras),
            "error_type": error_type,
            "predicted_signature": row.get("predicted_signature"),
            "target_signature": example.metadata.program_signature,
            "source_dataset": "repair",
        }
        record = build_sft_record(
            example,
            stage="stage3",
            prompt_mode=args.prompt_mode,
            trace_style=args.trace_style,
            include_metadata=True,
        )
        record["error_type"] = error_type
        record["predicted_signature"] = row.get("predicted_signature")
        record["target_signature"] = example.metadata.program_signature
        records.append(record)

    records.sort(
        key=lambda row: (
            0 if str(row.get("official_family")) in HARD_TRIAD_FAMILIES else 1,
            str(row.get("error_type", "")),
            str(row.get("id", "")),
        )
    )
    report = {
        "num_records": len(records),
        "error_type_counts": dict(sorted(Counter(str(row.get("error_type", "unknown")) for row in records).items())),
        "hard_triad_ratio": 0.0 if not records else sum(str(row.get("official_family")) in HARD_TRIAD_FAMILIES for row in records) / len(records),
    }
    write_jsonl(args.output, records)
    write_json(args.report, report)


if __name__ == "__main__":
    main()
