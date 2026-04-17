from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import load_jsonl, read_json, write_json, write_jsonl
from src.competition.schema import PuzzleExample
from src.student.sft_dataset_builder import PROMPT_MODE_RAW_WITH_GUARD, build_sft_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Build selected stage-2 SFT records from benchmark outputs.")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", default="data/processed/selected_train.jsonl")
    parser.add_argument("--report", default="data/processed/selected_report.json")
    parser.add_argument("--prompt-mode", default=PROMPT_MODE_RAW_WITH_GUARD)
    parser.add_argument("--trace-style", default="token_trace")
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--max-prompt-chars", type=int, default=8000)
    parser.add_argument("--max-completion-chars", type=int, default=512)
    args = parser.parse_args()

    examples = {example.id: example for example in (PuzzleExample.from_dict(row) for row in load_jsonl(args.examples))}
    benchmark_payload = read_json(args.benchmark)
    benchmark_rows = benchmark_payload.get("records", benchmark_payload.get("rows", []))

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    dedupe_keys: set[tuple[str, object, object]] = set()
    kept: list[dict[str, object]] = []
    for row in benchmark_rows:
        if not row.get("competition_correct", False):
            continue
        if not row.get("boxed_valid", False):
            continue
        if float(row.get("teacher_confidence", 0.0) or 0.0) < args.confidence_threshold:
            continue
        example = examples.get(str(row["id"]))
        if example is None:
            continue
        example.metadata.extras = {**dict(example.metadata.extras), "source_dataset": "selected"}
        record = build_sft_record(
            example,
            stage="stage2",
            prompt_mode=args.prompt_mode,
            trace_style=args.trace_style,
            include_metadata=True,
        )
        if len(record["prompt"]) > args.max_prompt_chars or len(record["completion"]) > args.max_completion_chars:
            continue
        dedupe_key = (example.query, example.target_answer, record.get("program_signature"))
        if dedupe_key in dedupe_keys:
            continue
        dedupe_keys.add(dedupe_key)
        grouped[str(record.get("official_family", "unknown"))].append(record)

    family_order = sorted(grouped, key=lambda family: (0 if family in {"cipher", "bit", "equation"} else 1, family))
    family_indices = {family: 0 for family in family_order}
    while True:
        made_progress = False
        for family in family_order:
            bucket = grouped[family]
            index = family_indices[family]
            if index >= len(bucket):
                continue
            kept.append(bucket[index])
            family_indices[family] += 1
            made_progress = True
        if not made_progress:
            break

    report = {
        "num_records": len(kept),
        "family_counts": dict(sorted(Counter(str(row.get("official_family", "unknown")) for row in kept).items())),
        "subtype_counts": dict(sorted(Counter(f"{row.get('official_family')}:{row.get('subtype')}" for row in kept).items())),
        "hard_triad_ratio": 0.0 if not kept else sum(str(row.get("official_family")) in {"cipher", "bit", "equation"} for row in kept) / len(kept),
    }
    write_jsonl(args.output, kept)
    write_json(args.report, report)


if __name__ == "__main__":
    main()
