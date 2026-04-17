from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.common.io import load_jsonl, write_json
from src.competition.metrics import competition_numeric_match, exact_match
from src.competition.parser import parse_competition_file
from src.competition.schema import PuzzleExample
from src.teacher.chain_search import ChainSearchEngine
from src.teacher.family_tagger import apply_family_tags
from src.teacher.program_signature import annotate_example_from_candidates


def _load_examples(input_path: str) -> list[PuzzleExample]:
    path = Path(input_path)
    if path.suffix.lower() == ".jsonl":
        return [PuzzleExample.from_dict(row) for row in load_jsonl(path)]
    return parse_competition_file(path, source="kaggle", split="train")


def _ensure_family_tags(examples: list[PuzzleExample]) -> list[PuzzleExample]:
    missing = [example for example in examples if not example.metadata.official_family or not example.metadata.subtype]
    if not missing:
        return examples
    resolved: dict[str, PuzzleExample] = {example.id: example for example in apply_family_tags(missing)}
    return [
        resolved.get(example.id, example)
        if not example.metadata.official_family or not example.metadata.subtype
        else example
        for example in examples
    ]


def _failure_type(record: dict[str, Any]) -> str:
    if not record.get("prediction"):
        return "no_candidate"
    if record.get("exact"):
        return "exact_match"
    if record.get("numeric"):
        return "numeric_only"
    if record.get("official_family") in {"bit", "cipher", "equation"}:
        return "hard_triad_failure"
    return "wrong_answer"


def benchmark_examples(
    examples: list[PuzzleExample],
    *,
    beam_width: int,
    max_depth: int,
    top_k: int,
    max_per_family: int,
    family_filter: set[str] | None = None,
    failures_only: bool = False,
) -> dict[str, Any]:
    grouped: dict[str, list[PuzzleExample]] = defaultdict(list)
    for example in _ensure_family_tags(examples):
        family = example.metadata.official_family or "unknown"
        if family_filter and family not in family_filter:
            continue
        if len(grouped[family]) < max_per_family:
            grouped[family].append(example)

    engine = ChainSearchEngine(beam_width=beam_width, max_depth=max_depth)
    family_reports: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for family, family_examples in grouped.items():
        correct = 0
        family_rows: list[dict[str, Any]] = []
        for example in family_examples:
            candidates = engine.solve_example(example, top_k=top_k)
            annotate_example_from_candidates(example, candidates)
            best = candidates[0] if candidates else None
            prediction = "" if best is None or best.query_prediction is None else best.query_prediction
            exact = exact_match(prediction, example.target_answer or "")
            numeric = competition_numeric_match(prediction, example.target_answer)
            competition_correct = exact or numeric
            if competition_correct:
                correct += 1
            row = {
                "id": example.id,
                "official_family": family,
                "subtype": example.metadata.subtype,
                "prediction": prediction,
                "target": example.target_answer or "",
                "exact": exact,
                "numeric": numeric,
                "competition_correct": competition_correct,
                "boxed_valid": bool(prediction),
                "teacher_confidence": example.metadata.teacher_confidence,
                "program_signature": example.metadata.program_signature,
                "steps": [] if best is None else [step.op_name for step in best.steps],
                "failure_type": "exact_match",
                "debug": None if best is None else best.to_debug_dict(),
            }
            row["failure_type"] = _failure_type(row)
            family_rows.append(row)
            records.append(row)
        if failures_only:
            family_rows = [row for row in family_rows if not row["competition_correct"]]
        family_reports[family] = {
            "num_examples": len(family_examples),
            "correct": correct,
            "accuracy": 0.0 if not family_examples else correct / len(family_examples),
            "rows": family_rows,
        }

    if failures_only:
        records = [row for row in records if not row["competition_correct"]]

    family_wise_competition_correct = {
        family: payload["accuracy"]
        for family, payload in sorted(family_reports.items())
    }
    num_examples = sum(payload["num_examples"] for payload in family_reports.values())
    competition_correct_rate = (
        0.0
        if not num_examples
        else sum(payload["correct"] for payload in family_reports.values()) / num_examples
    )
    return {
        "num_examples": num_examples,
        "competition_correct_rate": competition_correct_rate,
        "family_wise_competition_correct": family_wise_competition_correct,
        "records": records,
        "by_family": family_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark heuristic teacher coverage on an offline dataset subset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/teacher_benchmark.json")
    parser.add_argument("--max-per-family", type=int, default=50)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--family-filter", default="")
    parser.add_argument("--failures-only", action="store_true")
    args = parser.parse_args()

    family_filter = {
        token.strip()
        for token in args.family_filter.split(",")
        if token.strip()
    } or None
    payload = benchmark_examples(
        _load_examples(args.input),
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        top_k=args.top_k,
        max_per_family=args.max_per_family,
        family_filter=family_filter,
        failures_only=args.failures_only,
    )
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
