from __future__ import annotations

import argparse
from pathlib import Path

from src.common.io import load_jsonl, read_yaml, write_json
from src.competition.metrics import evaluate_predictions
from src.competition.parser import parse_competition_file
from src.competition.schema import PuzzleExample
from src.student.format_guard import wrap_boxed
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the heuristic teacher baseline locally.")
    parser.add_argument("--config")
    parser.add_argument("--input")
    parser.add_argument("--output", default="data/processed/baseline_eval.json")
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.config:
        config = read_yaml(args.config)
        args.input = config.get("processed_path", config.get("input_path", args.input))
        args.output = config.get("output_path", args.output)
        args.beam_width = int(config.get("beam_width", args.beam_width))
        args.max_depth = int(config.get("max_depth", args.max_depth))
        args.top_k = int(config.get("top_k", args.top_k))
    else:
        config = {}

    examples = _ensure_family_tags(_load_examples(args.input))
    engine = ChainSearchEngine(beam_width=args.beam_width, max_depth=args.max_depth)

    evaluation_rows = []
    record_index: dict[str, dict[str, object]] = {}
    for example in examples:
        candidates = engine.solve_example(example, top_k=args.top_k)
        annotate_example_from_candidates(example, candidates)
        best = candidates[0] if candidates else None
        answer = best.query_prediction if best and best.query_prediction is not None else ""
        row = {
            "id": example.id,
            "prediction": wrap_boxed(answer) if answer else "",
            "target_answer": example.target_answer or "",
            "official_family": example.metadata.official_family,
            "subtype": example.metadata.subtype,
            "teacher_confidence": example.metadata.teacher_confidence,
            "program_signature": example.metadata.program_signature,
            "steps": [] if best is None else [step.op_name for step in best.steps],
            "debug": None if best is None else best.to_debug_dict(),
        }
        evaluation_rows.append(row)
        record_index[example.id] = row

    summary = evaluate_predictions(
        evaluation_rows,
        numeric_rel_tolerance=float(config.get("numeric_rel_tolerance", 1e-2)),
        numeric_abs_tolerance=float(config.get("numeric_abs_tolerance", 1e-5)),
    )
    merged_records = []
    for record in summary["records"]:
        extras = record_index.get(record["id"], {})
        merged_records.append(
            {
                **record,
                "official_family": extras.get("official_family"),
                "subtype": extras.get("subtype"),
                "teacher_confidence": extras.get("teacher_confidence"),
                "program_signature": extras.get("program_signature"),
                "steps": extras.get("steps", []),
                "debug": extras.get("debug"),
            }
        )
    summary["records"] = merged_records
    summary["predictions"] = merged_records
    headline_metric = str(config.get("headline_metric", "competition_correct_rate"))
    summary["headline_metric"] = headline_metric
    summary["overall_accuracy"] = float(summary.get(headline_metric, summary["competition_correct_rate"]))
    write_json(args.output, summary)


if __name__ == "__main__":
    main()
