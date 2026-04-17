from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.common.io import load_jsonl, read_json, read_yaml, write_jsonl
from src.competition.prompt_templates import build_competition_prompt
from src.competition.schema import PuzzleExample
from src.teacher.chain_search import ChainSearchEngine
from src.teacher.family_tagger import apply_family_tags
from src.teacher.program_signature import annotate_example_from_candidates
from src.teacher.trace_compiler import compile_completion


def _parse_input_paths(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _load_examples(input_paths: list[str]) -> list[PuzzleExample]:
    examples: list[PuzzleExample] = []
    for input_path in input_paths:
        examples.extend(PuzzleExample.from_dict(row) for row in load_jsonl(input_path))
    return examples


def _annotate_examples(examples: list[PuzzleExample], *, top_k: int = 2) -> list[PuzzleExample]:
    apply_family_tags(examples)
    engine = ChainSearchEngine(beam_width=8, max_depth=2)
    for example in examples:
        if example.metadata.program_signature and example.metadata.teacher_confidence is not None:
            continue
        candidates = engine.solve_example(example, top_k=top_k)
        annotate_example_from_candidates(example, candidates)
    return examples


def _build_record(example: PuzzleExample, *, completion_style: str, source_label: str) -> dict[str, Any]:
    return {
        "id": example.id,
        "prompt": build_competition_prompt(example),
        "completion": compile_completion(example, style=completion_style),
        "target_answer": example.target_answer,
        "official_family": example.metadata.official_family,
        "subtype": example.metadata.subtype,
        "teacher_confidence": example.metadata.teacher_confidence,
        "program_signature": example.metadata.program_signature,
        "difficulty": example.metadata.difficulty,
        "source": source_label,
        "split": example.metadata.split,
    }


def filter_examples_by_split(
    examples: list[PuzzleExample],
    *,
    split_file: str | Path,
    split_name: str,
    split_role: str,
) -> list[PuzzleExample]:
    payload = read_json(split_file)
    if split_name not in payload:
        raise KeyError(f"Split '{split_name}' not found in {split_file}")
    split_payload = payload[split_name]
    key = f"{split_role}_ids"
    if key not in split_payload:
        raise KeyError(f"Split role '{split_role}' not found in split '{split_name}'")
    keep_ids = set(split_payload[key])
    return [example for example in examples if example.id in keep_ids]


def _source_kind(example: PuzzleExample) -> str:
    if example.metadata.source == "synthetic":
        return "synth"
    if example.metadata.source == "repair":
        return "repair"
    return "official"


def _filter_by_source(examples: list[PuzzleExample], source_filter: str | None) -> list[PuzzleExample]:
    if not source_filter:
        return examples
    wanted = source_filter.strip()
    return [example for example in examples if _source_kind(example) == wanted]


def build_stage1_sft(examples: list[PuzzleExample]) -> list[dict[str, Any]]:
    records = []
    for example in examples:
        if _source_kind(example) != "official":
            continue
        if example.target_answer is None or not example.query:
            continue
        records.append(_build_record(example, completion_style="answer_only", source_label="official"))
    return records


def _select_official_stage2(example: PuzzleExample) -> bool:
    if _source_kind(example) != "official":
        return False
    if example.target_answer is None or not example.query:
        return False
    if (example.metadata.teacher_confidence or 0.0) < 0.80:
        return False
    if float(example.metadata.extras.get("support_coverage", 0.0)) < 1.0:
        return False
    if float(example.metadata.extras.get("top1_top2_margin", 0.0)) < 0.05:
        return False
    return bool(example.metadata.program_signature)


def _select_synth_stage2(example: PuzzleExample, *, official_signatures: set[str]) -> bool:
    if _source_kind(example) != "synth":
        return False
    if example.target_answer is None or not example.query:
        return False
    if not bool(example.metadata.extras.get("solver_verifiable")):
        return False
    signature = example.metadata.program_signature
    if not signature:
        return False
    return signature not in official_signatures


def build_selected_sft(examples: list[PuzzleExample], *, completion_style: str) -> list[dict[str, Any]]:
    annotated = _annotate_examples(examples)
    selected_official = [example for example in annotated if _select_official_stage2(example)]
    official_signatures = {example.metadata.program_signature for example in selected_official if example.metadata.program_signature}
    selected_synth = [example for example in annotated if _select_synth_stage2(example, official_signatures=official_signatures)]
    records = [
        _build_record(example, completion_style=completion_style, source_label=_source_kind(example))
        for example in selected_official + selected_synth
    ]
    return records


def _classify_repair_bucket(example: PuzzleExample, failure_row: dict[str, Any]) -> str:
    if not failure_row.get("boxed_valid", True):
        return "format_only"
    family_legality = float((failure_row.get("debug") or {}).get("family_legality", 1.0))
    if family_legality < 1.0:
        return "wrong_family"
    return "right_family_wrong_program"


def build_repair_set(
    examples: list[PuzzleExample],
    *,
    repair_artifact: str | Path,
    completion_style: str,
) -> list[dict[str, Any]]:
    payload = read_json(repair_artifact)
    failures = payload.get("records", payload.get("rows", []))
    failure_index = {
        str(row["id"]): row
        for row in failures
        if not row.get("competition_correct", False)
    }
    annotated = _annotate_examples(examples)
    records: list[dict[str, Any]] = []
    for example in annotated:
        failure_row = failure_index.get(example.id)
        if failure_row is None:
            continue
        bucket = _classify_repair_bucket(example, failure_row)
        example.metadata.source = "repair"
        example.metadata.extras = {**dict(example.metadata.extras), "repair_bucket": bucket}
        record = _build_record(example, completion_style=completion_style, source_label="repair")
        record["repair_bucket"] = bucket
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage1/stage2/stage3 SFT datasets.")
    parser.add_argument("--config")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--selection-profile", choices=["stage1", "stage2", "stage3"], default="stage2")
    parser.add_argument("--completion-style", choices=["answer_only", "short_trace", "token_trace"], default="token_trace")
    parser.add_argument("--source-filter", choices=["official", "synth", "repair"])
    parser.add_argument("--split-file")
    parser.add_argument("--split-name")
    parser.add_argument("--split-role", choices=["train", "valid"])
    parser.add_argument("--repair-artifact")
    args = parser.parse_args()

    config = read_yaml(args.config) if args.config else {}
    input_paths = _parse_input_paths(
        args.input
        or config.get("input")
        or config.get("training", {}).get("dataset_input_paths")
        or config.get("training", {}).get("dataset_path")
    )
    if not input_paths:
        raise ValueError("At least one input JSONL path is required.")

    examples = _load_examples(input_paths)
    if args.split_file and args.split_name and args.split_role:
        examples = filter_examples_by_split(
            examples,
            split_file=args.split_file,
            split_name=args.split_name,
            split_role=args.split_role,
        )
    examples = _filter_by_source(examples, args.source_filter)

    completion_style = args.completion_style
    if args.selection_profile == "stage1":
        completion_style = "answer_only"
        dataset = build_stage1_sft(examples)
        default_output = "data/processed/stage1_format_align.jsonl"
    elif args.selection_profile == "stage2":
        dataset = build_selected_sft(examples, completion_style=completion_style)
        default_output = "data/processed/stage2_distill.jsonl"
    else:
        if not args.repair_artifact:
            raise ValueError("--repair-artifact is required for selection-profile=stage3")
        dataset = build_repair_set(examples, repair_artifact=args.repair_artifact, completion_style=completion_style)
        default_output = "data/processed/stage3_repair.jsonl"

    output_path = args.output or config.get("output") or default_output
    write_jsonl(output_path, dataset)


if __name__ == "__main__":
    main()
