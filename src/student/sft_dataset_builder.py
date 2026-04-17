from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.common.io import load_jsonl, read_json, read_yaml, write_jsonl
from src.competition.prompt_templates import build_competition_prompt
from src.competition.schema import PuzzleExample
from src.student.format_guard import wrap_boxed


def build_sft_record(example: PuzzleExample, *, stage: str) -> dict[str, Any]:
    prompt = build_competition_prompt(example)
    if stage == "format_only":
        completion = wrap_boxed(example.target_answer or "")
    else:
        completion = f"Infer the rule briefly, then answer.\n{wrap_boxed(example.target_answer or '')}"
    return {
        "id": example.id,
        "stage": stage,
        "prompt": prompt,
        "completion": completion,
        "target_answer": example.target_answer,
        "family_tags": example.metadata.family_tags,
    }


def build_stage_dataset(examples: list[PuzzleExample], *, stage: str) -> list[dict[str, Any]]:
    return [build_sft_record(example, stage=stage) for example in examples if example.target_answer is not None]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage-specific SFT datasets.")
    parser.add_argument("--config", default="configs/train_lora.yaml")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--stage", default="stage_c")
    parser.add_argument("--split-file")
    parser.add_argument("--split-name")
    parser.add_argument("--split-role", choices=["train", "valid"])
    args = parser.parse_args()

    config = read_yaml(args.config) if args.config else {}
    input_path = args.input or (
        config.get("training", {}).get("dataset_path") if args.stage != "format_only" else "data/processed/parsed_train.jsonl"
    )
    output_path = args.output or "data/processed/stage_c_train.jsonl"
    rows = [PuzzleExample.from_dict(row) for row in load_jsonl(input_path)]
    if args.split_file and args.split_name and args.split_role:
        rows = filter_examples_by_split(rows, split_file=args.split_file, split_name=args.split_name, split_role=args.split_role)
    dataset = build_stage_dataset(rows, stage=args.stage)
    write_jsonl(output_path, dataset)


if __name__ == "__main__":
    main()
