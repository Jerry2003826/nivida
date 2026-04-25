from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_PATH = Path(
    "artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default"
)
DEFAULT_TRAIN_INPUT = Path("data/processed/stage2_official_train_no_hard_valid.jsonl")
DEFAULT_VALID_INPUT = Path("data/processed/proxy_all_family_valid.jsonl")
DEFAULT_OUT_DIR = Path("data/processed")


def resolve_existing_path(
    path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
    allow_parent_data: bool = True,
) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = Path(repo_root) / target
    if target.exists():
        return target
    if allow_parent_data:
        relative = Path(path)
        if relative.is_absolute():
            relative = Path(*relative.parts[1:])
        parent_target = Path(repo_root).parent / relative
        if parent_target.exists():
            return parent_target
    raise FileNotFoundError(f"Missing input: {path}")


def build_split_command(
    *,
    python_executable: str,
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    tokenizer_path: str | Path,
    completion_style: str,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "src.student.sft_dataset_builder",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--selection-profile",
        "stage2",
        "--prompt-mode",
        "chat_thinking",
        "--tokenizer-path",
        str(tokenizer_path),
        "--completion-style",
        completion_style,
        "--beam-width",
        "10",
        "--max-depth",
        "3",
        "--top-k",
        "3",
        "--no-balance-by-family",
        "--hard-triad-repeat-factor",
        "1",
        "--max-per-signature-bucket",
        "0",
        "--include-metadata",
        "--report-output",
        str(report_path),
    ]


def validate_jsonl(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    count = 0
    empty = 0
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            if not row.get("prompt") or not row.get("completion"):
                empty += 1
    if count <= 0:
        raise ValueError(f"{target} is empty")
    if empty:
        raise ValueError(f"{target} has {empty} rows without prompt/completion")
    return {"path": str(target), "rows": count}


def planned_splits(
    *,
    train_input: str | Path,
    valid_input: str | Path,
    out_dir: str | Path,
) -> list[dict[str, Path | str]]:
    output_dir = Path(out_dir)
    return [
        {
            "split": "train",
            "input": Path(train_input),
            "style": "answer_only",
            "output": output_dir / "stage2_answer_only_train.jsonl",
            "report": output_dir / "stage2_answer_only_train_report.json",
        },
        {
            "split": "valid",
            "input": Path(valid_input),
            "style": "answer_only",
            "output": output_dir / "stage2_answer_only_valid.jsonl",
            "report": output_dir / "stage2_answer_only_valid_report.json",
        },
        {
            "split": "train",
            "input": Path(train_input),
            "style": "short_trace",
            "output": output_dir / "stage2_short_trace_train.jsonl",
            "report": output_dir / "stage2_short_trace_train_report.json",
        },
        {
            "split": "valid",
            "input": Path(valid_input),
            "style": "short_trace",
            "output": output_dir / "stage2_short_trace_valid.jsonl",
            "report": output_dir / "stage2_short_trace_valid_report.json",
        },
    ]


def build_answer_focused_data(
    *,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
    train_input: str | Path = DEFAULT_TRAIN_INPUT,
    valid_input: str | Path = DEFAULT_VALID_INPUT,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    python_executable: str = sys.executable,
    repo_root: str | Path = REPO_ROOT,
    allow_parent_data: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root)
    output_dir = Path(out_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Path(tokenizer_path)
    if not tokenizer.is_absolute():
        tokenizer = root / tokenizer
    resolved_train = resolve_existing_path(
        train_input,
        repo_root=root,
        allow_parent_data=allow_parent_data,
    )
    resolved_valid = resolve_existing_path(
        valid_input,
        repo_root=root,
        allow_parent_data=allow_parent_data,
    )

    results: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    for item in planned_splits(
        train_input=resolved_train,
        valid_input=resolved_valid,
        out_dir=output_dir,
    ):
        command = build_split_command(
            python_executable=python_executable,
            input_path=item["input"],
            output_path=item["output"],
            report_path=item["report"],
            tokenizer_path=tokenizer,
            completion_style=str(item["style"]),
        )
        commands.append(command)
        if dry_run:
            continue
        subprocess.run(command, cwd=root, check=True)
        results.append(validate_jsonl(item["output"]))

    return {
        "status": "dry_run" if dry_run else "done",
        "repo_root": str(root),
        "tokenizer_path": str(tokenizer),
        "train_input": str(resolved_train),
        "valid_input": str(resolved_valid),
        "out_dir": str(output_dir),
        "commands": commands,
        "validated_outputs": results,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build stage2 answer-only and short-trace datasets without requiring bash."
    )
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--valid-input", type=Path, default=DEFAULT_VALID_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--no-parent-data-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    payload = build_answer_focused_data(
        tokenizer_path=args.tokenizer_path,
        train_input=args.train_input,
        valid_input=args.valid_input,
        out_dir=args.out_dir,
        python_executable=args.python_executable,
        allow_parent_data=not args.no_parent_data_fallback,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
