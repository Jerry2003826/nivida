from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_stage2_answer_focused_data import (
    build_answer_focused_data,
    build_split_command,
    resolve_existing_path,
    validate_jsonl,
)


def test_resolve_existing_path_falls_back_to_parent_data(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    parent_input = tmp_path / "data" / "processed" / "train.jsonl"
    parent_input.parent.mkdir(parents=True)
    parent_input.write_text("{}\n", encoding="utf-8")

    resolved = resolve_existing_path(
        "data/processed/train.jsonl",
        repo_root=repo,
    )

    assert resolved == parent_input


def test_build_split_command_uses_stage2_answer_focused_contract(tmp_path: Path) -> None:
    command = build_split_command(
        python_executable="python",
        input_path=tmp_path / "input.jsonl",
        output_path=tmp_path / "output.jsonl",
        report_path=tmp_path / "report.json",
        tokenizer_path=tmp_path / "tokenizer",
        completion_style="answer_only",
    )

    assert command[:3] == ["python", "-m", "src.student.sft_dataset_builder"]
    assert command[command.index("--selection-profile") + 1] == "stage2"
    assert command[command.index("--prompt-mode") + 1] == "chat_thinking"
    assert command[command.index("--completion-style") + 1] == "answer_only"
    assert "--no-balance-by-family" in command
    assert "--include-metadata" in command


def test_build_answer_focused_data_dry_run_plans_four_splits_with_parent_inputs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    parent_processed = tmp_path / "data" / "processed"
    parent_processed.mkdir(parents=True)
    train = parent_processed / "stage2_official_train_no_hard_valid.jsonl"
    valid = parent_processed / "proxy_all_family_valid.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    valid.write_text("{}\n", encoding="utf-8")

    payload = build_answer_focused_data(
        repo_root=repo,
        tokenizer_path="artifacts/tokenizer",
        train_input="data/processed/stage2_official_train_no_hard_valid.jsonl",
        valid_input="data/processed/proxy_all_family_valid.jsonl",
        out_dir="data/processed",
        python_executable="python",
        dry_run=True,
    )

    styles = [
        command[command.index("--completion-style") + 1]
        for command in payload["commands"]
    ]
    outputs = [
        command[command.index("--output") + 1]
        for command in payload["commands"]
    ]

    assert payload["status"] == "dry_run"
    assert payload["train_input"] == str(train)
    assert payload["valid_input"] == str(valid)
    assert styles == ["answer_only", "answer_only", "short_trace", "short_trace"]
    assert any(output.endswith("stage2_answer_only_train.jsonl") for output in outputs)
    assert any(output.endswith("stage2_short_trace_valid.jsonl") for output in outputs)


def test_validate_jsonl_counts_rows_and_rejects_empty_completion(tmp_path: Path) -> None:
    good = tmp_path / "good.jsonl"
    good.write_text(
        json.dumps({"prompt": "p", "completion": "c"}) + "\n",
        encoding="utf-8",
    )
    assert validate_jsonl(good)["rows"] == 1

    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"prompt": "p", "completion": ""}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="without prompt/completion"):
        validate_jsonl(bad)
