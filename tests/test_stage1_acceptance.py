from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io import write_json
from src.competition.harness_prompt import EXPECTED_CHAT_TEMPLATE_SHA16
from scripts.check_stage1_acceptance import check_stage1_acceptance


def _make_stage1_adapter_dir(tmp_path: Path) -> Path:
    adapter_dir = tmp_path / "adapter_stage1_format"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    (adapter_dir / "adapter_config.json").write_text('{"r": 32}', encoding="utf-8")
    payload = {
        "preflight": {"status": "ok", "chat_template_sha16": EXPECTED_CHAT_TEMPLATE_SHA16},
        "dataset_stats": {"length_unit": "bpe_tokens"},
        "num_matched_target_modules": 42,
        "num_train_records": 7561,
    }
    write_json(adapter_dir / "training_metadata.json", payload)
    write_json(adapter_dir / "last_run_summary.json", payload)
    return adapter_dir


def test_check_stage1_acceptance_passes_with_complete_artifacts(tmp_path: Path) -> None:
    adapter_dir = _make_stage1_adapter_dir(tmp_path)
    log_path = tmp_path / "stage1.log"
    log_path.write_text(
        "Loading checkpoint shards: 100%|##########| 13/13\n"
        " 10/472 [00:11<07:30,  1.02it/s]\n"
        "Saving model checkpoint to artifacts/adapter_stage1_format/checkpoint-100\n"
        "eval_loss = 0.1234\n",
        encoding="utf-8",
    )

    payload = check_stage1_acceptance(adapter_dir=adapter_dir, log_path=log_path)

    assert payload["accepted"] is True
    assert payload["preflight_status"] == "ok"
    assert payload["dataset_length_unit"] == "bpe_tokens"
    assert payload["num_matched_target_modules"] == 42
    assert payload["num_train_records"] == 7561
    assert payload["log"]["last_progress"] == {"current": 10, "total": 472}
    assert payload["log"]["last_checkpoint"] == "checkpoint-100"


def test_check_stage1_acceptance_rejects_missing_required_files(tmp_path: Path) -> None:
    adapter_dir = _make_stage1_adapter_dir(tmp_path)
    (adapter_dir / "last_run_summary.json").unlink()

    with pytest.raises(SystemExit, match="last_run_summary.json"):
        check_stage1_acceptance(adapter_dir=adapter_dir)


def test_check_stage1_acceptance_rejects_bad_preflight_status(tmp_path: Path) -> None:
    adapter_dir = _make_stage1_adapter_dir(tmp_path)
    write_json(
        adapter_dir / "training_metadata.json",
        {
            "preflight": {"status": "failed"},
            "dataset_stats": {"length_unit": "bpe_tokens"},
            "num_matched_target_modules": 42,
        },
    )

    with pytest.raises(SystemExit, match="preflight.status"):
        check_stage1_acceptance(adapter_dir=adapter_dir)


def test_check_stage1_acceptance_rejects_non_bpe_dataset_stats(tmp_path: Path) -> None:
    adapter_dir = _make_stage1_adapter_dir(tmp_path)
    write_json(
        adapter_dir / "training_metadata.json",
        {
            "preflight": {"status": "ok", "chat_template_sha16": EXPECTED_CHAT_TEMPLATE_SHA16},
            "dataset_stats": {"length_unit": "whitespace_words"},
            "num_matched_target_modules": 42,
            "num_train_records": 7561,
        },
    )

    with pytest.raises(SystemExit, match="dataset_stats.length_unit"):
        check_stage1_acceptance(adapter_dir=adapter_dir)


def test_check_stage1_acceptance_rejects_zero_target_module_matches(tmp_path: Path) -> None:
    adapter_dir = _make_stage1_adapter_dir(tmp_path)
    write_json(
        adapter_dir / "training_metadata.json",
        {
            "preflight": {"status": "ok", "chat_template_sha16": EXPECTED_CHAT_TEMPLATE_SHA16},
            "dataset_stats": {"length_unit": "bpe_tokens"},
            "num_matched_target_modules": 0,
            "num_train_records": 7561,
        },
    )

    with pytest.raises(SystemExit, match="num_matched_target_modules"):
        check_stage1_acceptance(adapter_dir=adapter_dir)


def test_check_stage1_acceptance_requires_progress_when_log_given(tmp_path: Path) -> None:
    adapter_dir = _make_stage1_adapter_dir(tmp_path)
    log_path = tmp_path / "stage1.log"
    log_path.write_text("loading only\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="no training progress line found"):
        check_stage1_acceptance(adapter_dir=adapter_dir, log_path=log_path)


def test_check_stage1_acceptance_rejects_non_positive_num_train_records(tmp_path: Path) -> None:
    adapter_dir = _make_stage1_adapter_dir(tmp_path)
    write_json(
        adapter_dir / "training_metadata.json",
        {
            "preflight": {"status": "ok", "chat_template_sha16": EXPECTED_CHAT_TEMPLATE_SHA16},
            "dataset_stats": {"length_unit": "bpe_tokens"},
            "num_matched_target_modules": 42,
            "num_train_records": 0,
        },
    )

    with pytest.raises(SystemExit, match="num_train_records"):
        check_stage1_acceptance(adapter_dir=adapter_dir)


def test_check_stage1_acceptance_rejects_chat_template_sha_mismatch(tmp_path: Path) -> None:
    adapter_dir = _make_stage1_adapter_dir(tmp_path)
    write_json(
        adapter_dir / "training_metadata.json",
        {
            "preflight": {"status": "ok", "chat_template_sha16": "mismatch123456789"},
            "dataset_stats": {"length_unit": "bpe_tokens"},
            "num_matched_target_modules": 42,
            "num_train_records": 7561,
        },
    )

    with pytest.raises(SystemExit, match="chat_template_sha16 mismatch"):
        check_stage1_acceptance(adapter_dir=adapter_dir)


def test_check_stage1_acceptance_rejects_empty_adapter_weights(tmp_path: Path) -> None:
    adapter_dir = _make_stage1_adapter_dir(tmp_path)
    (adapter_dir / "adapter_model.safetensors").write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="must be non-empty"):
        check_stage1_acceptance(adapter_dir=adapter_dir)
