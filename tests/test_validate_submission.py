from __future__ import annotations

from pathlib import Path

import pytest

import scripts.validate_submission as validate_submission_module
from src.common.io import write_yaml


def _write_config(path: Path) -> Path:
    write_yaml(
        path,
        {
            "base_model": "nvidia/Nemotron-3-Nano-30B",
            "model_source": "kagglehub",
            "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
            "training": {"output_dir": "artifacts/adapter_stage2_selected_trace"},
        },
    )
    return path


def _write_adapter_dir(path: Path, *, rank: int = 32) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    (path / "adapter_config.json").write_text(f'{{"r": {rank}}}', encoding="utf-8")
    return path


def test_validate_submission_rejects_rank_above_limit(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    adapter_dir = _write_adapter_dir(tmp_path / "adapter", rank=64)

    with pytest.raises(validate_submission_module.SubmissionValidationError, match="rank"):
        validate_submission_module.validate_submission(
            config_path=config_path,
            adapter_dir=adapter_dir,
            output_path=tmp_path / "validation.json",
        )


def test_validate_submission_requires_smoke_input_for_labels(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    adapter_dir = _write_adapter_dir(tmp_path / "adapter")

    with pytest.raises(validate_submission_module.SubmissionValidationError, match="--labels requires --smoke-input"):
        validate_submission_module.validate_submission(
            config_path=config_path,
            adapter_dir=adapter_dir,
            output_path=tmp_path / "validation.json",
            labels=tmp_path / "labels.jsonl",
        )


def test_validate_submission_requires_smoke_input_for_packaging(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    adapter_dir = _write_adapter_dir(tmp_path / "adapter")

    with pytest.raises(
        validate_submission_module.SubmissionValidationError,
        match="--package-output requires --smoke-input",
    ):
        validate_submission_module.validate_submission(
            config_path=config_path,
            adapter_dir=adapter_dir,
            output_path=tmp_path / "validation.json",
            package_output=tmp_path / "submission.zip",
        )


def test_validate_submission_requires_labels_for_packaging(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    adapter_dir = _write_adapter_dir(tmp_path / "adapter")
    smoke_input = tmp_path / "smoke.jsonl"
    smoke_input.write_text('{"id":"x"}\n', encoding="utf-8")

    with pytest.raises(
        validate_submission_module.SubmissionValidationError,
        match="--package-output requires --labels",
    ):
        validate_submission_module.validate_submission(
            config_path=config_path,
            adapter_dir=adapter_dir,
            output_path=tmp_path / "validation.json",
            smoke_input=smoke_input,
            package_output=tmp_path / "submission.zip",
        )


def test_validate_submission_requires_readable_adapter_rank(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    adapter_dir = tmp_path / "adapter_no_rank"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(
        validate_submission_module.SubmissionValidationError,
        match="Adapter rank could not be read",
    ):
        validate_submission_module.validate_submission(
            config_path=config_path,
            adapter_dir=adapter_dir,
            output_path=tmp_path / "validation.json",
        )


def test_validate_submission_packages_after_local_eval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    adapter_dir = _write_adapter_dir(tmp_path / "adapter")
    smoke_input = tmp_path / "smoke.jsonl"
    smoke_input.write_text('{"id":"x"}\n', encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text('{"id":"x"}\n', encoding="utf-8")

    run_calls: list[tuple[str, str]] = []
    eval_calls: list[tuple[str, str]] = []

    def _fake_run_inference(config, *, input_path, adapter_dir, output_path, max_new_tokens=None):
        run_calls.append((str(input_path), str(output_path)))
        Path(output_path).write_text('{"id":"x","prediction":"\\\\boxed{1}"}\n', encoding="utf-8")
        return Path(output_path)

    def _fake_evaluate_replica(*, prediction_path, label_path, split_path=None):
        eval_calls.append((str(prediction_path), str(label_path)))
        return {"competition_correct_rate": 1.0}

    monkeypatch.setattr(validate_submission_module, "run_inference", _fake_run_inference)
    monkeypatch.setattr(validate_submission_module, "evaluate_replica", _fake_evaluate_replica)
    monkeypatch.setattr(validate_submission_module, "build_submission_zip", lambda adapter_dir, output: Path(output))

    payload = validate_submission_module.validate_submission(
        config_path=config_path,
        adapter_dir=adapter_dir,
        output_path=tmp_path / "validation.json",
        smoke_input=smoke_input,
        labels=labels,
        package_output=tmp_path / "submission.zip",
    )

    assert "local_eval" in payload
    assert payload["package_output"] == str(tmp_path / "submission.zip")
    assert run_calls
    assert eval_calls
