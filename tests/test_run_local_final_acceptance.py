from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io import read_json, write_json
import scripts.run_local_final_acceptance as acceptance_module


def _write_adapter_dir(path: Path, *, marker: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(f'{{"r": 32, "marker": "{marker}"}}', encoding="utf-8")
    (path / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    return path


def _write_eval(path: Path, rate: float) -> Path:
    write_json(
        path,
        {
            "competition_correct_rate": rate,
            "num_examples": 400,
            "coverage": {
                "num_missing": 0,
                "num_unexpected": 0,
                "num_duplicate": 0,
            },
        },
    )
    return path


def _acceptance_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "stage2_adapter_dir": tmp_path / "adapter_stage2_bestproxy",
        "stage3_adapter_dir": tmp_path / "adapter_stage3_bestproxy",
        "stage2_hard_eval": tmp_path / "stage2_bestproxy_hard_eval.json",
        "stage2_all_eval": tmp_path / "stage2_bestproxy_all_eval.json",
        "stage3_hard_eval": tmp_path / "stage3_bestproxy_hard_eval.json",
        "stage3_all_eval": tmp_path / "stage3_bestproxy_all_eval.json",
        "output_adapter_dir": tmp_path / "adapter_final_selected",
        "selection_json": tmp_path / "final_adapter_selection.json",
        "probe_json": tmp_path / "adapter_submission_probe.json",
        "validation_json": tmp_path / "submission_validation.json",
        "summary_json": tmp_path / "final_acceptance_report.json",
        "submission_zip": tmp_path / "submission.zip",
        "config_path": tmp_path / "train_stage3_repair.yaml",
        "smoke_input": tmp_path / "official_train_tagged.jsonl",
        "labels": tmp_path / "official_train_tagged.jsonl",
        "splits": tmp_path / "splits.json",
    }


def _write_required_inputs(paths: dict[str, Path]) -> None:
    _write_adapter_dir(paths["stage2_adapter_dir"], marker="stage2")
    _write_adapter_dir(paths["stage3_adapter_dir"], marker="stage3")
    _write_eval(paths["stage2_hard_eval"], 0.41)
    _write_eval(paths["stage2_all_eval"], 0.52)
    _write_eval(paths["stage3_hard_eval"], 0.43)
    _write_eval(paths["stage3_all_eval"], 0.54)
    paths["config_path"].write_text("training:\n  output_dir: artifacts/adapter_stage3_repair\n", encoding="utf-8")
    paths["smoke_input"].write_text('{"id":"x"}\n', encoding="utf-8")
    paths["splits"].write_text("{}", encoding="utf-8")


def test_run_local_final_acceptance_success_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _acceptance_paths(tmp_path)
    _write_required_inputs(paths)

    calls: list[str] = []

    def _fake_selection(**kwargs):
        calls.append("select")
        payload = {
            "decision": {"selected_stage": "stage3", "rule": "all_family_primary"},
            "selected_adapter_dir": str(kwargs["output_adapter_dir"]),
            "selected_source_dir": str(paths["stage3_adapter_dir"]),
            "stage2": {
                "hard_triad_proxy": {"competition_correct_rate": 0.41},
                "all_family_proxy": {"competition_correct_rate": 0.52},
            },
            "stage3": {
                "hard_triad_proxy": {"competition_correct_rate": 0.43},
                "all_family_proxy": {"competition_correct_rate": 0.54},
            },
        }
        _write_adapter_dir(Path(kwargs["output_adapter_dir"]), marker="selected")
        write_json(kwargs["output_json"], payload)
        return payload

    def _fake_probe(**kwargs):
        calls.append("probe")
        payload = {
            "probe_mode": "trained_adapter",
            "requested_rank": 32,
            "rank": 32,
            "requested_selected_suffixes": ["in_proj", "out_proj"],
            "artifact_selected_suffixes": ["in_proj", "out_proj"],
            "artifact_matches_requested_config": True,
            "artifact_shape_matches_formula": True,
            "lora_b_likely_untrained": False,
            "real_trained_adapter_archive_ratio": 0.22,
            "real_trained_adapter_weight_compression_ratio": 0.21,
            "formula_predicted_zip_bytes": 888_873_792,
            "zip_size_bytes": 812_000_000,
        }
        write_json(kwargs["output_json"], payload)
        return payload

    def _fake_validate(**kwargs):
        calls.append("validate")
        payload = {
            "submission_budget": {
                "status": "within_limit",
                "projected_submission_zip_bytes": 888_873_792,
            },
            "package_output": str(kwargs["package_output"]),
            "package_size_bytes": 801_000_000,
            "local_eval": {"competition_correct_rate": 0.55},
        }
        Path(kwargs["package_output"]).write_text("zip", encoding="utf-8")
        write_json(kwargs["output_json"], payload)
        return payload

    monkeypatch.setattr(acceptance_module, "run_final_adapter_selection", _fake_selection)
    monkeypatch.setattr(acceptance_module, "run_trained_probe", _fake_probe)
    monkeypatch.setattr(acceptance_module, "run_submission_validation", _fake_validate)

    report = acceptance_module.run_local_final_acceptance(
        **paths,
        max_new_tokens=2048,
    )

    assert calls == ["select", "probe", "validate"]
    assert report["status"] == "pass"
    assert report["selected_stage"] == "stage3"
    assert report["probe"]["artifact_shape_matches_formula"] is True
    assert report["validation"]["physical_packaged_zip_bytes"] == 801_000_000
    assert read_json(paths["summary_json"])["status"] == "pass"


def test_run_local_final_acceptance_rejects_missing_stage_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _acceptance_paths(tmp_path)
    _write_adapter_dir(paths["stage2_adapter_dir"], marker="stage2")
    _write_eval(paths["stage2_hard_eval"], 0.41)
    _write_eval(paths["stage2_all_eval"], 0.52)
    _write_eval(paths["stage3_hard_eval"], 0.43)
    _write_required_inputs(paths)
    paths["stage3_all_eval"].unlink()

    monkeypatch.setattr(
        acceptance_module,
        "run_final_adapter_selection",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("selection should not run")),
    )

    with pytest.raises(FileNotFoundError, match="Required file not found"):
        acceptance_module.run_local_final_acceptance(
            **paths,
            max_new_tokens=2048,
        )

    summary = read_json(paths["summary_json"])
    assert summary["status"] == "fail"
    assert summary["failure_stage"] == "input_validation"


def test_run_local_final_acceptance_records_probe_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _acceptance_paths(tmp_path)
    _write_required_inputs(paths)

    validate_called = False

    def _fake_selection(**kwargs):
        payload = {
            "decision": {"selected_stage": "stage2", "rule": "prefer_stage2_on_tie"},
            "selected_adapter_dir": str(kwargs["output_adapter_dir"]),
            "selected_source_dir": str(paths["stage2_adapter_dir"]),
            "stage2": {
                "hard_triad_proxy": {"competition_correct_rate": 0.41},
                "all_family_proxy": {"competition_correct_rate": 0.52},
            },
            "stage3": {
                "hard_triad_proxy": {"competition_correct_rate": 0.41},
                "all_family_proxy": {"competition_correct_rate": 0.52},
            },
        }
        _write_adapter_dir(Path(kwargs["output_adapter_dir"]), marker="selected")
        write_json(kwargs["output_json"], payload)
        return payload

    def _fake_probe(**_kwargs):
        raise RuntimeError("probe failed")

    def _fake_validate(**_kwargs):
        nonlocal validate_called
        validate_called = True
        raise AssertionError("validate should not run after probe failure")

    monkeypatch.setattr(acceptance_module, "run_final_adapter_selection", _fake_selection)
    monkeypatch.setattr(acceptance_module, "run_trained_probe", _fake_probe)
    monkeypatch.setattr(acceptance_module, "run_submission_validation", _fake_validate)

    with pytest.raises(RuntimeError, match="probe failed"):
        acceptance_module.run_local_final_acceptance(
            **paths,
            max_new_tokens=2048,
        )

    summary = read_json(paths["summary_json"])
    assert summary["status"] == "fail"
    assert summary["failure_stage"] == "trained_probe"
    assert validate_called is False


def test_run_local_final_acceptance_records_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _acceptance_paths(tmp_path)
    _write_required_inputs(paths)

    def _fake_selection(**kwargs):
        payload = {
            "decision": {"selected_stage": "stage3", "rule": "all_family_primary"},
            "selected_adapter_dir": str(kwargs["output_adapter_dir"]),
            "selected_source_dir": str(paths["stage3_adapter_dir"]),
            "stage2": {
                "hard_triad_proxy": {"competition_correct_rate": 0.41},
                "all_family_proxy": {"competition_correct_rate": 0.52},
            },
            "stage3": {
                "hard_triad_proxy": {"competition_correct_rate": 0.43},
                "all_family_proxy": {"competition_correct_rate": 0.54},
            },
        }
        _write_adapter_dir(Path(kwargs["output_adapter_dir"]), marker="selected")
        write_json(kwargs["output_json"], payload)
        return payload

    def _fake_probe(**kwargs):
        payload = {
            "probe_mode": "trained_adapter",
            "requested_rank": 32,
            "rank": 32,
            "requested_selected_suffixes": ["in_proj"],
            "artifact_selected_suffixes": ["in_proj"],
            "artifact_matches_requested_config": True,
            "artifact_shape_matches_formula": True,
            "lora_b_likely_untrained": False,
            "real_trained_adapter_archive_ratio": 0.22,
            "real_trained_adapter_weight_compression_ratio": 0.21,
            "formula_predicted_zip_bytes": 888_873_792,
            "zip_size_bytes": 812_000_000,
        }
        write_json(kwargs["output_json"], payload)
        return payload

    def _fake_validate(**_kwargs):
        raise RuntimeError("packaging rejected")

    monkeypatch.setattr(acceptance_module, "run_final_adapter_selection", _fake_selection)
    monkeypatch.setattr(acceptance_module, "run_trained_probe", _fake_probe)
    monkeypatch.setattr(acceptance_module, "run_submission_validation", _fake_validate)

    with pytest.raises(RuntimeError, match="packaging rejected"):
        acceptance_module.run_local_final_acceptance(
            **paths,
            max_new_tokens=2048,
        )

    summary = read_json(paths["summary_json"])
    assert summary["status"] == "fail"
    assert summary["failure_stage"] == "validate_submission"


def test_makefile_final_acceptance_target_points_to_script() -> None:
    makefile = Path(__file__).resolve().parents[1] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "final-acceptance:" in text
    assert "$(PYTHON) scripts/run_local_final_acceptance.py" in text
