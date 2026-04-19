from __future__ import annotations

from pathlib import Path

import pytest
from safetensors import safe_open

from src.common.io import write_yaml
from src.student.adapter_submission_budget import (
    FULL_WIDE_TARGET_REGEX,
    SUBMISSION_SAFE_WIDE_TARGET_REGEX,
)
from scripts.probe_adapter_submission_size import (
    TINY_NEMOTRON_LIKE_PAYLOAD,
    probe_adapter_submission_size,
)


def _write_probe_config(path: Path, *, target_modules: str = SUBMISSION_SAFE_WIDE_TARGET_REGEX) -> Path:
    write_yaml(
        path,
        {
            "base_model": "nvidia/Nemotron-3-Nano-30B",
            "model_source": "kagglehub",
            "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
            "trust_remote_code": True,
            "lora": {
                "rank": 32,
                "target_modules": target_modules,
            },
            "training": {
                "output_dir": "artifacts/adapter_stage2_selected_trace",
            },
        },
    )
    return path


def test_tiny_mode_formula_matches_real_zip_within_5pct(tmp_path: Path) -> None:
    config_path = _write_probe_config(tmp_path / "probe_config.yaml")
    output_path = tmp_path / "probe.json"

    payload = probe_adapter_submission_size(
        config={
            "base_model": "nvidia/Nemotron-3-Nano-30B",
            "model_source": "kagglehub",
            "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
            "trust_remote_code": True,
            "lora": {"rank": 32, "target_modules": SUBMISSION_SAFE_WIDE_TARGET_REGEX},
            "training": {"output_dir": "artifacts/adapter_stage2_selected_trace"},
        },
        config_path=config_path,
        output_path=output_path,
        tiny_mode=True,
    )

    assert payload["tiny_mode"] is True
    assert payload["formula_error_pct"] < 5.0


def test_tiny_mode_layer_count_matches_analytic(tmp_path: Path) -> None:
    config_path = _write_probe_config(tmp_path / "probe_config.yaml")
    output_path = tmp_path / "probe.json"

    payload = probe_adapter_submission_size(
        config={
            "base_model": "nvidia/Nemotron-3-Nano-30B",
            "model_source": "kagglehub",
            "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
            "trust_remote_code": True,
            "lora": {"rank": 32, "target_modules": SUBMISSION_SAFE_WIDE_TARGET_REGEX},
            "training": {"output_dir": "artifacts/adapter_stage2_selected_trace"},
        },
        config_path=config_path,
        output_path=output_path,
        tiny_mode=True,
    )

    assert payload["mamba_layers"] == TINY_NEMOTRON_LIKE_PAYLOAD["hybrid_override_pattern"].count("M")
    assert payload["moe_layers"] == TINY_NEMOTRON_LIKE_PAYLOAD["hybrid_override_pattern"].count("E")
    assert payload["attention_layers"] == TINY_NEMOTRON_LIKE_PAYLOAD["hybrid_override_pattern"].count("*")


def test_tiny_mode_fp32_dtype_confirmed(tmp_path: Path) -> None:
    config_path = _write_probe_config(tmp_path / "probe_config.yaml")
    output_path = tmp_path / "probe.json"

    payload = probe_adapter_submission_size(
        config={
            "base_model": "nvidia/Nemotron-3-Nano-30B",
            "model_source": "kagglehub",
            "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
            "trust_remote_code": True,
            "lora": {"rank": 32, "target_modules": SUBMISSION_SAFE_WIDE_TARGET_REGEX},
            "training": {"output_dir": "artifacts/adapter_stage2_selected_trace"},
        },
        config_path=config_path,
        output_path=output_path,
        tiny_mode=True,
    )

    with safe_open(Path(payload["adapter_model_path"]), framework="np") as handle:
        first_key = next(iter(handle.keys()))
        tensor = handle.get_tensor(first_key)
        assert str(tensor.dtype) in {"float32", "torch.float32"}


def test_probe_confirms_float32_adapter_dtype(tmp_path: Path) -> None:
    config_path = _write_probe_config(tmp_path / "probe_config.yaml")
    output_path = tmp_path / "probe.json"

    payload = probe_adapter_submission_size(
        config={
            "base_model": "nvidia/Nemotron-3-Nano-30B",
            "model_source": "kagglehub",
            "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
            "trust_remote_code": True,
            "lora": {"rank": 32, "target_modules": SUBMISSION_SAFE_WIDE_TARGET_REGEX},
            "training": {"output_dir": "artifacts/adapter_stage2_selected_trace"},
        },
        config_path=config_path,
        output_path=output_path,
        tiny_mode=True,
    )

    with safe_open(Path(payload["adapter_model_path"]), framework="np") as handle:
        for key in handle.keys():
            if key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"):
                tensor = handle.get_tensor(key)
                assert str(tensor.dtype) in {"float32", "torch.float32"}


def test_probe_refuses_to_run_with_unsafe_target_modules(tmp_path: Path) -> None:
    config_path = _write_probe_config(
        tmp_path / "unsafe_probe_config.yaml",
        target_modules=FULL_WIDE_TARGET_REGEX,
    )
    output_path = tmp_path / "probe.json"

    with pytest.raises(ValueError, match="submission-safe"):
        probe_adapter_submission_size(
            config={
                "base_model": "nvidia/Nemotron-3-Nano-30B",
                "model_source": "kagglehub",
                "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
                "trust_remote_code": True,
                "lora": {"rank": 32, "target_modules": FULL_WIDE_TARGET_REGEX},
                "training": {"output_dir": "artifacts/adapter_stage2_selected_trace"},
            },
            config_path=config_path,
            output_path=output_path,
            tiny_mode=True,
        )
