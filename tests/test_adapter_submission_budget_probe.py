from __future__ import annotations

from pathlib import Path

import pytest
from safetensors import safe_open

import src.student.adapter_submission_budget as budget_module
from scripts.probe_adapter_submission_size import (
    TINY_NEMOTRON_LIKE_PAYLOAD,
    probe_adapter_submission_size,
)
from src.common.io import write_yaml
from src.student.adapter_submission_budget import (
    SUBMISSION_SAFE_WIDE_TARGET_REGEX,
    _HYPOTHETICAL_OVER_LIMIT_TARGET_REGEX,
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


def _probe_config_dict(target_modules: str = SUBMISSION_SAFE_WIDE_TARGET_REGEX) -> dict[str, object]:
    return {
        "base_model": "nvidia/Nemotron-3-Nano-30B",
        "model_source": "kagglehub",
        "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
        "trust_remote_code": True,
        "lora": {"rank": 32, "target_modules": target_modules},
        "training": {"output_dir": "artifacts/adapter_stage2_selected_trace"},
    }


def test_tiny_mode_formula_matches_real_zip_within_5pct(tmp_path: Path) -> None:
    config_path = _write_probe_config(tmp_path / "probe_config.yaml")
    output_path = tmp_path / "probe.json"

    payload = probe_adapter_submission_size(
        config=_probe_config_dict(),
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
        config=_probe_config_dict(),
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
        config=_probe_config_dict(),
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
        config=_probe_config_dict(),
        config_path=config_path,
        output_path=output_path,
        tiny_mode=True,
    )

    with safe_open(Path(payload["adapter_model_path"]), framework="np") as handle:
        for key in handle.keys():
            if key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"):
                tensor = handle.get_tensor(key)
                assert str(tensor.dtype) in {"float32", "torch.float32"}


def test_probe_tiny_mode_json_marks_not_calibration(tmp_path: Path) -> None:
    config_path = _write_probe_config(tmp_path / "probe_config.yaml")
    output_path = tmp_path / "probe.json"

    payload = probe_adapter_submission_size(
        config=_probe_config_dict(),
        config_path=config_path,
        output_path=output_path,
        tiny_mode=True,
    )

    assert payload["tiny_mode_is_not_calibration"] is True
    assert "formula_smoke_ratio_after_entropy_pad" in payload
    assert "real_adapter_compression_ratio" not in payload


def test_probe_tiny_mode_respects_shared_expert_size(tmp_path: Path) -> None:
    config_path = _write_probe_config(tmp_path / "probe_config.yaml")
    output_path = tmp_path / "probe.json"

    payload = probe_adapter_submission_size(
        config=_probe_config_dict(),
        config_path=config_path,
        output_path=output_path,
        tiny_mode=True,
    )

    routed_up = budget_module._module_payload_bytes(
        in_features=64,
        out_features=48,
        rank=32,
    )
    shared_up = budget_module._module_payload_bytes(
        in_features=64,
        out_features=72,
        rank=32,
    )
    expected_up_total = TINY_NEMOTRON_LIKE_PAYLOAD["hybrid_override_pattern"].count("E") * (
        routed_up + shared_up
    )

    assert payload["formula_suffix_breakdown"]["up_proj"]["count"] == 6
    assert payload["formula_suffix_breakdown"]["up_proj"]["total_bytes"] == expected_up_total


def test_probe_tiny_mode_uses_realistic_nemotron_peft_key_paths(tmp_path: Path) -> None:
    config_path = _write_probe_config(tmp_path / "probe_config.yaml")
    output_path = tmp_path / "probe.json"

    payload = probe_adapter_submission_size(
        config=_probe_config_dict(),
        config_path=config_path,
        output_path=output_path,
        tiny_mode=True,
    )

    with safe_open(Path(payload["adapter_model_path"]), framework="np") as handle:
        keys = list(handle.keys())

    assert any(key.startswith("base_model.model.backbone.layers.") for key in keys)
    assert any(".mixer.in_proj.lora_A.weight" in key for key in keys)
    assert any(".mixer.experts.0.up_proj.lora_A.weight" in key for key in keys)
    assert any(".mixer.shared_experts.up_proj.lora_A.weight" in key for key in keys)
    assert any(".mixer.q_proj.lora_A.weight" in key for key in keys)


def test_probe_refuses_to_run_with_unsafe_target_modules(tmp_path: Path) -> None:
    config_path = _write_probe_config(
        tmp_path / "unsafe_probe_config.yaml",
        target_modules=_HYPOTHETICAL_OVER_LIMIT_TARGET_REGEX,
    )
    output_path = tmp_path / "probe.json"

    with pytest.raises(ValueError, match="submission-safe"):
        probe_adapter_submission_size(
            config=_probe_config_dict(_HYPOTHETICAL_OVER_LIMIT_TARGET_REGEX),
            config_path=config_path,
            output_path=output_path,
            tiny_mode=True,
        )
