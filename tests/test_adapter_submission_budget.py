from __future__ import annotations

import pytest

import src.student.adapter_submission_budget as budget_module
from src.common.io import read_yaml
from src.student.adapter_submission_budget import (
    CANDIDATE_WIDE_SUFFIXES,
    DEFAULT_TARGET_REGEX,
    FULL_WIDE_TARGET_REGEX,
    KNOWN_TARGET_SUFFIXES,
    NEMOTRON_3_NANO_30B_FALLBACK,
    SAFE_ADDITION_PRIORITY,
    SUBMISSION_SAFE_WIDE_TARGET_REGEX,
    _HYPOTHETICAL_OVER_LIMIT_TARGET_REGEX,
    estimate_submission_budget,
    propose_size_safe_target_modules,
)


def _nemotron_config() -> dict[str, object]:
    return {
        "base_model": "nvidia/Nemotron-3-Nano-30B",
        "model_source": "kagglehub",
        "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
        "trust_remote_code": True,
        "lora": {"rank": 32, "target_modules": DEFAULT_TARGET_REGEX},
    }


def test_estimate_submission_budget_marks_hypothetical_wide_regex_over_limit() -> None:
    budget = estimate_submission_budget(
        _nemotron_config(),
        target_modules=_HYPOTHETICAL_OVER_LIMIT_TARGET_REGEX,
        rank=32,
    )

    assert budget["status"] == "over_limit"
    assert budget["within_budget"] is False
    assert budget["projected_submission_zip_bytes"] > budget["max_submission_zip_bytes"]
    assert "hypothetical_bulk_proj" in budget["selected_suffixes"]


def test_full_wide_regex_is_equivalent_to_submission_safe_regex() -> None:
    assert FULL_WIDE_TARGET_REGEX == SUBMISSION_SAFE_WIDE_TARGET_REGEX


def test_nemotron_layer_count_matches_redhatai_spec() -> None:
    budget = estimate_submission_budget(_nemotron_config())

    assert budget["status"] == "ok"
    assert budget["mamba_layers"] == 23
    assert budget["moe_layers"] == 23
    assert budget["attention_layers"] == 6
    assert budget["n_routed_experts_per_layer"] == 128
    assert budget["n_shared_experts_per_layer"] == 1


def test_propose_size_safe_target_modules_reaches_canonical_safe_regex() -> None:
    recommendation = propose_size_safe_target_modules(_nemotron_config())

    assert recommendation["status"] == "ok"
    assert recommendation["proposed_budget"]["status"] == "ok"
    assert recommendation["proposed_regex"] == SUBMISSION_SAFE_WIDE_TARGET_REGEX
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= set(
        recommendation["proposed_target_suffixes"]
    )
    assert "gate_proj" not in recommendation["proposed_target_suffixes"]
    assert "gate_proj" not in recommendation["budget_blocked_suffixes"]


def test_propose_size_safe_target_modules_recovers_from_hypothetical_over_limit() -> None:
    recommendation = propose_size_safe_target_modules(
        _nemotron_config(),
        current_target_modules=_HYPOTHETICAL_OVER_LIMIT_TARGET_REGEX,
    )

    assert recommendation["status"] == "ok"
    assert recommendation["current_budget"]["status"] == "over_limit"
    assert recommendation["proposed_regex"] == SUBMISSION_SAFE_WIDE_TARGET_REGEX
    assert recommendation["proposed_budget"]["status"] == "ok"


def test_estimate_submission_budget_returns_unknown_for_unsupported_model() -> None:
    budget = estimate_submission_budget(
        {
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "model_source": "huggingface",
            "trust_remote_code": False,
            "lora": {"rank": 16, "target_modules": ["q_proj", "v_proj"]},
        }
    )

    assert budget["status"] == "unknown"


def test_invalid_hybrid_pattern_returns_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    invalid_payload = dict(NEMOTRON_3_NANO_30B_FALLBACK)
    invalid_payload["hybrid_override_pattern"] = "MME"
    monkeypatch.setattr(
        budget_module,
        "_load_model_config_payload",
        lambda _config: invalid_payload,
    )

    budget = estimate_submission_budget(_nemotron_config())

    assert budget["status"] == "unknown"
    assert "pattern" in str(budget["reason"]).lower()


def test_estimate_submission_budget_known_suffixes_are_real_nemotron_modules() -> None:
    assert "gate_proj" not in KNOWN_TARGET_SUFFIXES
    assert "gate_proj" not in CANDIDATE_WIDE_SUFFIXES
    assert "gate_proj" not in SAFE_ADDITION_PRIORITY


def test_shared_expert_uses_moe_shared_expert_intermediate_size() -> None:
    budget = estimate_submission_budget(
        _nemotron_config(),
        target_modules=r".*\.(up_proj|down_proj)$",
        rank=32,
    )

    per_routed = budget_module._module_payload_bytes(
        in_features=2688,
        out_features=1856,
        rank=32,
    )
    per_shared = budget_module._module_payload_bytes(
        in_features=2688,
        out_features=3712,
        rank=32,
    )
    expected_up_total = 23 * (128 * per_routed + per_shared)

    assert budget["suffix_breakdown"]["up_proj"]["count"] == 23 * 129
    assert budget["suffix_breakdown"]["up_proj"]["total_bytes"] == expected_up_total
    assert budget["suffix_breakdown"]["up_proj"]["total_bytes"] > 23 * 129 * per_routed


def test_canonical_projected_zip_matches_analytic_value() -> None:
    budget = estimate_submission_budget(
        _nemotron_config(),
        target_modules=SUBMISSION_SAFE_WIDE_TARGET_REGEX,
        rank=32,
    )

    assert budget["status"] == "ok"
    assert budget["projected_adapter_bytes"] == 3_535_495_168
    assert budget["projected_submission_zip_bytes"] == 888_873_792


def test_canonical_stage_configs_share_submission_safe_target_modules() -> None:
    config_paths = [
        "configs/train_stage1_format.yaml",
        "configs/train_stage2_selected_trace.yaml",
        "configs/train_stage2_selected_trace_subtype_rescue.yaml",
        "configs/train_stage3_repair.yaml",
    ]

    observed_regexes = {
        read_yaml(path)["lora"]["target_modules"]
        for path in config_paths
    }
    assert observed_regexes == {SUBMISSION_SAFE_WIDE_TARGET_REGEX}

    for path in config_paths:
        budget = estimate_submission_budget(read_yaml(path))
        assert budget["status"] == "ok", path
        assert budget["target_modules"] == SUBMISSION_SAFE_WIDE_TARGET_REGEX, path
        assert 880_000_000 <= budget["projected_submission_zip_bytes"] <= 900_000_000, (
            f"{path} projected zip {budget['projected_submission_zip_bytes']} "
            "fell outside the expected 880M-900M canonical window"
        )
