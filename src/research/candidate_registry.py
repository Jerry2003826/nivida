from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.io import read_json, write_json


DEFAULT_BASELINE_NAME = "official_balanced"
DEFAULT_BASELINE_PUBLIC_SCORE = 0.57
REQUIRED_CANDIDATE_FIELDS = (
    "name",
    "type",
    "adapter_path",
    "prompt_profile",
    "data_recipe",
    "family_focus",
    "gpu_required",
    "expected_runtime",
    "artifacts",
)


@dataclass(frozen=True, slots=True)
class CandidateValidationError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _candidate(
    name: str,
    kind: str,
    *,
    adapter_path: str,
    prompt_profile: str,
    data_recipe: str,
    family_focus: list[str],
    gpu_required: bool,
    expected_runtime: str,
    artifacts: list[str],
    config_path: str | None = None,
    enabled: bool = True,
    submission_safe: bool = True,
    notes: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "type": kind,
        "adapter_path": adapter_path,
        "prompt_profile": prompt_profile,
        "data_recipe": data_recipe,
        "family_focus": family_focus,
        "gpu_required": gpu_required,
        "expected_runtime": expected_runtime,
        "artifacts": artifacts,
        "enabled": enabled,
        "submission_safe": submission_safe,
        "notes": notes,
    }
    if config_path is not None:
        payload["config_path"] = config_path
    return payload


def default_candidates() -> list[dict[str, Any]]:
    """Return the fixed research-breakout candidate matrix.

    The rank-64 variant is retained as an explicit research candidate but is
    marked submission-unsafe because the current Kaggle runtime contract uses
    ``max_lora_rank=32``.
    """

    return [
        _candidate(
            "b_thin",
            "reference_adapter",
            adapter_path="artifacts/adapter_stage2_thin",
            prompt_profile="chat_thinking",
            data_recipe="stage2_thin",
            family_focus=["all"],
            gpu_required=True,
            expected_runtime="eval-only",
            artifacts=["raw_predictions", "exact_report"],
            notes="Historical 0.54 baseline; never use as the primary ranking baseline.",
        ),
        _candidate(
            DEFAULT_BASELINE_NAME,
            "leaderboard_baseline",
            adapter_path="artifacts/adapter_stage2_thin_official_balanced_20260424_161110Z",
            prompt_profile="chat_thinking",
            data_recipe="official_balanced",
            family_focus=["all"],
            gpu_required=True,
            expected_runtime="eval-only",
            artifacts=["raw_predictions", "exact_report"],
            notes="Current public LB baseline: 0.57.",
        ),
        _candidate(
            "answer_only_continuation",
            "train_variant",
            adapter_path="artifacts/adapter_stage2_official_balanced_answer_only",
            prompt_profile="chat_thinking",
            data_recipe="answer_only",
            family_focus=["all"],
            gpu_required=True,
            expected_runtime="train+eval",
            artifacts=["adapter", "checkpoints", "raw_predictions", "exact_report"],
            config_path="configs/train_stage2_official_balanced_answer_only.yaml",
        ),
        _candidate(
            "short_trace_continuation",
            "train_variant",
            adapter_path="artifacts/adapter_stage2_official_balanced_short_trace",
            prompt_profile="chat_thinking",
            data_recipe="short_trace",
            family_focus=["all"],
            gpu_required=True,
            expected_runtime="train+eval",
            artifacts=["adapter", "checkpoints", "raw_predictions", "exact_report"],
            config_path="configs/train_stage2_official_balanced_short_trace.yaml",
        ),
        _candidate(
            "mixed_answer_short",
            "train_variant",
            adapter_path="artifacts/adapter_stage2_mixed_answer_short",
            prompt_profile="chat_thinking",
            data_recipe="mixed_answer_short",
            family_focus=["all"],
            gpu_required=True,
            expected_runtime="train+eval",
            artifacts=["adapter", "checkpoints", "raw_predictions", "exact_report"],
            config_path="configs/train_stage2_mixed_answer_short.yaml",
        ),
        _candidate(
            "equation_rescue",
            "train_variant",
            adapter_path="artifacts/adapter_stage2_equation_rescue",
            prompt_profile="chat_thinking",
            data_recipe="equation_rescue",
            family_focus=["equation"],
            gpu_required=True,
            expected_runtime="train+eval",
            artifacts=["adapter", "checkpoints", "raw_predictions", "exact_report"],
            config_path="configs/train_stage2_equation_rescue.yaml",
        ),
        _candidate(
            "bit_rescue",
            "train_variant",
            adapter_path="artifacts/adapter_stage2_bit_rescue",
            prompt_profile="chat_thinking",
            data_recipe="bit_rescue",
            family_focus=["bit"],
            gpu_required=True,
            expected_runtime="train+eval",
            artifacts=["adapter", "checkpoints", "raw_predictions", "exact_report"],
            config_path="configs/train_stage2_bit_rescue.yaml",
        ),
        _candidate(
            "eq_bit_rescue",
            "train_variant",
            adapter_path="artifacts/adapter_stage2_eq_bit_rescue",
            prompt_profile="chat_thinking",
            data_recipe="eq_bit_rescue",
            family_focus=["equation", "bit"],
            gpu_required=True,
            expected_runtime="train+eval",
            artifacts=["adapter", "checkpoints", "raw_predictions", "exact_report"],
            config_path="configs/train_stage2_eq_bit_rescue.yaml",
        ),
        _candidate(
            "rank64_answer_only",
            "train_variant",
            adapter_path="artifacts/adapter_stage2_rank64_answer_only",
            prompt_profile="chat_thinking",
            data_recipe="answer_only",
            family_focus=["all"],
            gpu_required=True,
            expected_runtime="train+eval",
            artifacts=["adapter", "checkpoints", "raw_predictions", "exact_report"],
            config_path="configs/train_stage2_rank64_answer_only.yaml",
            submission_safe=False,
            notes="Research-only unless Kaggle max_lora_rank changes above 32.",
        ),
        _candidate(
            "final_answer_weighted_loss",
            "train_variant",
            adapter_path="artifacts/adapter_stage2_final_answer_weighted",
            prompt_profile="chat_thinking",
            data_recipe="mixed_answer_short",
            family_focus=["all"],
            gpu_required=True,
            expected_runtime="train+eval",
            artifacts=["adapter", "checkpoints", "raw_predictions", "exact_report"],
            config_path="configs/train_stage2_final_answer_weighted.yaml",
        ),
        _candidate(
            "official_balanced_solver_assisted",
            "solver_assisted",
            adapter_path="artifacts/adapter_stage2_thin_official_balanced_20260424_161110Z",
            prompt_profile="chat_thinking",
            data_recipe="official_balanced",
            family_focus=["equation", "bit"],
            gpu_required=False,
            expected_runtime="postprocess-only",
            artifacts=["solver_finalized_predictions", "exact_report"],
            notes="CPU postprocess candidate: override only when solver/verifier is high confidence.",
        ),
        _candidate(
            "official_balanced_prompt_ensemble",
            "prompt_ensemble",
            adapter_path="artifacts/adapter_stage2_thin_official_balanced_20260424_161110Z",
            prompt_profile="short_answer_biased+format_strict",
            data_recipe="official_balanced",
            family_focus=["equation", "bit"],
            gpu_required=True,
            expected_runtime="eval-only",
            artifacts=["profiled_manifests", "raw_predictions", "ensemble_report"],
            notes="Small weak-family prompt ensemble; use only after deterministic smoke.",
        ),
    ]


def build_default_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "baseline": DEFAULT_BASELINE_NAME,
        "baseline_public_score": DEFAULT_BASELINE_PUBLIC_SCORE,
        "submission_policy": {
            "min_overall_delta": 0.0,
            "max_family_regression_samples": 1.0,
            "require_boxed_valid_not_below_baseline": True,
            "max_kaggle_submissions_per_batch": 2,
        },
        "stop_rules": [
            "If two GPU batches improve local exact but not Kaggle, pause training and recalibrate eval.",
            "If most gains come from solver override, prioritize inference-time pipeline over LoRA training.",
            "Do not submit route/shared transplant candidates unless exact arena proves stable gain.",
        ],
        "prompt_profiles": [
            "chat_thinking",
            "short_answer_biased",
            "format_strict",
        ],
        "candidates": default_candidates(),
    }


def validate_registry(
    registry: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
    check_paths: bool = False,
) -> list[str]:
    errors: list[str] = []
    candidates = registry.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ["registry must contain a non-empty candidates list"]

    names: set[str] = set()
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            errors.append(f"candidate[{index}] must be an object")
            continue
        missing = [field for field in REQUIRED_CANDIDATE_FIELDS if field not in item]
        if missing:
            errors.append(f"candidate[{index}] missing fields: {', '.join(missing)}")
        name = str(item.get("name", ""))
        if not name:
            errors.append(f"candidate[{index}] has empty name")
        if name in names:
            errors.append(f"duplicate candidate name: {name}")
        names.add(name)
        if not isinstance(item.get("family_focus"), list) or not item.get("family_focus"):
            errors.append(f"candidate {name or index} family_focus must be a non-empty list")
        if not isinstance(item.get("artifacts"), list):
            errors.append(f"candidate {name or index} artifacts must be a list")
        if check_paths and repo_root is not None:
            root = Path(repo_root)
            for key in ("adapter_path", "config_path"):
                raw = item.get(key)
                if raw and not (root / str(raw)).exists():
                    errors.append(f"candidate {name} {key} does not exist: {raw}")

    baseline = str(registry.get("baseline", ""))
    if baseline != DEFAULT_BASELINE_NAME:
        errors.append(f"baseline must be {DEFAULT_BASELINE_NAME!r}, got {baseline!r}")
    if baseline not in names:
        errors.append(f"baseline candidate {baseline!r} missing from registry")
    return errors


def load_registry(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise CandidateValidationError(f"registry must be a JSON object: {path}")
    errors = validate_registry(payload)
    if errors:
        raise CandidateValidationError("; ".join(errors))
    return payload


def write_default_registry(path: str | Path) -> dict[str, Any]:
    payload = build_default_registry()
    write_json(path, payload)
    return payload


def canonical_registry_matches(path: str | Path) -> bool:
    target = Path(path)
    if not target.is_file():
        return False
    return read_json(target) == build_default_registry()


def candidate_by_name(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): dict(item) for item in registry.get("candidates", [])}


def adapter_env_value(
    registry: dict[str, Any],
    *,
    include_types: set[str] | None = None,
    only_enabled: bool = True,
) -> str:
    pieces: list[str] = []
    for item in registry.get("candidates", []):
        if only_enabled and item.get("enabled") is False:
            continue
        if include_types is not None and str(item.get("type")) not in include_types:
            continue
        adapter_path = str(item.get("adapter_path") or "")
        if not adapter_path:
            continue
        pieces.append(f"{item['name']}={adapter_path}")
    return ",".join(pieces)


def registry_with_updated_candidate(
    registry: dict[str, Any],
    name: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    clone = deepcopy(registry)
    for item in clone.get("candidates", []):
        if item.get("name") == name:
            item.update(updates)
            return clone
    raise KeyError(name)

