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
NON_ADAPTER_ONLY_CANDIDATE_TYPES = frozenset({"solver_assisted", "prompt_ensemble"})


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
    research_only_reason: str | None = None,
    extra: dict[str, Any] | None = None,
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
    if research_only_reason is not None:
        payload["research_only_reason"] = research_only_reason
    if extra:
        payload.update(extra)
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
            "equation_rescue_v2",
            "train_variant",
            adapter_path="artifacts/adapter_stage2_equation_rescue_v2",
            prompt_profile="chat_thinking",
            data_recipe="equation_rescue_v2",
            family_focus=["equation"],
            gpu_required=True,
            expected_runtime="train+eval",
            artifacts=["adapter", "checkpoints", "raw_predictions", "exact_report", "solver_breakout_v2"],
            config_path="configs/train_stage2_equation_rescue_v2.yaml",
            submission_safe=False,
            research_only_reason="Weak-family v2 data recipe is a training candidate only until a submit-safe adapter is trained and ranked.",
            notes="Equation-focused answer-only/safe-short recipe backed by solver breakout v2 diagnostics.",
        ),
        _candidate(
            "bit_rescue_v2",
            "train_variant",
            adapter_path="artifacts/adapter_stage2_bit_rescue_v2",
            prompt_profile="chat_thinking",
            data_recipe="bit_rescue_v2",
            family_focus=["bit"],
            gpu_required=True,
            expected_runtime="train+eval",
            artifacts=["adapter", "checkpoints", "raw_predictions", "exact_report", "solver_breakout_v2"],
            config_path="configs/train_stage2_bit_rescue_v2.yaml",
            submission_safe=False,
            research_only_reason="Weak-family v2 data recipe is a training candidate only until a submit-safe adapter is trained and ranked.",
            notes="Bit-focused answer-only/safe-short recipe backed by solver breakout v2 diagnostics.",
        ),
        _candidate(
            "eq_bit_rescue_v2",
            "train_variant",
            adapter_path="artifacts/adapter_stage2_eq_bit_rescue_v2",
            prompt_profile="chat_thinking",
            data_recipe="eq_bit_rescue_v2",
            family_focus=["equation", "bit"],
            gpu_required=True,
            expected_runtime="train+eval",
            artifacts=["adapter", "checkpoints", "raw_predictions", "exact_report", "solver_breakout_v2"],
            config_path="configs/train_stage2_eq_bit_rescue_v2.yaml",
            submission_safe=False,
            research_only_reason="Weak-family v2 data recipe is a training candidate only until a submit-safe adapter is trained and ranked.",
            notes="Combined equation+bit v2 recipe for specialist training and later adapter soup.",
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
            "soup_answer_short",
            "merged_adapter",
            adapter_path="artifacts/merged/soup_answer_short",
            prompt_profile="chat_thinking",
            data_recipe="adapter_soup",
            family_focus=["all"],
            gpu_required=True,
            expected_runtime="merge+eval",
            artifacts=["adapter", "merge_manifest", "raw_predictions", "exact_report"],
            notes="Submit-safe linear soup of answer-only and short-trace specialists.",
            extra={
                "merge_method": "linear",
                "merge_sources": [
                    {"candidate": "answer_only_continuation", "weight": 0.5},
                    {"candidate": "short_trace_continuation", "weight": 0.5},
                ],
                "merge_manifest": "artifacts/merged/soup_answer_short/merge_manifest.json",
            },
        ),
        _candidate(
            "soup_eq_bit",
            "merged_adapter",
            adapter_path="artifacts/merged/soup_eq_bit",
            prompt_profile="chat_thinking",
            data_recipe="adapter_soup",
            family_focus=["equation", "bit"],
            gpu_required=True,
            expected_runtime="merge+eval",
            artifacts=["adapter", "merge_manifest", "raw_predictions", "exact_report"],
            notes="Submit-safe soup of weak-family rescue specialists.",
            extra={
                "merge_method": "linear",
                "merge_sources": [
                    {"candidate": "equation_rescue", "weight": 0.35},
                    {"candidate": "bit_rescue", "weight": 0.35},
                    {"candidate": "eq_bit_rescue", "weight": 0.30},
                ],
                "merge_manifest": "artifacts/merged/soup_eq_bit/merge_manifest.json",
            },
        ),
        _candidate(
            "soup_all_rescue",
            "merged_adapter",
            adapter_path="artifacts/merged/soup_all_rescue",
            prompt_profile="chat_thinking",
            data_recipe="adapter_soup",
            family_focus=["all"],
            gpu_required=True,
            expected_runtime="merge+eval",
            artifacts=["adapter", "merge_manifest", "raw_predictions", "exact_report"],
            notes="Submit-safe broad soup across answer, trace, and weak-family specialists.",
            extra={
                "merge_method": "linear",
                "merge_sources": [
                    {"candidate": "answer_only_continuation", "weight": 0.25},
                    {"candidate": "short_trace_continuation", "weight": 0.25},
                    {"candidate": "equation_rescue", "weight": 0.15},
                    {"candidate": "bit_rescue", "weight": 0.15},
                    {"candidate": "eq_bit_rescue", "weight": 0.20},
                ],
                "merge_manifest": "artifacts/merged/soup_all_rescue/merge_manifest.json",
            },
        ),
        _candidate(
            "soup_official_answer_rescue",
            "merged_adapter",
            adapter_path="artifacts/merged/soup_official_answer_rescue",
            prompt_profile="chat_thinking",
            data_recipe="adapter_soup",
            family_focus=["all", "equation", "bit"],
            gpu_required=True,
            expected_runtime="merge+eval",
            artifacts=["adapter", "merge_manifest", "raw_predictions", "exact_report"],
            notes="Submit-safe soup anchored by the 0.57 official-balanced adapter.",
            extra={
                "merge_method": "linear",
                "merge_sources": [
                    {"candidate": DEFAULT_BASELINE_NAME, "weight": 0.40},
                    {"candidate": "answer_only_continuation", "weight": 0.30},
                    {"candidate": "eq_bit_rescue", "weight": 0.30},
                ],
                "merge_manifest": "artifacts/merged/soup_official_answer_rescue/merge_manifest.json",
            },
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
            submission_safe=False,
            research_only_reason="Kaggle submission is adapter-only; solver overrides are not shipped in submission.zip.",
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
            submission_safe=False,
            research_only_reason="Kaggle submission is adapter-only; prompt ensembling is not shipped in submission.zip.",
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
            "If most gains come from solver override, convert them into answer-only or safe short-trace data.",
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
        candidate_type = str(item.get("type", ""))
        if candidate_type in NON_ADAPTER_ONLY_CANDIDATE_TYPES:
            if item.get("submission_safe") is not False:
                errors.append(
                    f"candidate {name or index} type {candidate_type} must be submission_safe=false"
                )
            if not str(item.get("research_only_reason", "")).strip():
                errors.append(
                    f"candidate {name or index} type {candidate_type} must include research_only_reason"
                )
        if candidate_type == "merged_adapter":
            sources = item.get("merge_sources")
            if not isinstance(sources, list) or not sources:
                errors.append(f"candidate {name or index} merged_adapter must include merge_sources")
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
