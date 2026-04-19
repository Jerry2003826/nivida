from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any


KAGGLE_SINGLE_FILE_LIMIT_BYTES = 1_000_000_000
NEMOTRON_PROJECTED_ZIP_RATIO = 0.25
NEMOTRON_PROJECTED_ZIP_OVERHEAD_BYTES = 5_000_000
FLOAT32_BYTES = 4

KNOWN_TARGET_SUFFIXES: tuple[str, ...] = (
    "in_proj",
    "out_proj",
    "up_proj",
    "down_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
)
CANDIDATE_WIDE_SUFFIXES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
)
SAFE_ADDITION_PRIORITY: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
)
DEFAULT_TARGET_REGEX = r".*\.(in_proj|out_proj|up_proj|down_proj)$"
SUBMISSION_SAFE_WIDE_TARGET_REGEX = (
    r".*\.(in_proj|out_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)$"
)
FULL_WIDE_TARGET_REGEX = (
    r".*\.(in_proj|out_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj|gate_proj)$"
)
NEMOTRON_3_NANO_30B_FALLBACK = {
    "model_type": "nemotron_h",
    "hidden_size": 2688,
    "head_dim": 128,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "num_hidden_layers": 52,
    "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
    "n_routed_experts": 128,
    "n_shared_experts": 1,
    "moe_intermediate_size": 1856,
    "mamba_in_proj_out_features": 10304,
    "mamba_out_proj_in_features": 4096,
}


def _resolve_kagglehub_cache_path(config: dict[str, Any]) -> Path | None:
    handle = config.get("model_handle")
    if not handle:
        return None
    configured_cache = dict(config.get("environment", {})).get("KAGGLEHUB_CACHE")
    cache_base = configured_cache or os.environ.get("KAGGLEHUB_CACHE")
    if cache_base:
        cache_root = Path(str(cache_base)) / "models" / Path(str(handle))
    else:
        cache_root = Path.home() / ".cache" / "kagglehub" / "models" / Path(str(handle))
    if not cache_root.exists():
        return None
    version_dirs = [path for path in cache_root.iterdir() if path.is_dir()]
    if not version_dirs:
        return None
    try:
        return max(version_dirs, key=lambda path: int(path.name))
    except ValueError:
        return sorted(version_dirs)[-1]


def _known_model_fallback(config: dict[str, Any]) -> dict[str, Any] | None:
    base_model = str(config.get("base_model", ""))
    model_handle = str(config.get("model_handle", "")).lower()
    base_model_lower = base_model.lower()
    if (
        "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default" in model_handle
        or "nvidia/nemotron-3-nano-30b" in base_model_lower
        or "nemotron-3-nano-30b" in base_model_lower
    ):
        return dict(NEMOTRON_3_NANO_30B_FALLBACK)
    return None


def _load_model_config_payload(config: dict[str, Any]) -> dict[str, Any] | None:
    fallback = _known_model_fallback(config)
    model_source = str(config.get("model_source", "huggingface")).lower()
    model_path = None
    if model_source == "kagglehub":
        model_path = _resolve_kagglehub_cache_path(config)
    elif config.get("base_model"):
        base_model = str(config.get("base_model"))
        base_path = Path(base_model)
        if base_path.exists():
            model_path = base_path
    if model_path is not None:
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            try:
                import json

                payload = json.loads(config_path.read_text(encoding="utf-8"))
                if fallback is not None:
                    merged = dict(fallback)
                    merged.update(payload)
                    return merged
                return payload
            except Exception:
                pass

    return fallback


def _normalise_target_modules(raw_target_modules: Any) -> str | list[str]:
    if isinstance(raw_target_modules, (list, tuple)):
        return [str(item) for item in raw_target_modules]
    if raw_target_modules is None:
        return DEFAULT_TARGET_REGEX
    return str(raw_target_modules)


def _selected_suffixes(target_modules: str | list[str]) -> list[str]:
    if isinstance(target_modules, list):
        selected = []
        for suffix in KNOWN_TARGET_SUFFIXES:
            if suffix in target_modules:
                selected.append(suffix)
        return selected
    pattern = re.compile(target_modules)
    selected: list[str] = []
    for suffix in KNOWN_TARGET_SUFFIXES:
        fake_name = f"model.layers.0.{suffix}"
        if pattern.search(fake_name):
            selected.append(suffix)
    return selected


def _target_suffix_regex(suffixes: list[str]) -> str:
    ordered = [suffix for suffix in KNOWN_TARGET_SUFFIXES if suffix in suffixes]
    if not ordered:
        return DEFAULT_TARGET_REGEX
    return r".*\.(" + "|".join(ordered) + r")$"


def _nemotron_arch_summary(config_payload: dict[str, Any]) -> dict[str, Any] | None:
    if str(config_payload.get("model_type", "")).lower() != "nemotron_h":
        return None
    num_hidden_layers = int(config_payload["num_hidden_layers"])
    hybrid_pattern = str(config_payload.get("hybrid_override_pattern", ""))
    mamba_layers = hybrid_pattern.count("M") if hybrid_pattern else 0
    if mamba_layers <= 0 or mamba_layers > num_hidden_layers:
        return None
    attention_layers = num_hidden_layers - mamba_layers
    experts_per_layer = int(config_payload["n_routed_experts"]) + int(config_payload["n_shared_experts"])
    hidden_size = int(config_payload["hidden_size"])
    head_dim = int(config_payload["head_dim"])
    q_out = int(config_payload["num_attention_heads"]) * head_dim
    kv_out = int(config_payload["num_key_value_heads"]) * head_dim
    moe_intermediate_size = int(config_payload["moe_intermediate_size"])
    return {
        "model_family": "nemotron_h",
        "mamba_layers": mamba_layers,
        "attention_layers": attention_layers,
        "experts_per_layer": experts_per_layer,
        "hidden_size": hidden_size,
        "moe_intermediate_size": moe_intermediate_size,
        "q_out_features": q_out,
        "kv_out_features": kv_out,
        "mamba_in_proj_out_features": int(config_payload["mamba_in_proj_out_features"]),
        "mamba_out_proj_in_features": int(config_payload["mamba_out_proj_in_features"]),
    }


def _module_payload_bytes(*, in_features: int, out_features: int, rank: int) -> int:
    return (rank * int(in_features) + int(out_features) * rank) * FLOAT32_BYTES


def estimate_submission_budget(
    config: dict[str, Any],
    *,
    target_modules: str | list[str] | None = None,
    rank: int | None = None,
) -> dict[str, Any]:
    resolved_target_modules = _normalise_target_modules(
        target_modules if target_modules is not None else dict(config.get("lora", {})).get("target_modules")
    )
    selected_suffixes = _selected_suffixes(resolved_target_modules)
    config_payload = _load_model_config_payload(config)
    if config_payload is None:
        return {
            "status": "unknown",
            "reason": "model config unavailable for submission size estimation",
            "target_modules": resolved_target_modules,
            "selected_suffixes": selected_suffixes,
            "max_submission_zip_bytes": KAGGLE_SINGLE_FILE_LIMIT_BYTES,
        }

    arch = _nemotron_arch_summary(config_payload)
    if arch is None:
        return {
            "status": "unknown",
            "reason": f"unsupported model_type={config_payload.get('model_type')!r} for submission size estimation",
            "target_modules": resolved_target_modules,
            "selected_suffixes": selected_suffixes,
            "max_submission_zip_bytes": KAGGLE_SINGLE_FILE_LIMIT_BYTES,
        }

    resolved_rank = int(rank if rank is not None else dict(config.get("lora", {})).get("rank", 16))
    per_suffix_counts = {
        "in_proj": arch["mamba_layers"],
        "out_proj": arch["mamba_layers"],
        "up_proj": arch["mamba_layers"] * arch["experts_per_layer"],
        "down_proj": arch["mamba_layers"] * arch["experts_per_layer"],
        "gate_proj": arch["mamba_layers"] * arch["experts_per_layer"],
        "q_proj": arch["attention_layers"],
        "k_proj": arch["attention_layers"],
        "v_proj": arch["attention_layers"],
        "o_proj": arch["attention_layers"],
    }
    per_suffix_bytes = {
        "in_proj": _module_payload_bytes(
            in_features=arch["hidden_size"],
            out_features=arch["mamba_in_proj_out_features"],
            rank=resolved_rank,
        ),
        "out_proj": _module_payload_bytes(
            in_features=arch["mamba_out_proj_in_features"],
            out_features=arch["hidden_size"],
            rank=resolved_rank,
        ),
        "up_proj": _module_payload_bytes(
            in_features=arch["hidden_size"],
            out_features=arch["moe_intermediate_size"],
            rank=resolved_rank,
        ),
        "down_proj": _module_payload_bytes(
            in_features=arch["moe_intermediate_size"],
            out_features=arch["hidden_size"],
            rank=resolved_rank,
        ),
        "gate_proj": _module_payload_bytes(
            in_features=arch["hidden_size"],
            out_features=arch["moe_intermediate_size"],
            rank=resolved_rank,
        ),
        "q_proj": _module_payload_bytes(
            in_features=arch["hidden_size"],
            out_features=arch["q_out_features"],
            rank=resolved_rank,
        ),
        "k_proj": _module_payload_bytes(
            in_features=arch["hidden_size"],
            out_features=arch["kv_out_features"],
            rank=resolved_rank,
        ),
        "v_proj": _module_payload_bytes(
            in_features=arch["hidden_size"],
            out_features=arch["kv_out_features"],
            rank=resolved_rank,
        ),
        "o_proj": _module_payload_bytes(
            in_features=arch["q_out_features"],
            out_features=arch["hidden_size"],
            rank=resolved_rank,
        ),
    }

    suffix_breakdown: dict[str, dict[str, int]] = {}
    projected_adapter_bytes = 0
    for suffix in selected_suffixes:
        count = per_suffix_counts[suffix]
        total_bytes = per_suffix_bytes[suffix] * count
        suffix_breakdown[suffix] = {
            "count": count,
            "bytes_per_module": per_suffix_bytes[suffix],
            "total_bytes": total_bytes,
        }
        projected_adapter_bytes += total_bytes

    projected_submission_zip_bytes = int(
        math.ceil(projected_adapter_bytes * NEMOTRON_PROJECTED_ZIP_RATIO + NEMOTRON_PROJECTED_ZIP_OVERHEAD_BYTES)
    )
    within_budget = projected_submission_zip_bytes <= KAGGLE_SINGLE_FILE_LIMIT_BYTES
    return {
        "status": "ok" if within_budget else "over_limit",
        "model_family": arch["model_family"],
        "target_modules": resolved_target_modules,
        "selected_suffixes": selected_suffixes,
        "rank": resolved_rank,
        "suffix_breakdown": suffix_breakdown,
        "projected_adapter_bytes": projected_adapter_bytes,
        "projected_submission_zip_bytes": projected_submission_zip_bytes,
        "max_submission_zip_bytes": KAGGLE_SINGLE_FILE_LIMIT_BYTES,
        "compression_ratio_estimate": NEMOTRON_PROJECTED_ZIP_RATIO,
        "compression_overhead_bytes": NEMOTRON_PROJECTED_ZIP_OVERHEAD_BYTES,
        "within_budget": within_budget,
        "reason": (
            None
            if within_budget
            else (
                f"projected submission zip size {projected_submission_zip_bytes} exceeds "
                f"Kaggle limit {KAGGLE_SINGLE_FILE_LIMIT_BYTES}"
            )
        ),
    }


def propose_size_safe_target_modules(
    config: dict[str, Any],
    *,
    current_target_modules: str | list[str] | None = None,
    rank: int | None = None,
) -> dict[str, Any]:
    resolved_target_modules = _normalise_target_modules(
        current_target_modules if current_target_modules is not None else dict(config.get("lora", {})).get("target_modules")
    )
    current_suffixes = _selected_suffixes(resolved_target_modules)
    if not current_suffixes:
        return {
            "status": "unknown",
            "reason": "could not resolve current target module suffixes",
            "current_target_modules": resolved_target_modules,
        }

    current_budget = estimate_submission_budget(
        config,
        target_modules=resolved_target_modules,
        rank=rank,
    )
    if current_budget["status"] == "unknown":
        return {
            "status": "unknown",
            "reason": current_budget["reason"],
            "current_target_modules": resolved_target_modules,
            "current_budget": current_budget,
        }

    proposed_suffixes = [suffix for suffix in current_suffixes if suffix not in CANDIDATE_WIDE_SUFFIXES]
    if not proposed_suffixes:
        proposed_suffixes = list(current_suffixes)
    search_base_suffixes = list(proposed_suffixes)
    budget_blocked_suffixes: list[str] = []
    candidate_evaluations: dict[str, dict[str, Any]] = {}
    for suffix in SAFE_ADDITION_PRIORITY:
        if suffix in proposed_suffixes:
            continue
        trial_suffixes = proposed_suffixes + [suffix]
        trial_budget = estimate_submission_budget(
            config,
            target_modules=_target_suffix_regex(trial_suffixes),
            rank=rank,
        )
        candidate_evaluations[suffix] = {
            "projected_submission_zip_bytes": trial_budget.get("projected_submission_zip_bytes"),
            "status": trial_budget["status"],
            "within_budget": trial_budget.get("within_budget"),
        }
        if trial_budget["status"] == "ok":
            proposed_suffixes = trial_suffixes
        else:
            budget_blocked_suffixes.append(suffix)

    proposed_regex = _target_suffix_regex(proposed_suffixes)
    proposed_budget = estimate_submission_budget(
        config,
        target_modules=proposed_regex,
        rank=rank,
    )
    full_wide_budget = estimate_submission_budget(
        config,
        target_modules=FULL_WIDE_TARGET_REGEX,
        rank=rank,
    )
    return {
        "status": "ok",
        "current_target_modules": resolved_target_modules,
        "current_budget": current_budget,
        "search_base_suffixes": search_base_suffixes,
        "proposed_target_suffixes": proposed_suffixes,
        "proposed_regex": proposed_regex,
        "proposed_budget": proposed_budget,
        "candidate_evaluations": candidate_evaluations,
        "budget_blocked_suffixes": budget_blocked_suffixes,
        "full_wide_regex": FULL_WIDE_TARGET_REGEX,
        "full_wide_budget": full_wide_budget,
    }


def ensure_submission_budget_safe(config: dict[str, Any]) -> dict[str, Any]:
    budget = estimate_submission_budget(config)
    if budget["status"] == "over_limit":
        raise ValueError(
            "Configured LoRA target_modules are not submission-safe: "
            f"{budget['reason']}. "
            "Use scripts/list_model_linear_modules.py to derive a size-safe alternative."
        )
    return budget
