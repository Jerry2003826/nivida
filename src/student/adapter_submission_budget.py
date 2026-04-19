from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any


KAGGLE_SINGLE_FILE_LIMIT_BYTES = 1_000_000_000

# peft saves LoRA adapter weights (lora_A, lora_B) as float32 by default:
# they are new trainable parameters and do not inherit the base model's
# bf16/fp16 dtype. Verified against adapter_model.safetensors on commit
# c40b027, where keys such as
# base_model.model.backbone.layers.0.mixer.in_proj.lora_A.weight carry
# dtype=float32. Re-calibrate only after rerunning the probe script and its
# float32 dtype guard.
ADAPTER_WEIGHT_BYTES = 4

# Empirical zip compression ratio for float32 LoRA weight data packed with
# ZIP_DEFLATED. Round-1 probe: formula estimate 3_509_624_832 bytes vs
# measured zip 784_310_174 bytes => implied ratio ~0.2234. We keep 0.25 as a
# conservative upper bound so the budget guard trips before a real submission
# can exceed 1 GB.
NEMOTRON_ZIP_COMPRESSION_RATIO = 0.25
NEMOTRON_ZIP_OVERHEAD_BYTES = 5_000_000

# Back-compat alias for any external tooling that still imports the old name;
# prefer NEMOTRON_ZIP_COMPRESSION_RATIO in new code.
NEMOTRON_PROJECTED_ZIP_RATIO = NEMOTRON_ZIP_COMPRESSION_RATIO
NEMOTRON_PROJECTED_ZIP_OVERHEAD_BYTES = NEMOTRON_ZIP_OVERHEAD_BYTES

KNOWN_TARGET_SUFFIXES: tuple[str, ...] = (
    "in_proj",
    "out_proj",
    "up_proj",
    "down_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)
CANDIDATE_WIDE_SUFFIXES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)
SAFE_ADDITION_PRIORITY: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)
DEFAULT_TARGET_REGEX = r".*\.(in_proj|out_proj|up_proj|down_proj)$"
SUBMISSION_SAFE_WIDE_TARGET_REGEX = (
    r".*\.(in_proj|out_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)$"
)
# NOTE: gate_proj is intentionally absent. The current HF Nemotron-H
# implementation does not expose a gate_proj nn.Linear. NemotronHMLP has only
# up_proj and down_proj; NemotronHMOE.gate is a router module, not a
# LoRA-compatible Linear. If a future model revision adds gate_proj as a real
# Linear, update these suffix lists only after verifying it with
# scripts/list_model_linear_modules.py against the actual model source.
FULL_WIDE_TARGET_REGEX = SUBMISSION_SAFE_WIDE_TARGET_REGEX
_HYPOTHETICAL_OVER_LIMIT_TARGET_REGEX = (
    r".*\.(in_proj|out_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj|hypothetical_bulk_proj)$"
)
_ALL_RECOGNISED_SUFFIXES: tuple[str, ...] = KNOWN_TARGET_SUFFIXES + (
    "hypothetical_bulk_proj",
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
    "moe_shared_expert_intermediate_size": 3712,
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
        for suffix in _ALL_RECOGNISED_SUFFIXES:
            if suffix in target_modules:
                selected.append(suffix)
        return selected
    pattern = re.compile(target_modules)
    selected: list[str] = []
    for suffix in _ALL_RECOGNISED_SUFFIXES:
        fake_name = f"model.layers.0.{suffix}"
        if pattern.search(fake_name):
            selected.append(suffix)
    return selected


def _target_suffix_regex(suffixes: list[str]) -> str:
    ordered = [suffix for suffix in _ALL_RECOGNISED_SUFFIXES if suffix in suffixes]
    if not ordered:
        return DEFAULT_TARGET_REGEX
    return r".*\.(" + "|".join(ordered) + r")$"


def _parse_nemotron_hybrid_pattern(
    hybrid_pattern: str,
    *,
    num_hidden_layers: int,
) -> tuple[dict[str, int] | None, str | None]:
    if not hybrid_pattern:
        return None, "invalid hybrid pattern: missing hybrid_override_pattern"
    invalid_chars = sorted({char for char in hybrid_pattern if char not in {"M", "E", "*", "-"}})
    if invalid_chars:
        return (
            None,
            "invalid hybrid pattern: unsupported characters "
            + ", ".join(repr(char) for char in invalid_chars),
        )

    mamba_layers = hybrid_pattern.count("M")
    moe_layers = hybrid_pattern.count("E")
    attention_layers = hybrid_pattern.count("*")
    accounted = mamba_layers + moe_layers + attention_layers
    if accounted != num_hidden_layers:
        return (
            None,
            "invalid hybrid pattern: accounted "
            f"{accounted} layers but num_hidden_layers={num_hidden_layers}",
        )

    return {
        "mamba_layers": mamba_layers,
        "moe_layers": moe_layers,
        "attention_layers": attention_layers,
    }, None


def _nemotron_arch_summary(
    config_payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    if str(config_payload.get("model_type", "")).lower() != "nemotron_h":
        return None, None
    num_hidden_layers = int(config_payload["num_hidden_layers"])
    hybrid_pattern = str(config_payload.get("hybrid_override_pattern", ""))
    layer_counts, layer_error = _parse_nemotron_hybrid_pattern(
        hybrid_pattern,
        num_hidden_layers=num_hidden_layers,
    )
    if layer_counts is None:
        return None, layer_error
    routed_experts = int(config_payload["n_routed_experts"])
    shared_experts = int(config_payload["n_shared_experts"])
    experts_per_layer = routed_experts + shared_experts
    hidden_size = int(config_payload["hidden_size"])
    head_dim = int(config_payload["head_dim"])
    q_out = int(config_payload["num_attention_heads"]) * head_dim
    kv_out = int(config_payload["num_key_value_heads"]) * head_dim
    return {
        "model_family": "nemotron_h",
        "mamba_layers": layer_counts["mamba_layers"],
        "moe_layers": layer_counts["moe_layers"],
        "attention_layers": layer_counts["attention_layers"],
        "experts_per_layer": experts_per_layer,
        "n_routed_experts_per_layer": routed_experts,
        "n_shared_experts_per_layer": shared_experts,
        "hidden_size": hidden_size,
        "moe_intermediate_size": int(config_payload["moe_intermediate_size"]),
        "moe_shared_expert_intermediate_size": int(
            config_payload.get(
                "moe_shared_expert_intermediate_size",
                config_payload["moe_intermediate_size"],
            )
        ),
        "q_out_features": q_out,
        "kv_out_features": kv_out,
        "mamba_in_proj_out_features": int(config_payload["mamba_in_proj_out_features"]),
        "mamba_out_proj_in_features": int(config_payload["mamba_out_proj_in_features"]),
    }, None


def _module_payload_bytes(*, in_features: int, out_features: int, rank: int) -> int:
    return (rank * int(in_features) + int(out_features) * rank) * ADAPTER_WEIGHT_BYTES


def _nemotron_per_suffix_counts(arch: dict[str, Any]) -> dict[str, int]:
    moe_module_count = arch["moe_layers"] * (
        arch["n_routed_experts_per_layer"] + arch["n_shared_experts_per_layer"]
    )
    return {
        "in_proj": arch["mamba_layers"],
        "out_proj": arch["mamba_layers"],
        "up_proj": moe_module_count,
        "down_proj": moe_module_count,
        "q_proj": arch["attention_layers"],
        "k_proj": arch["attention_layers"],
        "v_proj": arch["attention_layers"],
        "o_proj": arch["attention_layers"],
        "hypothetical_bulk_proj": moe_module_count,
    }


def _nemotron_per_suffix_total_bytes(
    arch: dict[str, Any],
    *,
    rank: int,
) -> dict[str, int]:
    hidden = arch["hidden_size"]
    routed_int = arch["moe_intermediate_size"]
    shared_int = arch["moe_shared_expert_intermediate_size"]
    routed_count = arch["n_routed_experts_per_layer"]
    shared_count = arch["n_shared_experts_per_layer"]
    moe_layers = arch["moe_layers"]
    moe_counts = _nemotron_per_suffix_counts(arch)

    up_total_per_moe_layer = (
        routed_count
        * _module_payload_bytes(
            in_features=hidden,
            out_features=routed_int,
            rank=rank,
        )
        + shared_count
        * _module_payload_bytes(
            in_features=hidden,
            out_features=shared_int,
            rank=rank,
        )
    )
    down_total_per_moe_layer = (
        routed_count
        * _module_payload_bytes(
            in_features=routed_int,
            out_features=hidden,
            rank=rank,
        )
        + shared_count
        * _module_payload_bytes(
            in_features=shared_int,
            out_features=hidden,
            rank=rank,
        )
    )

    return {
        "in_proj": _module_payload_bytes(
            in_features=hidden,
            out_features=arch["mamba_in_proj_out_features"],
            rank=rank,
        )
        * arch["mamba_layers"],
        "out_proj": _module_payload_bytes(
            in_features=arch["mamba_out_proj_in_features"],
            out_features=hidden,
            rank=rank,
        )
        * arch["mamba_layers"],
        "up_proj": up_total_per_moe_layer * moe_layers,
        "down_proj": down_total_per_moe_layer * moe_layers,
        "q_proj": _module_payload_bytes(
            in_features=hidden,
            out_features=arch["q_out_features"],
            rank=rank,
        )
        * arch["attention_layers"],
        "k_proj": _module_payload_bytes(
            in_features=hidden,
            out_features=arch["kv_out_features"],
            rank=rank,
        )
        * arch["attention_layers"],
        "v_proj": _module_payload_bytes(
            in_features=hidden,
            out_features=arch["kv_out_features"],
            rank=rank,
        )
        * arch["attention_layers"],
        "o_proj": _module_payload_bytes(
            in_features=arch["q_out_features"],
            out_features=hidden,
            rank=rank,
        )
        * arch["attention_layers"],
        "hypothetical_bulk_proj": _module_payload_bytes(
            in_features=hidden,
            out_features=routed_int,
            rank=rank,
        )
        * moe_counts["hypothetical_bulk_proj"],
    }


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

    arch, arch_error = _nemotron_arch_summary(config_payload)
    if arch is None:
        return {
            "status": "unknown",
            "reason": (
                arch_error
                or f"unsupported model_type={config_payload.get('model_type')!r} for submission size estimation"
            ),
            "target_modules": resolved_target_modules,
            "selected_suffixes": selected_suffixes,
            "max_submission_zip_bytes": KAGGLE_SINGLE_FILE_LIMIT_BYTES,
        }

    resolved_rank = int(rank if rank is not None else dict(config.get("lora", {})).get("rank", 16))
    per_suffix_counts = _nemotron_per_suffix_counts(arch)
    per_suffix_total_bytes = _nemotron_per_suffix_total_bytes(
        arch,
        rank=resolved_rank,
    )

    suffix_breakdown: dict[str, dict[str, int]] = {}
    projected_adapter_bytes = 0
    for suffix in selected_suffixes:
        count = per_suffix_counts[suffix]
        total_bytes = per_suffix_total_bytes[suffix]
        suffix_breakdown[suffix] = {
            "count": count,
            "bytes_per_module": int(total_bytes // max(1, count)),
            "total_bytes": total_bytes,
        }
        projected_adapter_bytes += total_bytes

    projected_submission_zip_bytes = int(
        math.ceil(projected_adapter_bytes * NEMOTRON_ZIP_COMPRESSION_RATIO + NEMOTRON_ZIP_OVERHEAD_BYTES)
    )
    within_budget = projected_submission_zip_bytes <= KAGGLE_SINGLE_FILE_LIMIT_BYTES
    return {
        "status": "ok" if within_budget else "over_limit",
        "model_family": arch["model_family"],
        "mamba_layers": arch["mamba_layers"],
        "moe_layers": arch["moe_layers"],
        "attention_layers": arch["attention_layers"],
        "n_routed_experts_per_layer": arch["n_routed_experts_per_layer"],
        "n_shared_experts_per_layer": arch["n_shared_experts_per_layer"],
        "target_modules": resolved_target_modules,
        "selected_suffixes": selected_suffixes,
        "rank": resolved_rank,
        "suffix_breakdown": suffix_breakdown,
        "projected_adapter_bytes": projected_adapter_bytes,
        "projected_submission_zip_bytes": projected_submission_zip_bytes,
        "max_submission_zip_bytes": KAGGLE_SINGLE_FILE_LIMIT_BYTES,
        "compression_ratio_estimate": NEMOTRON_ZIP_COMPRESSION_RATIO,
        "compression_overhead_bytes": NEMOTRON_ZIP_OVERHEAD_BYTES,
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
    current_known_suffixes = [suffix for suffix in current_suffixes if suffix in KNOWN_TARGET_SUFFIXES]
    if not current_known_suffixes:
        return {
            "status": "unknown",
            "reason": "could not resolve current known target module suffixes",
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

    proposed_suffixes = [
        suffix for suffix in current_known_suffixes if suffix not in CANDIDATE_WIDE_SUFFIXES
    ]
    if not proposed_suffixes:
        proposed_suffixes = list(current_known_suffixes)
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


def ensure_submission_budget_safe(
    config: dict[str, Any],
    *,
    target_modules: str | list[str] | None = None,
    rank: int | None = None,
    allow_unknown_model: bool = False,
) -> dict[str, Any]:
    budget = estimate_submission_budget(
        config,
        target_modules=target_modules,
        rank=rank,
    )
    if budget["status"] == "over_limit":
        raise ValueError(
            "Configured LoRA target_modules are not submission-safe: "
            f"{budget['reason']}. "
            "Use scripts/list_model_linear_modules.py to derive a size-safe alternative."
        )
    if budget["status"] == "unknown" and not allow_unknown_model:
        raise ValueError(
            "Submission budget cannot be estimated for this model: "
            f"{budget.get('reason', 'unknown model')}. "
            "Refusing to train without a budget guard. "
            "If this is intentional (e.g. a test harness on a non-Nemotron base), "
            "pass allow_unknown_model=True explicitly."
        )
    return budget
