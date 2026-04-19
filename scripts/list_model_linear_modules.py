from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import read_yaml, write_json
from src.student.adapter_submission_budget import (
    estimate_submission_budget,
    propose_size_safe_target_modules,
)


KNOWN_SUFFIXES: tuple[str, ...] = (
    "in_proj",
    "out_proj",
    "up_proj",
    "down_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "router",
    "gate",
)
CANDIDATE_WIDE_SUFFIXES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
)
DEFAULT_TARGET_REGEX = r".*\.(in_proj|out_proj|up_proj|down_proj)$"


def _import_optional(module_name: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Missing optional dependency '{module_name}' required for model module audit."
        ) from exc


def _nested_config(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _resolve_kagglehub_cache_path(config: dict[str, Any]) -> str | None:
    if str(config.get("model_source", "")).lower() != "kagglehub":
        return None
    handle = config.get("model_handle")
    if not handle:
        return None
    cache_root = Path.home() / ".cache" / "kagglehub" / "models" / Path(str(handle))
    if not cache_root.exists():
        return None
    version_dirs = [path for path in cache_root.iterdir() if path.is_dir()]
    if not version_dirs:
        return None
    try:
        best = max(version_dirs, key=lambda path: int(path.name))
    except ValueError:
        best = sorted(version_dirs)[-1]
    return str(best)


def _resolve_model_name_or_path(config: dict[str, Any], override: str) -> str:
    if override != "auto":
        return override
    kagglehub_path = _resolve_kagglehub_cache_path(config)
    if kagglehub_path:
        return kagglehub_path
    model_cfg = _nested_config(config, "model")
    if model_cfg.get("name"):
        return str(model_cfg["name"])
    if config.get("base_model"):
        return str(config["base_model"])
    raise ValueError("Could not resolve model_name_or_path from config.model.name or base_model.")


def _resolve_target_modules(config: dict[str, Any]) -> str | list[str]:
    model_cfg = _nested_config(config, "model")
    if model_cfg.get("target_modules") is not None:
        value = model_cfg.get("target_modules")
    else:
        value = _nested_config(config, "lora").get("target_modules", DEFAULT_TARGET_REGEX)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return str(value)


def _load_model_for_audit(
    *,
    model_name_or_path: str,
    trust_remote_code: bool,
    no_load_weights: bool,
) -> Any:
    transformers = _import_optional("transformers")
    if no_load_weights:
        auto_config = transformers.AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        try:
            return transformers.AutoModelForCausalLM.from_config(
                auto_config,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            torch = _import_optional("torch")
            return transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="meta",
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )

    torch = _import_optional("torch")
    return transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )


def _is_linear_like(module: Any, torch_module: Any | None) -> bool:
    if torch_module is not None and isinstance(module, torch_module.nn.Linear):
        return True
    if not all(hasattr(module, attr) for attr in ("weight", "in_features", "out_features")):
        return False
    class_name = type(module).__name__.lower()
    if "linear" in class_name:
        return True
    return getattr(module, "weight", None) is not None


def _module_category(
    *,
    module_name: str,
    suffix: str,
    parent: Any | None,
) -> str:
    parent_path = module_name.rsplit(".", 1)[0].lower() if "." in module_name else ""
    parent_class = type(parent).__name__.lower() if parent is not None else ""
    attention_hint = (
        "attention" in parent_class
        or "attn" in parent_class
        or "attention" in parent_path
        or "attn" in parent_path
    )

    if suffix in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return "attention"
    if suffix in {"in_proj", "out_proj"}:
        return "attention" if attention_hint else "mamba"
    if suffix in {"up_proj", "down_proj", "gate_proj"}:
        return "moe_expert" if "experts" in parent_path else "mlp"
    if suffix in {"router", "gate"}:
        return "moe_router"
    return "unknown"


def _match_target_modules(module_name: str, target_modules: str | list[str]) -> bool:
    if isinstance(target_modules, list):
        return any(module_name.endswith(token) for token in target_modules)
    return bool(re.compile(target_modules).search(module_name))


def audit_linear_modules(
    *,
    config: dict[str, Any],
    config_path: str | Path,
    model_name_or_path: str,
    no_load_weights: bool,
) -> dict[str, Any]:
    trust_remote_code = bool(config.get("trust_remote_code", True))
    target_modules = _resolve_target_modules(config)
    model = _load_model_for_audit(
        model_name_or_path=model_name_or_path,
        trust_remote_code=trust_remote_code,
        no_load_weights=no_load_weights,
    )

    try:
        torch = _import_optional("torch")
    except ImportError:
        torch = None

    module_index = dict(model.named_modules())
    suffix_counts = Counter({suffix: 0 for suffix in KNOWN_SUFFIXES})
    suffix_categories = {
        "mamba": set(),
        "attention": set(),
        "mlp": set(),
        "moe_expert": set(),
        "moe_router": set(),
        "unknown": set(),
    }
    linear_modules: list[str] = []

    for module_name, module in module_index.items():
        if not module_name:
            continue
        if not _is_linear_like(module, torch):
            continue
        linear_modules.append(module_name)
        suffix = module_name.rsplit(".", 1)[-1]
        suffix_counts[suffix] += 1
        parent_name = module_name.rsplit(".", 1)[0] if "." in module_name else ""
        category = _module_category(
            module_name=module_name,
            suffix=suffix,
            parent=module_index.get(parent_name),
        )
        suffix_categories[category].add(suffix)

    currently_matched_modules = [
        module_name
        for module_name in linear_modules
        if _match_target_modules(module_name, target_modules)
    ]
    uncovered_candidate_suffixes = [
        suffix
        for suffix in CANDIDATE_WIDE_SUFFIXES
        if suffix_counts[suffix] > 0
        and not any(
            name.endswith(f".{suffix}") or name == suffix
            for name in currently_matched_modules
        )
    ]
    uncovered_total = sum(suffix_counts[suffix] for suffix in uncovered_candidate_suffixes)
    eligible = len(uncovered_candidate_suffixes) >= 2 and uncovered_total >= 8
    uncovered_summary = ", ".join(
        f"{suffix}={suffix_counts[suffix]}" for suffix in uncovered_candidate_suffixes
    ) or "none"
    size_safe_recommendation = propose_size_safe_target_modules(
        config,
        current_target_modules=target_modules,
    )
    current_submission_budget = estimate_submission_budget(
        config,
        target_modules=target_modules,
    )

    return {
        "config_path": str(config_path),
        "model_name_or_path": model_name_or_path,
        "current_target_regex": target_modules,
        "num_linear_modules_total": len(linear_modules),
        "linear_suffix_counts": dict(sorted(suffix_counts.items())),
        "linear_suffix_category": {
            category: sorted(values)
            for category, values in suffix_categories.items()
        },
        "currently_matched_modules": currently_matched_modules,
        "num_currently_matched": len(currently_matched_modules),
        "uncovered_candidate_suffixes": uncovered_candidate_suffixes,
        "current_submission_budget": current_submission_budget,
        "wide_branch_recommendation": {
            "eligible": eligible,
            "proposed_regex": size_safe_recommendation.get("proposed_regex"),
            "proposed_target_suffixes": size_safe_recommendation.get("proposed_target_suffixes"),
            "proposed_budget": size_safe_recommendation.get("proposed_budget"),
            "budget_blocked_suffixes": size_safe_recommendation.get("budget_blocked_suffixes"),
            "candidate_evaluations": size_safe_recommendation.get("candidate_evaluations"),
            "full_wide_regex": size_safe_recommendation.get("full_wide_regex"),
            "full_wide_budget": size_safe_recommendation.get("full_wide_budget"),
            "rationale": (
                "Uncovered candidate suffix counts: "
                f"{uncovered_summary}. "
                "Eligible when at least two uncovered suffixes account for eight or more modules. "
                "The proposed regex is submission-budget-safe; suffixes that would push the "
                "projected Kaggle zip over 1 GB are listed in budget_blocked_suffixes."
            ),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Enumerate linear-like modules and audit LoRA target coverage.")
    parser.add_argument("--config", default="configs/train_stage1_format.yaml")
    parser.add_argument("--output", default="data/processed/linear_modules_audit.json")
    parser.add_argument("--model-name-or-path", default="auto")
    parser.add_argument("--no-load-weights", action="store_true")
    args = parser.parse_args(argv)

    config = read_yaml(args.config)
    payload = audit_linear_modules(
        config=config,
        config_path=args.config,
        model_name_or_path=_resolve_model_name_or_path(config, args.model_name_or_path),
        no_load_weights=bool(args.no_load_weights),
    )
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
