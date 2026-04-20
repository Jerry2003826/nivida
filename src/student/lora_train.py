from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.io import load_jsonl, read_yaml, write_json
from src.student.adapter_submission_budget import ensure_submission_budget_safe
from src.student.package_submission import (
    read_adapter_rank,
    read_adapter_target_modules,
    validate_adapter_dir,
)
from src.student.preflight import requires_mamba_ssm, run_training_preflight


def validate_lora_config(
    config: dict[str, Any],
    *,
    allow_unknown_model: bool = False,
) -> None:
    rank = int(config.get("lora", {}).get("rank", 16))
    if rank > 32:
        raise ValueError(f"LoRA rank must be <= 32, got {rank}")
    ensure_submission_budget_safe(config, allow_unknown_model=allow_unknown_model)


def _normalise_dtype(dtype_name: str) -> str:
    value = str(dtype_name).lower()
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    return aliases.get(value, value)


def _import_or_raise(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Missing optional dependency '{module_name}'. Install training extras before running non-dry training."
        ) from exc


def _maybe_add_kaggle_cutlass_path() -> str | None:
    site = _import_or_raise("site")
    cutlass_pkg_path = Path(
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"
    )
    if cutlass_pkg_path.exists():
        site.addsitedir(str(cutlass_pkg_path))
        return str(cutlass_pkg_path)
    return None


@dataclass(slots=True)
class SupervisedRecord:
    prompt: str
    completion: str
    official_family: str | None = None
    subtype: str | None = None


def load_supervised_records(path: str | Path) -> list[SupervisedRecord]:
    rows = load_jsonl(path)
    return [
        SupervisedRecord(
            prompt=str(row["prompt"]),
            completion=str(row["completion"]),
            official_family=None if row.get("official_family") is None else str(row.get("official_family")),
            subtype=None if row.get("subtype") is None else str(row.get("subtype")),
        )
        for row in rows
        if row.get("prompt") and row.get("completion")
    ]


def _build_text_sample(record: SupervisedRecord, eos_token: str = "") -> str:
    return f"{record.prompt.rstrip()}\n{record.completion.strip()}{eos_token}"


def _simple_token_count(text: str) -> int:
    """Whitespace-based length fallback used when no tokenizer is available.

    Only appropriate for dry-runs without a real tokenizer; any production
    accounting of BPE length goes through the ``tokenizer`` path.
    """
    return len(text.split())


def _measure_length(text: str, tokenizer: Any | None) -> int:
    """Return BPE length if a tokenizer is provided, else the whitespace fallback."""
    if tokenizer is None:
        return _simple_token_count(text)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _percentile(lengths: list[int], percentile: float) -> int:
    if not lengths:
        return 0
    ordered = sorted(lengths)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarise_supervised_records(
    records: list[SupervisedRecord],
    *,
    max_seq_length: int | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    prompt_lengths = [_measure_length(record.prompt, tokenizer) for record in records]
    completion_lengths = [
        _measure_length(record.completion, tokenizer) for record in records
    ]
    total_lengths = [
        _measure_length(_build_text_sample(record), tokenizer) for record in records
    ]

    family_distribution: dict[str, int] = {}
    subtype_distribution: dict[str, int] = {}
    for record in records:
        family = record.official_family or "unknown"
        family_distribution[family] = family_distribution.get(family, 0) + 1
        if record.subtype:
            key = f"{family}:{record.subtype}"
            subtype_distribution[key] = subtype_distribution.get(key, 0) + 1

    truncation_ratio = 0.0
    if max_seq_length:
        truncation_ratio = sum(length > max_seq_length for length in total_lengths) / max(1, len(total_lengths))

    return {
        "num_samples": len(records),
        "length_unit": "bpe_tokens" if tokenizer is not None else "whitespace_words",
        "family_distribution": dict(sorted(family_distribution.items())),
        "subtype_distribution": dict(sorted(subtype_distribution.items())),
        "prompt_length_p50": _percentile(prompt_lengths, 0.50),
        "prompt_length_p95": _percentile(prompt_lengths, 0.95),
        "completion_length_p50": _percentile(completion_lengths, 0.50),
        "completion_length_p95": _percentile(completion_lengths, 0.95),
        "total_length_p95": _percentile(total_lengths, 0.95),
        "truncation_ratio": truncation_ratio,
    }


def infer_recommended_max_seq_length(
    records: list[SupervisedRecord],
    *,
    floor: int = 1024,
    minimum_above_floor: int = 1536,
    cap: int = 2048,
    tokenizer: Any | None = None,
) -> int:
    if not records:
        return floor
    lengths: list[int] = []
    for record in records:
        sample = _build_text_sample(record)
        if tokenizer is None:
            lengths.append(_simple_token_count(sample))
        else:
            lengths.append(len(tokenizer(sample, add_special_tokens=False)["input_ids"]))
    p95 = _percentile(lengths, 0.95)
    if p95 <= floor:
        return floor
    return min(cap, max(minimum_above_floor, p95))


def resolve_max_seq_length(config: dict[str, Any], *, tokenizer: Any | None = None) -> int:
    training = config.get("training", {})
    value = training.get("max_seq_length", 1024)
    if value != "auto":
        return int(value)
    dataset_path = training.get("dataset_path")
    if not dataset_path or not Path(dataset_path).exists():
        return int(training.get("auto_floor_seq_length", 1024))
    records = load_supervised_records(dataset_path)
    return infer_recommended_max_seq_length(
        records,
        floor=int(training.get("auto_floor_seq_length", 1024)),
        minimum_above_floor=int(training.get("auto_min_seq_length_above_floor", 1536)),
        cap=int(training.get("auto_max_seq_length_cap", 2048)),
        tokenizer=tokenizer,
    )


def _load_metrics_tokenizer(config: dict[str, Any]) -> Any | None:
    """Resolve an optional tokenizer for dataset-stats accounting.

    Looks up ``tokenizer_path`` at the top level of the config. The tokenizer
    is loaded via transformers.AutoTokenizer and never downloads weights, so
    it is safe for dry-run contexts. If the path is missing or transformers is
    not installed, returns ``None`` and the caller falls back to whitespace
    length.
    """
    tokenizer_path = config.get("tokenizer_path")
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        return AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    except Exception:
        return None


def dry_run_manifest(config: dict[str, Any]) -> dict[str, Any]:
    validate_lora_config(config)
    preflight = run_training_preflight(config, dry_run=True)
    training = config.get("training", {})
    metrics_tokenizer = _load_metrics_tokenizer(config)
    resolved_max_seq_length = resolve_max_seq_length(config, tokenizer=metrics_tokenizer)
    dataset_stats = None
    dataset_path = training.get("dataset_path")
    if dataset_path and Path(dataset_path).exists():
        dataset_stats = summarise_supervised_records(
            load_supervised_records(dataset_path),
            max_seq_length=resolved_max_seq_length,
            tokenizer=metrics_tokenizer,
        )
    return {
        "base_model": config.get("base_model"),
        "model_source": config.get("model_source", "huggingface"),
        "model_handle": config.get("model_handle"),
        "lora": config.get("lora", {}),
        "environment": {
            str(key): str(value)
            for key, value in dict(config.get("environment", {})).items()
            if value not in (None, "")
        },
        "training": training,
        "tokenizer_path": config.get("tokenizer_path"),
        "dataset_stats_tokenizer": (
            type(metrics_tokenizer).__name__ if metrics_tokenizer is not None else None
        ),
        "preflight": preflight,
        "submission_budget": ensure_submission_budget_safe(config),
        "resolved_max_seq_length": resolved_max_seq_length,
        "resolved_target_modules": normalise_target_modules(config.get("lora", {}).get("target_modules")),
        "dataset_stats": dataset_stats,
        "status": "dry_run_ok",
        "notes": "Training loop wired for Kaggle demo compatibility; dry-run skips model loading and optimization.",
    }


def _load_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    normalised = _normalise_dtype(dtype_name)
    if not hasattr(torch_module, normalised):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch_module, normalised)


def resolve_model_path(config: dict[str, Any]) -> str:
    source = str(config.get("model_source", "huggingface")).lower()
    if source == "kagglehub":
        kagglehub = _import_or_raise("kagglehub")
        handle = config.get("model_handle")
        if not handle:
            raise ValueError("model_handle is required when model_source=kagglehub")
        return str(kagglehub.model_download(handle))
    base_model = config.get("base_model")
    if not base_model:
        raise ValueError("base_model is required")
    return str(base_model)


def build_training_arguments_kwargs(
    transformers_module: Any,
    config: dict[str, Any],
    *,
    output_dir: str,
    has_eval_dataset: bool,
) -> dict[str, Any]:
    training_config = config.get("training", {})
    kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": float(training_config.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(training_config.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(training_config.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(training_config.get("learning_rate", 1e-4)),
        "warmup_ratio": float(training_config.get("warmup_ratio", 0.0)),
        "lr_scheduler_type": str(training_config.get("lr_scheduler_type", "linear")),
        "max_grad_norm": float(training_config.get("max_grad_norm", 1.0)),
        "logging_steps": int(training_config.get("logging_steps", 10)),
        "save_steps": int(training_config.get("save_steps", 100)),
        "eval_steps": int(training_config.get("eval_steps", 100)),
        "save_total_limit": int(training_config.get("save_total_limit", 2)),
        "bf16": _normalise_dtype(config.get("torch_dtype", "bfloat16")) == "bfloat16",
        "fp16": _normalise_dtype(config.get("torch_dtype", "bfloat16")) == "float16",
        "report_to": list(training_config.get("report_to", [])),
        "remove_unused_columns": False,
        "dataloader_pin_memory": bool(training_config.get("dataloader_pin_memory", False)),
        "gradient_checkpointing": bool(training_config.get("gradient_checkpointing", False)),
        "seed": int(config.get("seed", 42)),
    }
    evaluation_value = "steps" if has_eval_dataset else "no"
    training_args_params = inspect.signature(transformers_module.TrainingArguments.__init__).parameters
    if "evaluation_strategy" in training_args_params:
        kwargs["evaluation_strategy"] = evaluation_value
    elif "eval_strategy" in training_args_params:
        kwargs["eval_strategy"] = evaluation_value
    return kwargs


def normalise_target_modules(raw_target_modules: Any) -> str | list[str]:
    if isinstance(raw_target_modules, (list, tuple)):
        return [str(item) for item in raw_target_modules]
    if raw_target_modules is None:
        return r".*\.(in_proj|out_proj|up_proj|down_proj)$"
    return str(raw_target_modules)


def resolve_target_module_matches(model: Any, target_modules: str | list[str]) -> list[str]:
    module_names = [name for name, _ in model.named_modules() if name]
    if isinstance(target_modules, list):
        return [name for name in module_names if any(name.endswith(token) for token in target_modules)]
    pattern = re.compile(target_modules)
    return [name for name in module_names if pattern.search(name)]


def apply_runtime_environment(config: dict[str, Any]) -> dict[str, str]:
    applied: dict[str, str] = {}
    for key, value in dict(config.get("environment", {})).items():
        if value in (None, ""):
            continue
        env_key = str(key)
        env_value = str(value)
        os.environ[env_key] = env_value
        applied[env_key] = env_value
    return applied


def ensure_generation_output_aliases(transformers_module: Any) -> dict[str, str]:
    generation_module = getattr(transformers_module, "generation", None)
    if generation_module is None:
        generation_module = importlib.import_module("transformers.generation")
    fallback = getattr(generation_module, "GenerateDecoderOnlyOutput", None)
    if fallback is None:
        return {}
    aliases: dict[str, str] = {}
    for alias in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
        if hasattr(generation_module, alias):
            continue
        setattr(generation_module, alias, fallback)
        aliases[alias] = "GenerateDecoderOnlyOutput"
    return aliases


def _default_use_fast_tokenizer(config: dict[str, Any]) -> bool:
    if requires_mamba_ssm(config):
        return False
    tokenizer_config = dict(config.get("tokenizer", {}))
    if "use_fast" in tokenizer_config:
        return bool(tokenizer_config["use_fast"])
    return True


def load_tokenizer(transformers_module: Any, model_path: str, config: dict[str, Any]) -> Any:
    trust_remote_code = bool(config.get("trust_remote_code", True))
    tokenizer_config = dict(config.get("tokenizer", {}))
    use_fast = bool(tokenizer_config.get("use_fast", _default_use_fast_tokenizer(config)))
    kwargs = {
        "trust_remote_code": trust_remote_code,
        "use_fast": use_fast,
    }
    try:
        tokenizer = transformers_module.AutoTokenizer.from_pretrained(model_path, **kwargs)
    except Exception:
        if not use_fast:
            raise
        fallback_kwargs = {
            "trust_remote_code": trust_remote_code,
            "use_fast": False,
        }
        tokenizer = transformers_module.AutoTokenizer.from_pretrained(model_path, **fallback_kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def configure_model_for_training(model: Any, config: dict[str, Any]) -> Any:
    training_config = dict(config.get("training", {}))
    if bool(training_config.get("gradient_checkpointing", False)):
        model_config = getattr(model, "config", None)
        if model_config is not None and hasattr(model_config, "use_cache"):
            model_config.use_cache = False
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "use_cache"):
            generation_config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    return model


def _iter_lora_named_parameters(model: Any):
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            yield name, parameter


def ensure_lora_parameters_trainable(model: Any) -> dict[str, Any]:
    total_tensors = 0
    total_numel = 0
    trainable_tensors = 0
    trainable_numel = 0
    reenabled_names: list[str] = []

    for name, parameter in _iter_lora_named_parameters(model):
        total_tensors += 1
        total_numel += int(parameter.numel())
        if not parameter.requires_grad:
            parameter.requires_grad = True
            reenabled_names.append(name)
        if parameter.requires_grad:
            trainable_tensors += 1
            trainable_numel += int(parameter.numel())

    if total_tensors == 0:
        raise ValueError("No LoRA parameters were found after adapter initialisation.")

    return {
        "lora_parameter_tensors": total_tensors,
        "lora_parameter_numel": total_numel,
        "trainable_lora_parameter_tensors": trainable_tensors,
        "trainable_lora_parameter_numel": trainable_numel,
        "reenabled_lora_parameter_tensors": len(reenabled_names),
        "reenabled_lora_parameter_names": reenabled_names,
    }


def summarise_lora_gradients(model: Any) -> dict[str, Any]:
    total_tensors = 0
    trainable_tensors = 0
    tensors_with_grad = 0
    grad_norm_sum = 0.0
    grad_norm_max = 0.0
    nonfinite_grad_tensors = 0
    nonfinite_grad_elements = 0
    try:
        import torch
    except Exception:
        torch = None

    for _name, parameter in _iter_lora_named_parameters(model):
        total_tensors += 1
        if parameter.requires_grad:
            trainable_tensors += 1
        if parameter.grad is None:
            continue
        tensors_with_grad += 1
        gradient = parameter.grad.detach()
        if hasattr(gradient, "float"):
            gradient = gradient.float()
        if torch is not None:
            try:
                finite_mask = torch.isfinite(gradient)
            except Exception:
                finite_mask = None
        else:
            finite_mask = None
        if finite_mask is not None:
            current_nonfinite = int((~finite_mask).sum().item())
            if current_nonfinite:
                nonfinite_grad_tensors += 1
                nonfinite_grad_elements += current_nonfinite
                gradient = gradient.clone()
                gradient.masked_fill_(~finite_mask, 0)
        grad_norm = float(gradient.norm().item())
        grad_norm_sum += grad_norm
        grad_norm_max = max(grad_norm_max, grad_norm)

    mean_grad_norm = None
    if tensors_with_grad:
        mean_grad_norm = grad_norm_sum / tensors_with_grad

    return {
        "lora_parameter_tensors": total_tensors,
        "trainable_lora_parameter_tensors": trainable_tensors,
        "lora_tensors_with_grad": tensors_with_grad,
        "mean_grad_norm": mean_grad_norm,
        "max_grad_norm": grad_norm_max if tensors_with_grad else None,
        "nonfinite_grad_tensors": nonfinite_grad_tensors,
        "nonfinite_grad_elements": nonfinite_grad_elements,
    }


def summarise_lora_runtime_state(model: Any) -> dict[str, Any]:
    lora_modules = 0
    disabled_modules = 0
    merged_modules = 0
    active_adapters: set[str] = set()

    for _name, module in model.named_modules():
        if not (hasattr(module, "lora_A") or hasattr(module, "lora_B")):
            continue
        lora_modules += 1
        if bool(getattr(module, "disable_adapters", False)):
            disabled_modules += 1
        if bool(getattr(module, "merged", False)):
            merged_modules += 1

        current_active = getattr(module, "active_adapters", None)
        if current_active is None:
            current_active = getattr(module, "active_adapter", None)
        if current_active is None:
            continue
        if isinstance(current_active, str):
            active_adapters.add(current_active)
            continue
        if isinstance(current_active, (list, tuple, set)):
            active_adapters.update(str(item) for item in current_active)
            continue
        active_adapters.add(str(current_active))

    return {
        "lora_module_count": lora_modules,
        "disabled_lora_module_count": disabled_modules,
        "merged_lora_module_count": merged_modules,
        "active_adapters": sorted(active_adapters),
    }


def sanitize_nonfinite_lora_gradients(model: Any) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to sanitise non-finite LoRA gradients during training."
        ) from exc

    tensors_with_grad = 0
    nonfinite_grad_tensors = 0
    nonfinite_grad_elements = 0
    sanitized_parameter_names: list[str] = []

    for name, parameter in _iter_lora_named_parameters(model):
        if parameter.grad is None:
            continue
        tensors_with_grad += 1
        finite_mask = torch.isfinite(parameter.grad)
        if bool(finite_mask.all()):
            continue
        nonfinite_grad_tensors += 1
        current_nonfinite = int((~finite_mask).sum().item())
        nonfinite_grad_elements += current_nonfinite
        parameter.grad.masked_fill_(~finite_mask, 0)
        sanitized_parameter_names.append(name)

    return {
        "lora_tensors_with_grad": tensors_with_grad,
        "nonfinite_grad_tensors": nonfinite_grad_tensors,
        "nonfinite_grad_elements": nonfinite_grad_elements,
        "sanitized_parameter_names": sanitized_parameter_names,
    }


def _to_float_scalar(value: Any) -> float | None:
    if value is None:
        return None
    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "float"):
        candidate = candidate.float()
    if hasattr(candidate, "item"):
        try:
            return float(candidate.item())
        except (TypeError, ValueError):
            return None
    try:
        return float(candidate)
    except (TypeError, ValueError):
        return None


def summarise_model_losses(outputs: Any) -> dict[str, float]:
    summary: dict[str, float] = {}
    keys = (
        "loss",
        "aux_loss",
        "router_aux_loss",
        "router_z_loss",
        "moe_aux_loss",
        "z_loss",
    )
    for key in keys:
        if isinstance(outputs, dict):
            value = outputs.get(key)
        else:
            value = getattr(outputs, key, None)
        scalar = _to_float_scalar(value)
        if scalar is not None:
            summary[key] = scalar
    return summary


def should_capture_loss_outputs(
    *,
    global_step: int,
    logging_steps: int,
    return_outputs: bool,
) -> bool:
    if return_outputs:
        return True
    logging_every = max(1, int(logging_steps))
    candidate_step = max(1, int(global_step) + 1)
    return candidate_step % logging_every == 0


def should_require_strict_divergence_check(
    *,
    global_step: int,
    warmup_steps: int,
    floor_step: int = 250,
    warmup_multiplier: float = 2.0,
) -> bool:
    thresholds: list[int] = []
    scaled_warmup = math.ceil(max(0, int(warmup_steps)) * max(0.0, float(warmup_multiplier)))
    if scaled_warmup > 0:
        thresholds.append(scaled_warmup)
    if int(floor_step) > 0:
        thresholds.append(int(floor_step))
    if not thresholds:
        return True
    return int(global_step) >= min(thresholds)


def _sample_keys_evenly(keys: list[str], limit: int) -> list[str]:
    if len(keys) <= limit:
        return list(keys)
    if limit <= 1:
        return [keys[0]]

    last_index = len(keys) - 1
    indexes = {
        min(last_index, round(position * last_index / (limit - 1)))
        for position in range(limit)
    }
    return [keys[index] for index in sorted(indexes)]


def probe_saved_lora_artifact(
    adapter_model_path: str | Path,
    *,
    max_tensors: int = 128,
) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "safetensors is required to inspect saved LoRA adapters during training diagnostics."
        ) from exc
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is required to inspect saved LoRA adapters during training diagnostics."
        ) from exc

    path = Path(adapter_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Saved adapter artifact is missing: {path}")

    probe_framework = "np"
    try:
        handle_cm = safe_open(str(path), framework="pt")
        probe_framework = "pt"
    except (ImportError, ModuleNotFoundError):
        handle_cm = safe_open(str(path), framework="np")

    with handle_cm as handle:
        lora_keys = sorted(
            key
            for key in handle.keys()
            if key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight")
        )
        lora_b_keys = [key for key in lora_keys if key.endswith(".lora_B.weight")]
        if not lora_keys:
            raise ValueError(f"No LoRA tensors were found in saved adapter artifact: {path}")
        sample_limit = max(1, int(max_tensors))
        sampled_keys = _sample_keys_evenly(lora_keys, sample_limit)
        sampled_b_keys = _sample_keys_evenly(lora_b_keys, sample_limit)

        digest = hashlib.sha256()

        for key in sampled_keys:
            try:
                raw_tensor = handle.get_tensor(key)
            except Exception as exc:
                raise RuntimeError(
                    f"[LoRA] cannot read saved tensor '{key}' from {path}. "
                    "The adapter storage dtype may not be supported by the current probe backend."
                ) from exc
            if probe_framework == "pt":
                tensor = raw_tensor.detach().float().cpu().contiguous()
                tensor_bytes = tensor.numpy().tobytes()
            else:
                try:
                    tensor = np.asarray(raw_tensor, dtype=np.float32)
                except Exception as exc:
                    raise RuntimeError(
                        f"[LoRA] cannot normalise saved tensor '{key}' from {path} with numpy. "
                        "If the adapter was saved in bf16/bfloat16 storage, rerun this probe in an environment "
                        "with torch so safetensors can load it via framework='pt'."
                    ) from exc
                tensor_bytes = tensor.tobytes()
            digest.update(key.encode("utf-8"))
            digest.update(str(tuple(int(dim) for dim in tensor.shape)).encode("ascii"))
            digest.update(tensor_bytes)

        sampled_b_total = 0
        sampled_b_zero_count = 0
        sampled_b_max_abs = 0.0
        sampled_b_tensors = 0
        for key in sampled_b_keys:
            try:
                raw_tensor = handle.get_tensor(key)
            except Exception as exc:
                raise RuntimeError(
                    f"[LoRA] cannot read saved tensor '{key}' from {path}. "
                    "The adapter storage dtype may not be supported by the current probe backend."
                ) from exc
            if probe_framework == "pt":
                tensor = raw_tensor.detach().float().cpu().contiguous()
                tensor_size = int(tensor.numel())
                zero_count = int((tensor == 0).sum().item())
                tensor_max_abs = float(tensor.abs().max().item()) if tensor_size else 0.0
            else:
                try:
                    tensor = np.asarray(raw_tensor, dtype=np.float32)
                except Exception as exc:
                    raise RuntimeError(
                        f"[LoRA] cannot normalise saved tensor '{key}' from {path} with numpy. "
                        "If the adapter was saved in bf16/bfloat16 storage, rerun this probe in an environment "
                        "with torch so safetensors can load it via framework='pt'."
                    ) from exc
                tensor_size = int(tensor.size)
                zero_count = int((tensor == 0).sum())
                tensor_max_abs = float(np.abs(tensor).max()) if tensor_size else 0.0
            sampled_b_tensors += 1
            sampled_b_total += tensor_size
            sampled_b_zero_count += zero_count
            sampled_b_max_abs = max(sampled_b_max_abs, tensor_max_abs)

    sampled_b_zero_fraction = None
    if sampled_b_total:
        sampled_b_zero_fraction = sampled_b_zero_count / sampled_b_total

    return {
        "adapter_model_path": str(path),
        "sampled_tensor_count": len(sampled_keys),
        "sampled_tensor_names": sampled_keys,
        "sampled_tensor_digest": digest.hexdigest(),
        "lora_b_tensor_count": len(lora_b_keys),
        "sampled_lora_b_tensor_count": sampled_b_tensors,
        "sampled_lora_b_zero_fraction": sampled_b_zero_fraction,
        "sampled_lora_b_max_abs": sampled_b_max_abs if sampled_b_tensors else None,
    }


def assert_saved_lora_artifact_healthy(
    adapter_model_path: str | Path,
    *,
    step_label: str,
    initial_probe: dict[str, Any] | None = None,
    max_tensors: int = 128,
    require_divergence_from_initial: bool = True,
) -> dict[str, Any]:
    probe = probe_saved_lora_artifact(adapter_model_path, max_tensors=max_tensors)

    if probe["sampled_lora_b_tensor_count"] <= 0:
        raise RuntimeError(
            f"[LoRA] {step_label}: sampled checkpoint did not contain any lora_B tensors at "
            f"{probe['adapter_model_path']}"
        )

    zero_fraction = probe["sampled_lora_b_zero_fraction"]
    max_abs = probe["sampled_lora_b_max_abs"]
    if zero_fraction == 1.0 or max_abs == 0.0:
        raise RuntimeError(
            f"[LoRA] {step_label}: sampled lora_B tensors are still all-zero after save "
            f"(zero_fraction={zero_fraction}, max_abs={max_abs})."
        )

    if (
        require_divergence_from_initial
        and initial_probe
        and probe["sampled_tensor_digest"] == initial_probe["sampled_tensor_digest"]
    ):
        raise RuntimeError(
            f"[LoRA] {step_label}: sampled saved adapter tensors are byte-identical to the init adapter "
            f"({probe['adapter_model_path']})."
        )

    return probe


class LoraTrainingHealth:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        initial_probe: dict[str, Any] | None,
        probe_tensor_limit: int,
        strict_divergence_after_step: int,
        strict_divergence_after_warmup_multiplier: float,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.initial_probe = initial_probe
        self.probe_tensor_limit = max(1, int(probe_tensor_limit))
        self.strict_divergence_after_step = int(strict_divergence_after_step)
        self.strict_divergence_after_warmup_multiplier = float(strict_divergence_after_warmup_multiplier)
        self.diagnostic_log_path = self.output_dir / "training_diagnostics.log"
        self.last_logged_step = -1
        self.last_loss_logged_step = -1
        self.last_grad_summary: dict[str, Any] | None = None
        self.last_loss_summary: dict[str, Any] | None = None
        self.last_eval_metrics: dict[str, Any] | None = None
        self.last_eval_runtime_state: dict[str, Any] | None = None
        self.last_saved_probe: dict[str, Any] | None = None
        self.last_sanitized_step = -1
        self.first_nonfinite_grad_step: int | None = None

    def _emit_diagnostic(self, message: str) -> None:
        print(message, flush=True)
        with self.diagnostic_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")

    def log_trainable_summary(self, summary: dict[str, Any]) -> None:
        self._emit_diagnostic(
            "[LoRA] trainable tensors="
            f"{summary['trainable_lora_parameter_tensors']}/{summary['lora_parameter_tensors']} "
            f"numel={summary['trainable_lora_parameter_numel']}/{summary['lora_parameter_numel']} "
            f"reenabled={summary['reenabled_lora_parameter_tensors']}"
        )

    def maybe_log_gradients(self, *, model: Any, global_step: int, logging_steps: int) -> None:
        logging_every = max(1, int(logging_steps))
        candidate_step = max(1, int(global_step) + 1)
        if candidate_step == self.last_logged_step or candidate_step % logging_every != 0:
            return

        summary = summarise_lora_gradients(model)
        self.last_grad_summary = summary
        self.last_logged_step = candidate_step
        self._emit_diagnostic(
            "[LoRA] step="
            f"{candidate_step} grad_norm mean={summary['mean_grad_norm']} "
            f"max={summary['max_grad_norm']} tensors_with_grad="
            f"{summary['lora_tensors_with_grad']}/{summary['trainable_lora_parameter_tensors']} "
            f"nonfinite_tensors={summary['nonfinite_grad_tensors']} "
            f"nonfinite_elements={summary['nonfinite_grad_elements']}"
        )

    def maybe_log_losses(self, *, outputs: Any, global_step: int, logging_steps: int) -> None:
        logging_every = max(1, int(logging_steps))
        candidate_step = max(1, int(global_step) + 1)
        if candidate_step == self.last_loss_logged_step or candidate_step % logging_every != 0:
            return

        summary = summarise_model_losses(outputs)
        self.last_loss_logged_step = candidate_step
        self.last_loss_summary = summary
        if summary:
            formatted = " ".join(f"{key}={value}" for key, value in summary.items())
            self._emit_diagnostic(f"[LossDiag] step={candidate_step} {formatted}")
        else:
            self._emit_diagnostic(f"[LossDiag] step={candidate_step} no_named_loss_components_found")

    def log_eval_diagnostics(self, *, model: Any, global_step: int, metrics: dict[str, Any]) -> None:
        runtime_state = summarise_lora_runtime_state(model)
        param_state = summarise_lora_gradients(model)
        self.last_eval_runtime_state = runtime_state
        self.last_eval_metrics = dict(metrics)
        self._emit_diagnostic(
            "[EvalDiag] step="
            f"{int(global_step)} eval_loss={metrics.get('eval_loss')!r} "
            f"trainable_tensors={param_state['trainable_lora_parameter_tensors']}/"
            f"{param_state['lora_parameter_tensors']} disabled_modules="
            f"{runtime_state['disabled_lora_module_count']}/{runtime_state['lora_module_count']} "
            f"merged_modules={runtime_state['merged_lora_module_count']} "
            f"active_adapters={runtime_state['active_adapters']} "
            f"nonfinite_tensors={param_state['nonfinite_grad_tensors']} "
            f"nonfinite_elements={param_state['nonfinite_grad_elements']}"
        )

    def verify_saved_artifact(
        self,
        output_dir: str | Path,
        *,
        require_divergence_from_initial: bool,
    ) -> dict[str, Any]:
        adapter_model_path = Path(output_dir) / "adapter_model.safetensors"
        probe = assert_saved_lora_artifact_healthy(
            adapter_model_path,
            step_label=Path(output_dir).name,
            initial_probe=self.initial_probe,
            max_tensors=self.probe_tensor_limit,
            require_divergence_from_initial=require_divergence_from_initial,
        )
        self.last_saved_probe = probe
        self._emit_diagnostic(
            "[LoRA] saved "
            f"{Path(output_dir).name} sampled_lora_b_zero_fraction={probe['sampled_lora_b_zero_fraction']} "
            f"sampled_lora_b_max_abs={probe['sampled_lora_b_max_abs']}"
        )
        return probe

    def should_require_strict_divergence(self, *, global_step: int, warmup_steps: int) -> bool:
        return should_require_strict_divergence_check(
            global_step=global_step,
            warmup_steps=warmup_steps,
            floor_step=self.strict_divergence_after_step,
            warmup_multiplier=self.strict_divergence_after_warmup_multiplier,
        )

    def maybe_sanitize_gradients(self, *, model: Any, global_step: int, logging_steps: int) -> dict[str, Any]:
        logging_every = max(1, int(logging_steps))
        candidate_step = max(1, int(global_step) + 1)
        summary = sanitize_nonfinite_lora_gradients(model)
        if summary["nonfinite_grad_tensors"] <= 0:
            return summary

        if self.first_nonfinite_grad_step is None:
            self.first_nonfinite_grad_step = candidate_step

        should_log = (
            candidate_step % logging_every == 0
            or self.last_sanitized_step != candidate_step
            and self.first_nonfinite_grad_step == candidate_step
        )
        if should_log:
            self.last_sanitized_step = candidate_step
            self._emit_diagnostic(
                "[LoRA] step="
                f"{candidate_step} sanitized_nonfinite_gradients tensors="
                f"{summary['nonfinite_grad_tensors']} elements={summary['nonfinite_grad_elements']}"
            )
        return summary


def _load_trainable_adapter(peft_module: Any, model: Any, adapter_dir: str | Path) -> Any:
    try:
        return peft_module.PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    except TypeError:
        return peft_module.PeftModel.from_pretrained(model, str(adapter_dir))


def validate_init_adapter_compatibility(
    config: dict[str, Any],
    init_adapter_dir: str | Path,
) -> dict[str, Any]:
    validate_adapter_dir(init_adapter_dir)

    config_rank = int(config.get("lora", {}).get("rank", 16))
    init_rank = read_adapter_rank(init_adapter_dir)
    if init_rank is not None and init_rank != config_rank:
        raise ValueError(
            f"Init adapter rank {init_rank} != config rank {config_rank}. "
            "Keep ranks identical across staged fine-tuning."
        )

    config_target_modules = normalise_target_modules(config.get("lora", {}).get("target_modules"))
    init_target_modules = normalise_target_modules(read_adapter_target_modules(init_adapter_dir))
    if config_target_modules != init_target_modules:
        raise ValueError(
            "Init adapter target_modules do not match the current config after normalisation. "
            f"config={config_target_modules!r} init={init_target_modules!r}"
        )

    return {
        "init_adapter_dir": str(init_adapter_dir),
        "init_adapter_rank": init_rank,
        "init_adapter_target_modules": init_target_modules,
    }


def initialise_lora_model(
    model: Any,
    peft_module: Any,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    target_modules = normalise_target_modules(config.get("lora", {}).get("target_modules"))
    matched_target_modules = resolve_target_module_matches(model, target_modules)
    if not matched_target_modules:
        raise ValueError(
            f"No modules matched target_modules={target_modules!r}. "
            "Run `python scripts/inspect_target_modules.py --config ...` first."
        )

    training_config = dict(config.get("training", {}))
    init_adapter_dir = training_config.get("init_adapter_dir")
    init_metadata = {
        "init_adapter_dir": None,
        "init_adapter_rank": None,
        "init_adapter_target_modules": None,
    }
    if init_adapter_dir:
        init_metadata = validate_init_adapter_compatibility(config, init_adapter_dir)
        model = _load_trainable_adapter(peft_module, model, init_adapter_dir)
    else:
        lora_config = peft_module.LoraConfig(
            r=int(config["lora"]["rank"]),
            lora_alpha=int(config["lora"].get("alpha", 16)),
            target_modules=target_modules,
            lora_dropout=float(config["lora"].get("dropout", 0.05)),
            bias=str(config["lora"].get("bias", "none")),
            task_type=peft_module.TaskType.CAUSAL_LM,
        )
        model = peft_module.get_peft_model(model, lora_config)

    return model, {
        "target_modules": target_modules,
        "matched_target_modules": matched_target_modules,
        "num_matched_target_modules": len(matched_target_modules),
        **init_metadata,
    }


def train_lora(config: dict[str, Any]) -> dict[str, Any]:
    validate_lora_config(config)
    applied_environment = apply_runtime_environment(config)
    preflight = run_training_preflight(config, dry_run=False)

    _maybe_add_kaggle_cutlass_path()
    torch = _import_or_raise("torch")
    transformers = _import_or_raise("transformers")
    generation_aliases = ensure_generation_output_aliases(transformers)
    if requires_mamba_ssm(config):
        _import_or_raise("mamba_ssm")
    peft = _import_or_raise("peft")

    records = load_supervised_records(config["training"]["dataset_path"])
    if not records:
        raise ValueError("Training dataset is empty.")

    eval_path = config.get("training", {}).get("eval_path")
    eval_records = load_supervised_records(eval_path) if eval_path and Path(eval_path).exists() else []
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(config)
    dtype = _load_torch_dtype(torch, config.get("torch_dtype", "bfloat16"))
    tokenizer = load_tokenizer(transformers, model_path, config)
    resolved_max_seq_length = resolve_max_seq_length(config, tokenizer=tokenizer)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=config.get("device_map", "auto"),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        torch_dtype=dtype,
    )
    model = configure_model_for_training(model, config)
    model, lora_initialisation = initialise_lora_model(model, peft, config)
    lora_trainable_summary = ensure_lora_parameters_trainable(model)

    training_config = dict(config.get("training", {}))
    probe_tensor_limit = int(training_config.get("lora_probe_tensor_limit", 128))
    strict_divergence_after_step = int(training_config.get("strict_divergence_after_step", 250))
    strict_divergence_after_warmup_multiplier = float(
        training_config.get("strict_divergence_after_warmup_multiplier", 2.0)
    )
    initial_probe = None
    init_adapter_dir = lora_initialisation.get("init_adapter_dir")
    if init_adapter_dir:
        init_adapter_model_path = Path(init_adapter_dir) / "adapter_model.safetensors"
        if init_adapter_model_path.exists():
            initial_probe = probe_saved_lora_artifact(
                init_adapter_model_path,
                max_tensors=probe_tensor_limit,
            )
    lora_health = LoraTrainingHealth(
        output_dir=output_dir,
        initial_probe=initial_probe,
        probe_tensor_limit=probe_tensor_limit,
        strict_divergence_after_step=strict_divergence_after_step,
        strict_divergence_after_warmup_multiplier=strict_divergence_after_warmup_multiplier,
    )
    lora_health.log_trainable_summary(lora_trainable_summary)

    class PromptCompletionDataset(torch.utils.data.Dataset):
        def __init__(self, items: list[SupervisedRecord]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> dict[str, list[int]]:
            item = self.items[idx]
            prompt_ids = tokenizer(
                item.prompt.rstrip() + "\n",
                add_special_tokens=False,
                truncation=True,
                max_length=resolved_max_seq_length,
            )["input_ids"]
            full_ids = tokenizer(
                _build_text_sample(item, tokenizer.eos_token or ""),
                add_special_tokens=False,
                truncation=True,
                max_length=resolved_max_seq_length,
            )["input_ids"]
            labels = list(full_ids)
            prompt_len = min(len(prompt_ids), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
            attention_mask = [1] * len(full_ids)
            return {"input_ids": full_ids, "labels": labels, "attention_mask": attention_mask}

    train_dataset = PromptCompletionDataset(records)
    eval_dataset = PromptCompletionDataset(eval_records) if eval_records else None

    class SupervisedCollator:
        def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
            max_length = max(len(feature["input_ids"]) for feature in features)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            batch = {"input_ids": [], "labels": [], "attention_mask": []}
            for feature in features:
                pad_len = max_length - len(feature["input_ids"])
                batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
                batch["labels"].append(feature["labels"] + [-100] * pad_len)
                batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            return {key: torch.tensor(value) for key, value in batch.items()}

    training_args = transformers.TrainingArguments(
        **build_training_arguments_kwargs(
            transformers,
            config,
            output_dir=str(output_dir),
            has_eval_dataset=eval_dataset is not None,
        )
    )

    class DiagnosticTrainer(transformers.Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            *args: Any,
            **kwargs: Any,
        ):
            capture_outputs = should_capture_loss_outputs(
                global_step=int(getattr(self.state, "global_step", 0)),
                logging_steps=int(getattr(self.args, "logging_steps", 10)),
                return_outputs=return_outputs,
            )
            if capture_outputs:
                loss, outputs = super().compute_loss(
                    model,
                    inputs,
                    return_outputs=True,
                    *args,
                    **kwargs,
                )
                if getattr(model, "training", False):
                    lora_health.maybe_log_losses(
                        outputs=outputs,
                        global_step=int(getattr(self.state, "global_step", 0)),
                        logging_steps=int(getattr(self.args, "logging_steps", 10)),
                    )
                if return_outputs:
                    return loss, outputs
                return loss
            return super().compute_loss(
                model,
                inputs,
                return_outputs=False,
                *args,
                **kwargs,
            )

        def training_step(self, model: Any, inputs: dict[str, Any], *args: Any, **kwargs: Any):
            loss = super().training_step(model, inputs, *args, **kwargs)
            lora_health.maybe_log_gradients(
                model=model,
                global_step=int(getattr(self.state, "global_step", 0)),
                logging_steps=int(getattr(self.args, "logging_steps", 10)),
            )
            return loss

        def evaluate(self, *args: Any, **kwargs: Any):
            metrics = super().evaluate(*args, **kwargs)
            lora_health.log_eval_diagnostics(
                model=self.model,
                global_step=int(getattr(self.state, "global_step", 0)),
                metrics=metrics,
            )
            return metrics

        def _clip_grad_norm(self, model: Any):
            lora_health.maybe_sanitize_gradients(
                model=model,
                global_step=int(getattr(self.state, "global_step", 0)),
                logging_steps=int(getattr(self.args, "logging_steps", 10)),
            )
            return super()._clip_grad_norm(model)

        def _save(self, output_dir: str | None = None, state_dict: Any = None) -> None:
            super()._save(output_dir=output_dir, state_dict=state_dict)
            if hasattr(self.args, "get_warmup_steps"):
                warmup_steps = int(self.args.get_warmup_steps(int(getattr(self.state, "max_steps", 0))))
            else:
                warmup_steps = int(getattr(self.args, "warmup_steps", 0))
            lora_health.verify_saved_artifact(
                output_dir or self.args.output_dir,
                require_divergence_from_initial=lora_health.should_require_strict_divergence(
                    global_step=int(getattr(self.state, "global_step", 0)),
                    warmup_steps=warmup_steps,
                ),
            )

    trainer = DiagnosticTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedCollator(),
    )
    trainer.train(resume_from_checkpoint=config["training"].get("resume_from_checkpoint"))

    model.save_pretrained(str(output_dir))
    lora_health.verify_saved_artifact(
        output_dir,
        require_divergence_from_initial=True,
    )
    adapter_files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    metadata = {
        "base_model": config.get("base_model"),
        "resolved_model_path": model_path,
        "model_source": config.get("model_source", "huggingface"),
        "model_handle": config.get("model_handle"),
        "lora": config.get("lora", {}),
        "environment": applied_environment,
        "generation_aliases": generation_aliases,
        "num_train_records": len(records),
        "num_eval_records": len(eval_records),
        "preflight": preflight,
        "adapter_files": adapter_files,
        "training_diagnostics_log": str(lora_health.diagnostic_log_path),
        "target_modules": lora_initialisation["target_modules"],
        "matched_target_modules": lora_initialisation["matched_target_modules"],
        "num_matched_target_modules": lora_initialisation["num_matched_target_modules"],
        "init_adapter_dir": lora_initialisation["init_adapter_dir"],
        "init_adapter_rank": lora_initialisation["init_adapter_rank"],
        "init_adapter_target_modules": lora_initialisation["init_adapter_target_modules"],
        "lora_trainable_summary": lora_trainable_summary,
        "initial_saved_probe": initial_probe,
        "strict_divergence_after_step": strict_divergence_after_step,
        "strict_divergence_after_warmup_multiplier": strict_divergence_after_warmup_multiplier,
        "last_grad_summary": lora_health.last_grad_summary,
        "last_loss_summary": lora_health.last_loss_summary,
        "last_eval_metrics": lora_health.last_eval_metrics,
        "last_eval_runtime_state": lora_health.last_eval_runtime_state,
        "last_saved_probe": lora_health.last_saved_probe,
        "dataset_stats": summarise_supervised_records(
            records,
            max_seq_length=resolved_max_seq_length,
            tokenizer=tokenizer,
        ),
        "resolved_max_seq_length": resolved_max_seq_length,
    }
    write_json(output_dir / "training_metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or dry-run a Nemotron LoRA adapter.")
    parser.add_argument("--config", default="configs/train_lora.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-stats-only", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()

    config = read_yaml(args.config)
    validate_lora_config(config)

    config_dry_run = bool(config.get("training", {}).get("dry_run", False))
    if args.dry_run or args.dry_run_stats_only or (config_dry_run and not args.force_train):
        output_dir = Path(config.get("training", {}).get("output_dir", "artifacts/adapter"))
        write_json(output_dir / "dry_run_manifest.json", dry_run_manifest(config))
        return

    metadata = train_lora(config)
    output_dir = Path(config.get("training", {}).get("output_dir", "artifacts/adapter"))
    write_json(output_dir / "last_run_summary.json", metadata)


if __name__ == "__main__":
    main()
