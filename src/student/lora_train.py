from __future__ import annotations

import argparse
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


def validate_lora_config(config: dict[str, Any]) -> None:
    rank = int(config.get("lora", {}).get("rank", 16))
    if rank > 32:
        raise ValueError(f"LoRA rank must be <= 32, got {rank}")
    ensure_submission_budget_safe(config)


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

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedCollator(),
    )
    trainer.train(resume_from_checkpoint=config["training"].get("resume_from_checkpoint"))

    model.save_pretrained(output_dir)
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
        "target_modules": lora_initialisation["target_modules"],
        "matched_target_modules": lora_initialisation["matched_target_modules"],
        "num_matched_target_modules": lora_initialisation["num_matched_target_modules"],
        "init_adapter_dir": lora_initialisation["init_adapter_dir"],
        "init_adapter_rank": lora_initialisation["init_adapter_rank"],
        "init_adapter_target_modules": lora_initialisation["init_adapter_target_modules"],
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
