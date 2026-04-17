from __future__ import annotations

import argparse
import importlib
import inspect
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.io import load_jsonl, read_yaml, write_json


def validate_lora_config(config: dict[str, Any]) -> None:
    rank = int(config.get("lora", {}).get("rank", 16))
    if rank > 32:
        raise ValueError(f"LoRA rank must be <= 32, got {rank}")


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


def dry_run_manifest(config: dict[str, Any]) -> dict[str, Any]:
    validate_lora_config(config)
    training = config.get("training", {})
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
        "resolved_max_seq_length": resolve_max_seq_length(config),
        "resolved_target_modules": normalise_target_modules(config.get("lora", {}).get("target_modules")),
        "status": "dry_run_ok",
        "notes": "Training loop wired for Kaggle demo compatibility; dry-run skips model loading and optimization.",
    }


@dataclass(slots=True)
class SupervisedRecord:
    prompt: str
    completion: str


def load_supervised_records(path: str | Path) -> list[SupervisedRecord]:
    rows = load_jsonl(path)
    return [
        SupervisedRecord(prompt=str(row["prompt"]), completion=str(row["completion"]))
        for row in rows
        if row.get("prompt") and row.get("completion")
    ]


def _build_text_sample(record: SupervisedRecord, eos_token: str = "") -> str:
    return f"{record.prompt.rstrip()}\n{record.completion.strip()}{eos_token}"


def _simple_token_count(text: str) -> int:
    return len(text.split())


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
    lengths.sort()
    percentile_index = max(0, math.ceil(0.95 * len(lengths)) - 1)
    p95 = lengths[percentile_index]
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
        "logging_steps": int(training_config.get("logging_steps", 10)),
        "save_steps": int(training_config.get("save_steps", 100)),
        "eval_steps": int(training_config.get("eval_steps", 100)),
        "save_total_limit": int(training_config.get("save_total_limit", 2)),
        "bf16": _normalise_dtype(config.get("torch_dtype", "bfloat16")) == "bfloat16",
        "fp16": _normalise_dtype(config.get("torch_dtype", "bfloat16")) == "float16",
        "report_to": list(training_config.get("report_to", [])),
        "remove_unused_columns": False,
        "dataloader_pin_memory": bool(training_config.get("dataloader_pin_memory", False)),
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
    if isinstance(raw_target_modules, list):
        return [str(item) for item in raw_target_modules]
    if raw_target_modules is None:
        return r".*\.(in_proj|out_proj|up_proj|down_proj)$"
    return str(raw_target_modules)


def requires_mamba_ssm(config: dict[str, Any]) -> bool:
    source = str(config.get("model_source", "huggingface")).lower()
    base_model = str(config.get("base_model", "")).lower()
    model_handle = str(config.get("model_handle", "")).lower()
    if source == "kagglehub":
        return True
    if "nemotron" in base_model or "mamba" in base_model:
        return True
    if "nemotron" in model_handle or "mamba" in model_handle:
        return True
    return bool(config.get("trust_remote_code", False) and ("nemotron" in base_model or "nemotron" in model_handle))


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


def train_lora(config: dict[str, Any]) -> dict[str, Any]:
    validate_lora_config(config)
    applied_environment = apply_runtime_environment(config)

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

    target_modules = normalise_target_modules(config.get("lora", {}).get("target_modules"))
    lora_config = peft.LoraConfig(
        r=int(config["lora"]["rank"]),
        lora_alpha=int(config["lora"].get("alpha", 16)),
        target_modules=target_modules,
        lora_dropout=float(config["lora"].get("dropout", 0.05)),
        bias=str(config["lora"].get("bias", "none")),
        task_type=peft.TaskType.CAUSAL_LM,
    )
    model = peft.get_peft_model(model, lora_config)

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
        "adapter_files": adapter_files,
        "target_modules": target_modules,
        "resolved_max_seq_length": resolved_max_seq_length,
    }
    write_json(output_dir / "training_metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or dry-run a Nemotron LoRA adapter.")
    parser.add_argument("--config", default="configs/train_lora.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = read_yaml(args.config)
    validate_lora_config(config)
    output_dir = Path(config.get("training", {}).get("output_dir", "artifacts/adapter"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run or config.get("training", {}).get("dry_run", True):
        write_json(output_dir / "dry_run_manifest.json", dry_run_manifest(config))
        return

    metadata = train_lora(config)
    write_json(output_dir / "last_run_summary.json", metadata)


if __name__ == "__main__":
    main()
