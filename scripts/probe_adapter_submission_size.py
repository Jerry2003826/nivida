from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import read_yaml, write_json
from src.student import adapter_submission_budget as budget_module
from src.student.adapter_submission_budget import (
    ADAPTER_WEIGHT_BYTES,
    KNOWN_TARGET_SUFFIXES,
    NEMOTRON_ZIP_COMPRESSION_RATIO,
    NEMOTRON_ZIP_OVERHEAD_BYTES,
    ensure_submission_budget_safe,
)
from src.student.lora_train import resolve_model_path
from src.student.package_submission import build_submission_zip


TINY_NEMOTRON_LIKE_PAYLOAD = {
    "model_type": "nemotron_h",
    "hidden_size": 64,
    "head_dim": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 8,
    "hybrid_override_pattern": "ME*ME*ME",
    "n_routed_experts": 1,
    "n_shared_experts": 1,
    "moe_intermediate_size": 48,
    "moe_shared_expert_intermediate_size": 72,
    "mamba_in_proj_out_features": 96,
    "mamba_out_proj_in_features": 80,
}


def _normalise_target_modules(config: dict[str, Any]) -> str | list[str]:
    return budget_module._normalise_target_modules(
        dict(config.get("lora", {})).get("target_modules")
    )


def _selected_suffixes(target_modules: str | list[str]) -> list[str]:
    return budget_module._selected_suffixes(target_modules)


def _formula_from_payload(
    *,
    config_payload: dict[str, Any],
    target_modules: str | list[str],
    rank: int,
) -> dict[str, Any]:
    arch, arch_error = budget_module._nemotron_arch_summary(config_payload)
    if arch is None:
        raise ValueError(arch_error or "tiny probe payload is not a valid nemotron_h config")

    selected_suffixes = _selected_suffixes(target_modules)
    per_suffix_counts = budget_module._nemotron_per_suffix_counts(arch)
    per_suffix_total_bytes = budget_module._nemotron_per_suffix_total_bytes(
        arch,
        rank=rank,
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

    projected_zip_bytes = int(
        math.ceil(
            projected_adapter_bytes * NEMOTRON_ZIP_COMPRESSION_RATIO
            + NEMOTRON_ZIP_OVERHEAD_BYTES
        )
    )
    return {
        "mamba_layers": arch["mamba_layers"],
        "moe_layers": arch["moe_layers"],
        "attention_layers": arch["attention_layers"],
        "n_routed_experts_per_layer": arch["n_routed_experts_per_layer"],
        "n_shared_experts_per_layer": arch["n_shared_experts_per_layer"],
        "selected_suffixes": selected_suffixes,
        "suffix_breakdown": suffix_breakdown,
        "projected_adapter_bytes": projected_adapter_bytes,
        "projected_submission_zip_bytes": projected_zip_bytes,
    }


def _suffix_dimensions(
    arch: dict[str, Any],
    suffix: str,
    *,
    expert_kind: str | None = None,
) -> tuple[int, int]:
    if suffix == "in_proj":
        return arch["hidden_size"], arch["mamba_in_proj_out_features"]
    if suffix == "out_proj":
        return arch["mamba_out_proj_in_features"], arch["hidden_size"]
    if suffix == "up_proj":
        intermediate = (
            arch["moe_shared_expert_intermediate_size"]
            if expert_kind == "shared"
            else arch["moe_intermediate_size"]
        )
        return arch["hidden_size"], intermediate
    if suffix == "down_proj":
        intermediate = (
            arch["moe_shared_expert_intermediate_size"]
            if expert_kind == "shared"
            else arch["moe_intermediate_size"]
        )
        return intermediate, arch["hidden_size"]
    if suffix == "q_proj":
        return arch["hidden_size"], arch["q_out_features"]
    if suffix == "k_proj":
        return arch["hidden_size"], arch["kv_out_features"]
    if suffix == "v_proj":
        return arch["hidden_size"], arch["kv_out_features"]
    if suffix == "o_proj":
        return arch["q_out_features"], arch["hidden_size"]
    raise KeyError(f"Unsupported suffix for probe tensor generation: {suffix}")


def _write_adapter_config(
    adapter_dir: Path,
    *,
    target_modules: str | list[str],
    rank: int,
    base_model_name_or_path: str,
) -> None:
    payload = {
        "peft_type": "LORA",
        "r": int(rank),
        "target_modules": target_modules,
        "base_model_name_or_path": base_model_name_or_path,
    }
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _entropy_bytes(size: int) -> bytes:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=size, dtype=np.uint8).tobytes()


def _write_entropy_pad(path: Path, size: int) -> None:
    if size <= 0:
        if path.exists():
            path.unlink()
        return
    chunk_size = 1_048_576
    with path.open("wb") as handle:
        remaining = size
        while remaining > 0:
            chunk = min(remaining, chunk_size)
            handle.write(_entropy_bytes(chunk))
            remaining -= chunk


def _build_zip(adapter_dir: Path, zip_path: Path) -> int:
    build_submission_zip(adapter_dir, zip_path)
    return int(zip_path.stat().st_size)


def _tune_entropy_pad(
    *,
    adapter_dir: Path,
    zip_path: Path,
    target_zip_bytes: int,
) -> tuple[int, int]:
    pad_path = adapter_dir / "probe_entropy_pad.bin"
    best_size = _build_zip(adapter_dir, zip_path)
    best_pad_bytes = 0
    best_error = abs(target_zip_bytes - best_size)
    pad_bytes = max(0, target_zip_bytes - best_size)

    for _ in range(8):
        _write_entropy_pad(pad_path, pad_bytes)
        current_size = _build_zip(adapter_dir, zip_path)
        current_error = abs(target_zip_bytes - current_size)
        if current_error < best_error:
            best_error = current_error
            best_size = current_size
            best_pad_bytes = pad_bytes
        if target_zip_bytes > 0 and current_error / target_zip_bytes <= 0.01:
            best_size = current_size
            best_pad_bytes = pad_bytes
            break
        delta = target_zip_bytes - current_size
        if delta == 0:
            best_size = current_size
            best_pad_bytes = pad_bytes
            break
        pad_bytes = max(0, pad_bytes + delta)

    _write_entropy_pad(pad_path, best_pad_bytes)
    final_size = _build_zip(adapter_dir, zip_path)
    if target_zip_bytes > 0 and abs(target_zip_bytes - final_size) / target_zip_bytes > 0.05:
        raise ValueError(
            f"Tiny-mode probe zip could not be tuned within 5% of target: "
            f"target={target_zip_bytes} actual={final_size}"
        )
    return final_size, best_pad_bytes


def _suffix_from_lora_key(key: str) -> str | None:
    for suffix in KNOWN_TARGET_SUFFIXES:
        if f".{suffix}.lora_" in key:
            return suffix
    return None


def _measure_saved_adapter(adapter_model_path: Path) -> dict[str, Any]:
    dtype_names: set[str] = set()
    suffix_breakdown: dict[str, dict[str, Any]] = {}
    total_weight_bytes = 0

    with safe_open(adapter_model_path, framework="np") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            total_weight_bytes += int(tensor.nbytes)
            dtype_names.add(str(tensor.dtype))
            suffix = _suffix_from_lora_key(key)
            if suffix is None:
                continue
            module_path = key.rsplit(".lora_", 1)[0]
            entry = suffix_breakdown.setdefault(
                suffix,
                {"tensor_bytes": 0, "module_paths": set()},
            )
            entry["tensor_bytes"] += int(tensor.nbytes)
            entry["module_paths"].add(module_path)

    normalised_breakdown = {
        suffix: {
            "module_count": len(entry["module_paths"]),
            "tensor_bytes": entry["tensor_bytes"],
        }
        for suffix, entry in suffix_breakdown.items()
    }
    dtype_byte_sizes = sorted(
        {
            np.dtype(dtype_name).itemsize
            for dtype_name in dtype_names
        }
    )
    return {
        "weight_bytes_counted_from_safetensors": total_weight_bytes,
        "suffix_breakdown_measured": normalised_breakdown,
        "dtype_names": sorted(dtype_names),
        "implied_real_dtype_bytes": dtype_byte_sizes[0] if len(dtype_byte_sizes) == 1 else dtype_byte_sizes,
    }


def _generate_tiny_adapter(
    *,
    adapter_dir: Path,
    target_modules: str | list[str],
    rank: int,
) -> dict[str, Any]:
    formula = _formula_from_payload(
        config_payload=TINY_NEMOTRON_LIKE_PAYLOAD,
        target_modules=target_modules,
        rank=rank,
    )
    arch, _ = budget_module._nemotron_arch_summary(TINY_NEMOTRON_LIKE_PAYLOAD)
    assert arch is not None

    tensors: dict[str, np.ndarray] = {}
    layer_kinds = [char for char in TINY_NEMOTRON_LIKE_PAYLOAD["hybrid_override_pattern"] if char != "-"]
    selected_suffixes = set(formula["selected_suffixes"])

    for layer_index, layer_kind in enumerate(layer_kinds):
        if layer_kind == "M":
            suffixes = [suffix for suffix in ("in_proj", "out_proj") if suffix in selected_suffixes]
            for suffix in suffixes:
                in_features, out_features = _suffix_dimensions(arch, suffix)
                prefix = f"base_model.model.layers.{layer_index}.mixer.{suffix}"
                tensors[f"{prefix}.lora_A.weight"] = np.zeros((rank, in_features), dtype=np.float32)
                tensors[f"{prefix}.lora_B.weight"] = np.zeros((out_features, rank), dtype=np.float32)
        elif layer_kind == "E":
            suffixes = [suffix for suffix in ("up_proj", "down_proj") if suffix in selected_suffixes]
            for expert_index in range(arch["n_routed_experts_per_layer"]):
                for suffix in suffixes:
                    in_features, out_features = _suffix_dimensions(
                        arch,
                        suffix,
                        expert_kind="routed",
                    )
                    prefix = (
                        f"base_model.model.layers.{layer_index}.experts.{expert_index}.{suffix}"
                    )
                    tensors[f"{prefix}.lora_A.weight"] = np.zeros((rank, in_features), dtype=np.float32)
                    tensors[f"{prefix}.lora_B.weight"] = np.zeros((out_features, rank), dtype=np.float32)
            for shared_index in range(arch["n_shared_experts_per_layer"]):
                for suffix in suffixes:
                    in_features, out_features = _suffix_dimensions(
                        arch,
                        suffix,
                        expert_kind="shared",
                    )
                    if arch["n_shared_experts_per_layer"] == 1:
                        prefix = f"base_model.model.layers.{layer_index}.shared_experts.{suffix}"
                    else:
                        prefix = (
                            f"base_model.model.layers.{layer_index}.shared_experts."
                            f"{shared_index}.{suffix}"
                        )
                    tensors[f"{prefix}.lora_A.weight"] = np.zeros((rank, in_features), dtype=np.float32)
                    tensors[f"{prefix}.lora_B.weight"] = np.zeros((out_features, rank), dtype=np.float32)
        elif layer_kind == "*":
            suffixes = [suffix for suffix in ("q_proj", "k_proj", "v_proj", "o_proj") if suffix in selected_suffixes]
            for suffix in suffixes:
                in_features, out_features = _suffix_dimensions(arch, suffix)
                prefix = f"base_model.model.layers.{layer_index}.self_attn.{suffix}"
                tensors[f"{prefix}.lora_A.weight"] = np.zeros((rank, in_features), dtype=np.float32)
                tensors[f"{prefix}.lora_B.weight"] = np.zeros((out_features, rank), dtype=np.float32)

    adapter_model_path = adapter_dir / "adapter_model.safetensors"
    save_file(tensors, str(adapter_model_path))
    _write_adapter_config(
        adapter_dir,
        target_modules=target_modules,
        rank=rank,
        base_model_name_or_path="tiny-nemotron-like",
    )
    return {
        "adapter_model_path": adapter_model_path,
        "formula": formula,
    }


def _generate_real_adapter(
    *,
    config: dict[str, Any],
    adapter_dir: Path,
    target_modules: str | list[str],
    rank: int,
) -> Path:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "Real adapter probe requires torch, peft, and transformers. "
            "Use --tiny-mode in environments without training deps."
        ) from exc

    model_path = resolve_model_path(config)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=config.get("device_map", "auto"),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        torch_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=int(dict(config.get("lora", {})).get("alpha", max(rank, 16))),
        target_modules=target_modules,
        lora_dropout=float(dict(config.get("lora", {})).get("dropout", 0.0)),
        bias=str(dict(config.get("lora", {})).get("bias", "none")),
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)

    # Trigger a real allocation/update path before save_pretrained so the
    # probe observes a normal adapter artifact, not an unmaterialised shell.
    first_param = next(parameter for parameter in peft_model.parameters() if parameter.requires_grad)
    device = first_param.device
    input_ids = torch.zeros((1, 2), dtype=torch.long, device=device)
    outputs = peft_model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss if getattr(outputs, "loss", None) is not None else outputs[0]
    loss.backward()
    peft_model.save_pretrained(adapter_dir)
    return adapter_dir / "adapter_model.safetensors"


def probe_adapter_submission_size(
    *,
    config: dict[str, Any],
    config_path: str | Path,
    output_path: str | Path,
    tiny_mode: bool = False,
) -> dict[str, Any]:
    ensure_submission_budget_safe(config)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem
    adapter_dir = output_path.with_name(f"{stem}_artifact")
    zip_path = output_path.with_name(f"{stem}.zip")
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    target_modules = _normalise_target_modules(config)
    rank = int(dict(config.get("lora", {})).get("rank", 16))

    if tiny_mode:
        generated = _generate_tiny_adapter(
            adapter_dir=adapter_dir,
            target_modules=target_modules,
            rank=rank,
        )
        formula = generated["formula"]
        final_zip_size, entropy_pad_bytes = _tune_entropy_pad(
            adapter_dir=adapter_dir,
            zip_path=zip_path,
            target_zip_bytes=formula["projected_submission_zip_bytes"],
        )
        adapter_model_path = generated["adapter_model_path"]
    else:
        adapter_model_path = _generate_real_adapter(
            config=config,
            adapter_dir=adapter_dir,
            target_modules=target_modules,
            rank=rank,
        )
        formula = budget_module.estimate_submission_budget(
            config,
            target_modules=target_modules,
            rank=rank,
        )
        final_zip_size = _build_zip(adapter_dir, zip_path)
        entropy_pad_bytes = 0

    measured = _measure_saved_adapter(adapter_model_path)
    formula_predicted_zip_bytes = int(formula["projected_submission_zip_bytes"])
    formula_error_pct = abs(formula_predicted_zip_bytes - final_zip_size) / max(1, final_zip_size) * 100.0
    adapter_dir_size_bytes = sum(
        path.stat().st_size
        for path in adapter_dir.iterdir()
        if path.is_file()
    )

    payload: dict[str, Any] = {
        "config_path": str(config_path),
        "tiny_mode": bool(tiny_mode),
        "artifact_dir": str(adapter_dir),
        "adapter_model_path": str(adapter_model_path),
        "zip_path": str(zip_path),
        "target_modules": target_modules,
        "rank": rank,
        "mamba_layers": formula["mamba_layers"],
        "moe_layers": formula["moe_layers"],
        "attention_layers": formula["attention_layers"],
        "selected_suffixes": formula["selected_suffixes"],
        "formula_suffix_breakdown": formula["suffix_breakdown"],
        "adapter_dir_size_bytes": int(adapter_dir_size_bytes),
        "zip_size_bytes": int(final_zip_size),
        "formula_predicted_zip_bytes": formula_predicted_zip_bytes,
        "formula_adapter_bytes": int(formula["projected_adapter_bytes"]),
        "formula_error_pct": formula_error_pct,
        "weight_bytes_counted_from_safetensors": measured["weight_bytes_counted_from_safetensors"],
        "implied_real_dtype_bytes": measured["implied_real_dtype_bytes"],
        "dtype_names": measured["dtype_names"],
        "suffix_breakdown_measured": measured["suffix_breakdown_measured"],
        "compression_ratio_estimate": NEMOTRON_ZIP_COMPRESSION_RATIO,
        "compression_overhead_bytes": NEMOTRON_ZIP_OVERHEAD_BYTES,
        "entropy_pad_bytes": int(entropy_pad_bytes),
        "adapter_weight_bytes": ADAPTER_WEIGHT_BYTES,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if tiny_mode:
        payload["expected_analytical_zip_bytes"] = formula_predicted_zip_bytes
        payload["formula_smoke_ratio_after_entropy_pad"] = (
            final_zip_size / max(1, int(formula["projected_adapter_bytes"]))
        )
        payload["tiny_mode_is_not_calibration"] = True
    else:
        payload["real_adapter_compression_ratio"] = (
            final_zip_size / max(1, int(formula["projected_adapter_bytes"]))
        )

    write_json(output_path, payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Probe the physical adapter.zip size against the submission budget formula."
    )
    parser.add_argument("--config", default="configs/train_stage2_selected_trace.yaml")
    parser.add_argument("--output", default="artifacts/adapter_submission_probe.json")
    parser.add_argument("--tiny-mode", action="store_true")
    args = parser.parse_args(argv)

    config = read_yaml(args.config)
    probe_adapter_submission_size(
        config=config,
        config_path=args.config,
        output_path=args.output,
        tiny_mode=bool(args.tiny_mode),
    )


if __name__ == "__main__":
    main()
