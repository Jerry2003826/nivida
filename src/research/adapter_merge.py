from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import struct
from typing import Any

import numpy as np


WEIGHT_FILE = "adapter_model.safetensors"
CONFIG_FILE = "adapter_config.json"
MERGE_MANIFEST = "merge_manifest.json"
DEFAULT_MAX_SUBMIT_RANK = 32


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    name: str
    path: Path
    weight: float


@dataclass(frozen=True, slots=True)
class TensorValue:
    array: np.ndarray
    dtype: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_adapter_spec(value: str) -> AdapterSpec:
    if "=" not in value or ":" not in value:
        raise ValueError(f"Adapter spec must be name=path:weight, got: {value}")
    name, rest = value.split("=", 1)
    path_text, weight_text = rest.rsplit(":", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Adapter spec has empty name: {value}")
    weight = float(weight_text)
    return AdapterSpec(name=name, path=Path(path_text), weight=weight)


def read_adapter_config(adapter_dir: str | Path) -> dict[str, Any]:
    path = Path(adapter_dir) / CONFIG_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing {CONFIG_FILE}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_adapter_config(adapter_dir: str | Path, config: dict[str, Any]) -> None:
    path = Path(adapter_dir) / CONFIG_FILE
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_safetensors(path: Path) -> dict[str, TensorValue]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"Invalid safetensors file: {path}")
    header_len = struct.unpack("<Q", raw[:8])[0]
    header_start = 8
    header_end = header_start + header_len
    header = json.loads(raw[header_start:header_end].decode("utf-8"))
    data_start = header_end
    tensors: dict[str, TensorValue] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = str(info["dtype"])
        shape = tuple(int(value) for value in info["shape"])
        start, end = (int(value) for value in info["data_offsets"])
        payload = raw[data_start + start : data_start + end]
        tensors[name] = TensorValue(array=_decode_tensor(payload, dtype=dtype, shape=shape), dtype=dtype)
    return tensors


def _write_safetensors(path: Path, tensors: dict[str, TensorValue], *, metadata: dict[str, str] | None = None) -> None:
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(key): str(value) for key, value in metadata.items()}
    data_parts: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        payload = _encode_tensor(tensor.array, dtype=tensor.dtype)
        header[name] = {
            "dtype": tensor.dtype,
            "shape": [int(value) for value in tensor.array.shape],
            "data_offsets": [offset, offset + len(payload)],
        }
        data_parts.append(payload)
        offset += len(payload)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(data_parts))


def _decode_tensor(payload: bytes, *, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    if dtype == "BF16":
        raw = np.frombuffer(payload, dtype="<u2").copy()
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    dtype_map = {
        "F64": "<f8",
        "F32": "<f4",
        "F16": "<f2",
        "I64": "<i8",
        "I32": "<i4",
        "I16": "<i2",
        "I8": "<i1",
        "U8": "<u1",
        "BOOL": "?",
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")
    return np.frombuffer(payload, dtype=np.dtype(dtype_map[dtype])).copy().reshape(shape)


def _encode_tensor(array: np.ndarray, *, dtype: str) -> bytes:
    if dtype == "BF16":
        values = np.ascontiguousarray(array.astype(np.float32, copy=False))
        bits = values.view(np.uint32)
        rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
        return (rounded >> np.uint32(16)).astype("<u2", copy=False).tobytes()
    dtype_map = {
        "F64": "<f8",
        "F32": "<f4",
        "F16": "<f2",
        "I64": "<i8",
        "I32": "<i4",
        "I16": "<i2",
        "I8": "<i1",
        "U8": "<u1",
        "BOOL": "?",
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")
    return array.astype(np.dtype(dtype_map[dtype]), copy=False).tobytes()


def _load_adapter_tensors(adapter_dir: Path) -> dict[str, TensorValue]:
    weights = adapter_dir / WEIGHT_FILE
    if not weights.is_file():
        raise FileNotFoundError(f"Only {WEIGHT_FILE} is supported for merge: {weights}")
    return _read_safetensors(weights)


def _check_compatible_configs(configs: list[dict[str, Any]]) -> None:
    if not configs:
        raise ValueError("At least one adapter is required")
    reference = configs[0]
    for config in configs[1:]:
        for key in ("peft_type", "task_type", "target_modules"):
            if config.get(key) != reference.get(key):
                raise ValueError(f"Adapter config mismatch for {key}: {config.get(key)!r} != {reference.get(key)!r}")


def _check_same_tensor_schema(named_tensors: list[tuple[str, dict[str, TensorValue]]]) -> list[str]:
    reference_name, reference = named_tensors[0]
    keys = sorted(reference)
    for name, tensors in named_tensors[1:]:
        if sorted(tensors) != keys:
            raise ValueError(f"Tensor key mismatch between {reference_name} and {name}")
        for key in keys:
            if tensors[key].array.shape != reference[key].array.shape:
                raise ValueError(
                    f"Tensor shape mismatch for {key}: {name} has {tensors[key].array.shape}, "
                    f"{reference_name} has {reference[key].array.shape}"
                )
    return keys


def _linear_merge(specs: list[AdapterSpec], tensors: list[tuple[str, dict[str, TensorValue]]]) -> dict[str, TensorValue]:
    keys = _check_same_tensor_schema(tensors)
    merged: dict[str, TensorValue] = {}
    for key in keys:
        reference = tensors[0][1][key]
        value = np.zeros_like(reference.array, dtype=np.float32)
        for spec, (_, adapter_tensors) in zip(specs, tensors, strict=True):
            value += float(spec.weight) * adapter_tensors[key].array.astype(np.float32)
        merged[key] = TensorValue(array=value, dtype=reference.dtype)
    return merged


def _lora_pair_key(key: str, marker: str) -> str | None:
    if marker not in key:
        return None
    return key.replace(marker, ".")


def _rank32_svd_merge(
    specs: list[AdapterSpec],
    configs: list[dict[str, Any]],
    tensors: list[tuple[str, dict[str, TensorValue]]],
    *,
    target_rank: int,
) -> tuple[dict[str, TensorValue], int]:
    keys = _check_same_tensor_schema(tensors)
    a_by_base = {_lora_pair_key(key, ".lora_A."): key for key in keys if ".lora_A." in key}
    b_by_base = {_lora_pair_key(key, ".lora_B."): key for key in keys if ".lora_B." in key}
    if sorted(a_by_base) != sorted(b_by_base):
        raise ValueError("SVD merge requires matching lora_A/lora_B tensor pairs")
    if len(a_by_base) * 2 != len(keys):
        extra = sorted(set(keys) - set(a_by_base.values()) - set(b_by_base.values()))
        raise ValueError(f"SVD merge only supports LoRA A/B tensors, found extra keys: {extra}")

    merged: dict[str, TensorValue] = {}
    actual_rank = int(target_rank)
    for base in sorted(a_by_base):
        a_key = a_by_base[base]
        b_key = b_by_base[base]
        ref_a = tensors[0][1][a_key]
        ref_b = tensors[0][1][b_key]
        out_dim, _ = ref_b.array.shape
        _, in_dim = ref_a.array.shape
        delta = np.zeros((out_dim, in_dim), dtype=np.float32)
        for spec, config, (_, adapter_tensors) in zip(specs, configs, tensors, strict=True):
            rank = int(config.get("r") or adapter_tensors[a_key].array.shape[0])
            alpha = float(config.get("lora_alpha") or rank)
            scale = alpha / float(rank)
            delta += float(spec.weight) * scale * (
                adapter_tensors[b_key].array.astype(np.float32)
                @ adapter_tensors[a_key].array.astype(np.float32)
            )
        u, singular, vh = np.linalg.svd(delta, full_matrices=False)
        rank = min(int(target_rank), int(singular.shape[0]))
        actual_rank = min(actual_rank, rank)
        root = np.sqrt(singular[:rank]).astype(np.float32)
        b_new = u[:, :rank].astype(np.float32) * root.reshape(1, rank)
        a_new = root.reshape(rank, 1) * vh[:rank, :].astype(np.float32)
        merged[a_key] = TensorValue(array=a_new, dtype="F32" if ref_a.dtype == "BF16" else ref_a.dtype)
        merged[b_key] = TensorValue(array=b_new, dtype="F32" if ref_b.dtype == "BF16" else ref_b.dtype)
    return merged, actual_rank


def _tensor_summary(tensors: dict[str, TensorValue]) -> dict[str, Any]:
    total_params = 0
    squared = 0.0
    max_abs = 0.0
    for tensor in tensors.values():
        arr = tensor.array.astype(np.float64)
        total_params += int(arr.size)
        squared += float(np.sum(arr * arr))
        if arr.size:
            max_abs = max(max_abs, float(np.max(np.abs(arr))))
    return {
        "num_tensors": len(tensors),
        "total_params": total_params,
        "l2_norm": float(np.sqrt(squared)),
        "max_abs": max_abs,
    }


def merge_lora_adapters(
    *,
    adapters: list[AdapterSpec],
    output_dir: str | Path,
    method: str = "linear",
    target_rank: int = DEFAULT_MAX_SUBMIT_RANK,
    max_submit_rank: int = DEFAULT_MAX_SUBMIT_RANK,
) -> dict[str, Any]:
    if not adapters:
        raise ValueError("At least one adapter is required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    configs = [read_adapter_config(spec.path) for spec in adapters]
    _check_compatible_configs(configs)
    named_tensors = [(spec.name, _load_adapter_tensors(spec.path)) for spec in adapters]
    if method == "linear":
        merged = _linear_merge(adapters, named_tensors)
        output_config = dict(configs[0])
        output_rank = int(output_config.get("r") or 0)
    elif method == "svd-rank32":
        merged, output_rank = _rank32_svd_merge(
            adapters,
            configs,
            named_tensors,
            target_rank=int(target_rank),
        )
        output_config = dict(configs[0])
        output_config["r"] = int(output_rank)
        output_config["lora_alpha"] = int(output_rank)
    else:
        raise ValueError(f"Unsupported merge method: {method}")

    write_adapter_config(output, output_config)
    _write_safetensors(
        output / WEIGHT_FILE,
        merged,
        metadata={
            "merge_method": method,
            "created_by": "scripts/merge_lora_adapters.py",
        },
    )

    extra_files = [
        item.name
        for item in output.iterdir()
        if item.is_file() and item.name not in {CONFIG_FILE, WEIGHT_FILE, MERGE_MANIFEST}
    ]
    submit_safe = int(output_rank or 0) <= int(max_submit_rank) and not extra_files
    manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "method": method,
        "output_dir": str(output),
        "output_rank": output_rank,
        "max_submit_rank": int(max_submit_rank),
        "submit_safe": submit_safe,
        "submit_safe_reason": "adapter-only rank check passed"
        if submit_safe
        else "output rank exceeds max_submit_rank or extra files are present",
        "sources": [
            {
                "name": spec.name,
                "path": str(spec.path),
                "weight": float(spec.weight),
                "adapter_config_sha256": sha256_file(spec.path / CONFIG_FILE),
                "adapter_model_sha256": sha256_file(spec.path / WEIGHT_FILE),
            }
            for spec in adapters
        ],
        "output": {
            "adapter_config_sha256": sha256_file(output / CONFIG_FILE),
            "adapter_model_sha256": sha256_file(output / WEIGHT_FILE),
            "allowlist_files": [CONFIG_FILE, WEIGHT_FILE],
        },
        "tensor_summary": _tensor_summary(merged),
        "config_summary": {
            "peft_type": output_config.get("peft_type"),
            "task_type": output_config.get("task_type"),
            "target_modules": output_config.get("target_modules"),
            "r": output_config.get("r"),
            "lora_alpha": output_config.get("lora_alpha"),
        },
    }
    (output / MERGE_MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def clean_output_dir(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
