from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from src.common.io import read_json, write_json


LOG_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_hashes(adapter_path: str | Path | None) -> dict[str, str]:
    if not adapter_path:
        return {}
    root = Path(adapter_path)
    hashes: dict[str, str] = {}
    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "merge_manifest.json"):
        path = root / name
        if path.is_file():
            hashes[name] = _sha256_file(path)
    return hashes


def _report_summary(report_path: str | Path | None) -> dict[str, Any]:
    if not report_path:
        return {}
    path = Path(report_path)
    payload = read_json(path)
    overall = payload.get("overall", {}) if isinstance(payload, dict) else {}
    family = payload.get("family", {}) if isinstance(payload, dict) else {}
    return {
        "report_path": str(path),
        "official_verify_accuracy": overall.get("official_verify_accuracy"),
        "local_competition_accuracy": overall.get("local_competition_accuracy"),
        "boxed_valid_rate": overall.get("boxed_valid_rate"),
        "avg_prediction_words": overall.get("avg_prediction_words"),
        "family": {
            str(name): {
                "n": values.get("n"),
                "official_verify_accuracy": values.get("official_verify_accuracy"),
                "boxed_valid_rate": values.get("boxed_valid_rate"),
            }
            for name, values in family.items()
            if isinstance(values, dict)
        },
    }


def load_correlation_log(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {"schema_version": LOG_SCHEMA_VERSION, "entries": []}
    payload = read_json(target)
    if not isinstance(payload, dict):
        raise ValueError(f"Correlation log must be a JSON object: {target}")
    payload.setdefault("schema_version", LOG_SCHEMA_VERSION)
    payload.setdefault("entries", [])
    return payload


def append_correlation_entry(
    *,
    log_path: str | Path,
    candidate: str,
    public_score: float | None,
    exact_report: str | Path | None = None,
    adapter_path: str | Path | None = None,
    training_recipe: str = "",
    merge_weights: dict[str, float] | None = None,
    submission_id: str = "",
    notes: str = "",
) -> dict[str, Any]:
    payload = load_correlation_log(log_path)
    entry = {
        "timestamp_utc": _utc_now(),
        "candidate": candidate,
        "submission_id": submission_id,
        "public_score": public_score,
        "training_recipe": training_recipe,
        "merge_weights": merge_weights or {},
        "adapter_path": "" if adapter_path is None else str(adapter_path),
        "adapter_hashes": _adapter_hashes(adapter_path),
        "exact_report": _report_summary(exact_report),
        "notes": notes,
    }
    payload["entries"].append(entry)
    write_json(log_path, payload)
    return payload


def parse_merge_weights(value: str | None) -> dict[str, float]:
    if not value:
        return {}
    weights: dict[str, float] = {}
    for piece in value.split(","):
        name, weight = piece.split("=", 1)
        weights[name.strip()] = float(weight)
    return weights
