from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
from typing import Any


WEIGHT_FILENAMES = ("adapter_model.safetensors", "adapter_model.bin")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _nvidia_smi() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _parse_name_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected name=path, got {value!r}")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"expected name=path, got {value!r}")
    return name.strip(), Path(path.strip())


def _candidate_payload(raw: str, *, repo_root: Path) -> dict[str, Any]:
    name, raw_path = _parse_name_path(raw)
    path = raw_path if raw_path.is_absolute() else repo_root / raw_path
    weights = [
        path / filename
        for filename in WEIGHT_FILENAMES
        if (path / filename).is_file()
    ]
    return {
        "name": name,
        "path": str(raw_path),
        "exists": path.is_dir(),
        "weights": [
            {
                "path": str(weight.relative_to(repo_root) if weight.is_relative_to(repo_root) else weight),
                "size_bytes": weight.stat().st_size,
                "sha256": _sha256(weight),
            }
            for weight in weights
        ],
    }


def _prediction_counts(out_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    if not out_dir.exists():
        return counts
    for path in sorted(out_dir.glob("*/*/raw/repeat_0.jsonl")):
        rel = path.relative_to(out_dir).parts
        if len(rel) < 4:
            continue
        eval_name, candidate = rel[0], rel[1]
        with path.open("r", encoding="utf-8") as handle:
            line_count = sum(1 for line in handle if line.strip())
        counts.setdefault(eval_name, {})[candidate] = line_count
    return counts


def build_cloud_artifact_manifest(
    *,
    repo_root: str | Path,
    out_dir: str | Path,
    eval_inputs: list[str],
    candidates: list[str],
    preflight_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    output_dir = Path(out_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    preflight = None
    preflight_sha = None
    if preflight_path is not None:
        path = Path(preflight_path)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            preflight = json.loads(path.read_text(encoding="utf-8"))
            preflight_sha = _sha256(path)
    return {
        "schema_version": 1,
        "timestamp_utc": _utc_now(),
        "repo": {
            "branch": _git_value(root, "branch", "--show-current"),
            "commit": _git_value(root, "rev-parse", "HEAD"),
            "dirty_status": _git_value(root, "status", "--short"),
        },
        "runtime": {
            "torch": _package_version("torch"),
            "vllm": _package_version("vllm"),
            "transformers": _package_version("transformers"),
            "cuda_visible_devices": None,
            "nvidia_smi": _nvidia_smi(),
        },
        "out_dir": str(output_dir),
        "eval_inputs": eval_inputs,
        "candidates": [_candidate_payload(candidate, repo_root=root) for candidate in candidates],
        "preflight_path": None if preflight_path is None else str(preflight_path),
        "preflight_sha256": preflight_sha,
        "preflight_status": None if not isinstance(preflight, dict) else preflight.get("status"),
        "prediction_line_counts": _prediction_counts(output_dir),
    }
