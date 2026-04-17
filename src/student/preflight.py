from __future__ import annotations

import hashlib
import importlib
import os
import shutil
from pathlib import Path
from typing import Any

from src.common.io import read_json


TOKENIZER_FILE_CANDIDATES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "chat_template.jinja",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
)
MODEL_WEIGHT_EXTENSIONS = {".safetensors", ".bin", ".pt"}
DEFAULT_OUTPUT_DIR = "artifacts/adapter"
DEFAULT_KAGGLEHUB_CACHE = Path.home() / ".cache" / "kagglehub"
REPO_ROOT = Path(__file__).resolve().parents[2]


class TrainingPreflightError(RuntimeError):
    """Raised when a training run is guaranteed to fail before the trainer loop."""


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
    return bool(
        config.get("trust_remote_code", False)
        and ("nemotron" in base_model or "nemotron" in model_handle)
    )


def _import_or_raise(module_name: str, remediation: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise TrainingPreflightError(
            f"Missing dependency '{module_name}'. {remediation}"
        ) from exc


def _repo_root(repo_root: Path | None) -> Path:
    return (repo_root or REPO_ROOT).resolve()


def _path_for_display(path: str | Path | None, repo_root: Path) -> str | None:
    if path in (None, ""):
        return None
    raw = Path(path)
    absolute = raw if raw.is_absolute() else repo_root / raw
    try:
        return absolute.resolve().relative_to(repo_root).as_posix()
    except Exception:
        return raw.as_posix()


def _path_for_compare(path: str | Path | None, repo_root: Path) -> str | None:
    if path in (None, ""):
        return None
    raw = Path(path)
    absolute = raw if raw.is_absolute() else repo_root / raw
    return str(absolute.resolve())


def _resolve_config_path(path: str | Path, repo_root: Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else repo_root / raw


def _record_success(
    report: dict[str, Any],
    name: str,
    detail: str,
    **extra: Any,
) -> None:
    report["checks"][name] = {"ok": True, "detail": detail, **extra}


def _raise_failure(name: str, detail: str, remediation: str) -> None:
    raise TrainingPreflightError(f"{name}: {detail}. {remediation}")


def _ensure_readable_file(path: Path, label: str) -> None:
    if not path.is_file():
        _raise_failure(
            label,
            f"Expected a readable file at {path}",
            "Regenerate the dataset artifact or fix the configured path.",
        )
    try:
        with path.open("rb"):
            return
    except OSError as exc:
        _raise_failure(
            label,
            f"Unable to read {path} ({type(exc).__name__})",
            "Fix the file permissions or regenerate the file.",
        )


def _ensure_output_dir_writable(output_dir: Path) -> Path:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _raise_failure(
            "output_dir",
            f"Unable to create {output_dir} ({type(exc).__name__})",
            "Choose a writable output_dir or fix the parent directory permissions.",
        )

    probe_path = output_dir / ".preflight_write_probe"
    try:
        with probe_path.open("w", encoding="utf-8") as handle:
            handle.write("ok")
    except OSError as exc:
        _raise_failure(
            "output_dir",
            f"Unable to write to {output_dir} ({type(exc).__name__})",
            "Choose a writable output_dir or clean up a stale file lock.",
        )
    finally:
        if probe_path.exists():
            probe_path.unlink()
    return output_dir


def _ensure_directory(path: Path, label: str, remediation: str) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _raise_failure(
            label,
            f"Unable to create {path} ({type(exc).__name__})",
            remediation,
        )
    return path


def _tokenizer_files(tokenizer_path: Path) -> list[str]:
    if not tokenizer_path.is_dir():
        _raise_failure(
            "tokenizer_path",
            f"Expected a tokenizer directory at {tokenizer_path}",
            "Populate it with scripts/probe_chat_template.py or fix tokenizer_path.",
        )
    files = [name for name in TOKENIZER_FILE_CANDIDATES if (tokenizer_path / name).exists()]
    if not files:
        _raise_failure(
            "tokenizer_path",
            f"No tokenizer files found in {tokenizer_path}",
            "Populate it with scripts/probe_chat_template.py before training.",
        )
    return files


def _download_kaggle_probe_file(model_handle: str) -> Path:
    if not model_handle:
        _raise_failure(
            "kaggle_model_access",
            "model_handle is required when model_source=kagglehub",
            "Set config.model_handle to the Kaggle model bundle handle.",
        )
    kagglehub = _import_or_raise(
        "kagglehub",
        "Install it with 'pip install kagglehub' before running non-dry training.",
    )
    try:
        return Path(
            kagglehub.model_download(
                model_handle,
                path="tokenizer_config.json",
            )
        )
    except Exception as exc:
        _raise_failure(
            "kaggle_model_access",
            (
                "Kaggle model access failed while downloading "
                f"'tokenizer_config.json' for {model_handle} ({type(exc).__name__}: {exc})"
            ),
            "Verify Kaggle credentials, model T&C acceptance, and network access.",
        )
    raise AssertionError("unreachable")


def _bundle_root_has_model_weights(bundle_root: Path) -> bool:
    if not bundle_root.exists():
        return False
    for candidate in bundle_root.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in MODEL_WEIGHT_EXTENSIONS:
            return True
    return False


def _configured_kagglehub_cache(config: dict[str, Any], repo_root: Path) -> Path:
    environment = dict(config.get("environment", {}))
    cache_raw = environment.get("KAGGLEHUB_CACHE") or os.environ.get("KAGGLEHUB_CACHE")
    if cache_raw:
        return _resolve_config_path(cache_raw, repo_root)
    return DEFAULT_KAGGLEHUB_CACHE


def _disk_space_report(target_dir: Path, required_gb: int) -> float:
    usage = shutil.disk_usage(target_dir)
    free_gb = usage.free / (1024**3)
    if free_gb < required_gb:
        _raise_failure(
            "disk_space",
            (
                f"Only {free_gb:.2f} GB free under {target_dir}; "
                f"required at least {required_gb} GB"
            ),
            "Free disk space or point the Kaggle/model output paths at a larger volume before retrying.",
        )
    return round(free_gb, 2)


def _probe_chat_template_sha16(
    config: dict[str, Any],
    tokenizer_path: Path | None,
    repo_root: Path,
) -> str | None:
    probe_path = repo_root / "artifacts" / "chat_template_probe.json"
    if not probe_path.exists():
        return None
    try:
        payload = read_json(probe_path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    payload_model_handle = payload.get("model_handle")
    config_model_handle = config.get("model_handle")
    if config_model_handle and payload_model_handle != config_model_handle:
        return None

    payload_tokenizer_path = payload.get("tokenizer_path")
    if tokenizer_path is not None and payload_tokenizer_path:
        if _path_for_compare(payload_tokenizer_path, repo_root) != str(tokenizer_path.resolve()):
            return None
    sha = payload.get("chat_template_sha16")
    return str(sha) if sha else None


def _load_chat_template_sha16(tokenizer_path: Path | None) -> str | None:
    if tokenizer_path is None or not tokenizer_path.exists():
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code=True,
        )
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                trust_remote_code=True,
                use_fast=False,
            )
        except Exception:
            return None
    template = getattr(tokenizer, "chat_template", "") or ""
    if not template:
        return None
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def run_training_preflight(
    config: dict[str, Any],
    *,
    dry_run: bool,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Fail fast on environment issues before the trainer loop begins."""
    repo_root = _repo_root(repo_root)
    training = dict(config.get("training", {}))
    dataset_path_raw = training.get("dataset_path")
    if not dataset_path_raw:
        _raise_failure(
            "dataset_path",
            "training.dataset_path is required",
            "Point the config at a generated SFT JSONL dataset.",
        )
    dataset_path = _resolve_config_path(dataset_path_raw, repo_root)
    _ensure_readable_file(dataset_path, "dataset_path")

    eval_path_raw = training.get("eval_path")
    if eval_path_raw:
        eval_path = _resolve_config_path(eval_path_raw, repo_root)
        _ensure_readable_file(eval_path, "eval_path")
    else:
        eval_path = None

    output_dir = _resolve_config_path(training.get("output_dir", DEFAULT_OUTPUT_DIR), repo_root)
    writable_output_dir = _ensure_output_dir_writable(output_dir)

    tokenizer_path_raw = config.get("tokenizer_path")
    tokenizer_path = (
        _resolve_config_path(tokenizer_path_raw, repo_root)
        if tokenizer_path_raw
        else None
    )
    tokenizer_files: list[str] = []
    if tokenizer_path is not None:
        tokenizer_files = _tokenizer_files(tokenizer_path)

    report: dict[str, Any] = {
        "status": "ok",
        "checks": {},
        "disk_free_gb": None,
        "disk_check_path": None,
        "required_disk_gb": None,
        "kaggle_model_cached": None,
        "tokenizer_path": _path_for_display(tokenizer_path, repo_root),
        "chat_template_sha16": None,
    }
    _record_success(
        report,
        "dataset_path",
        f"Readable dataset at {_path_for_display(dataset_path, repo_root)}",
    )
    if eval_path is not None:
        _record_success(
            report,
            "eval_path",
            f"Readable eval dataset at {_path_for_display(eval_path, repo_root)}",
        )
    else:
        _record_success(report, "eval_path", "Not configured; skipped.")
    _record_success(
        report,
        "output_dir",
        f"Writable output directory at {_path_for_display(writable_output_dir, repo_root)}",
    )
    if tokenizer_path is not None:
        _record_success(
            report,
            "tokenizer_path",
            f"Found tokenizer files: {', '.join(tokenizer_files)}",
            path=_path_for_display(tokenizer_path, repo_root),
        )
    else:
        _record_success(report, "tokenizer_path", "Not configured; skipped.")

    report["chat_template_sha16"] = _probe_chat_template_sha16(
        config,
        tokenizer_path,
        repo_root,
    ) or _load_chat_template_sha16(tokenizer_path)

    if dry_run:
        _record_success(
            report,
            "remote_runtime_checks",
            "Skipped for dry-run mode.",
        )
        return report

    runtime_modules = ("torch", "transformers", "peft")
    for module_name in runtime_modules:
        _import_or_raise(
            module_name,
            "Install the training extras with 'pip install -e .[train]' before retrying.",
        )
    _record_success(
        report,
        "runtime_modules",
        f"Imported runtime modules: {', '.join(runtime_modules)}",
    )

    if requires_mamba_ssm(config):
        _import_or_raise(
            "mamba_ssm",
            "Install mamba-ssm in the target environment before running Nemotron training.",
        )
        _record_success(report, "mamba_ssm", "Imported mamba_ssm successfully.")
    else:
        _record_success(report, "mamba_ssm", "Not required for this config.")

    model_source = str(config.get("model_source", "huggingface")).lower()
    model_handle = str(config.get("model_handle", ""))
    if model_source == "kagglehub":
        probe_file = _download_kaggle_probe_file(model_handle)
        report["kaggle_model_cached"] = _bundle_root_has_model_weights(probe_file.parent)
        _record_success(
            report,
            "kaggle_model_access",
            (
                "Successfully downloaded tokenizer_config.json via kagglehub at "
                f"{_path_for_display(probe_file, repo_root)}"
            ),
        )
    else:
        _record_success(report, "kaggle_model_access", "Not required for model_source!=kagglehub.")

    required_disk_gb = 20
    disk_check_path = writable_output_dir
    if model_source == "kagglehub" and report["kaggle_model_cached"] is False:
        required_disk_gb = 80
        disk_check_path = _ensure_directory(
            _configured_kagglehub_cache(config, repo_root),
            "kagglehub_cache",
            "Set environment.KAGGLEHUB_CACHE to a writable directory on a larger volume.",
        )
    report["required_disk_gb"] = required_disk_gb
    report["disk_check_path"] = _path_for_display(disk_check_path, repo_root)
    report["disk_free_gb"] = _disk_space_report(disk_check_path, required_disk_gb)
    _record_success(
        report,
        "disk_space",
        (
            f"{report['disk_free_gb']:.2f} GB free at "
            f"{report['disk_check_path']} "
            f"(required {required_disk_gb} GB)"
        ),
    )
    return report
