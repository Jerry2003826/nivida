from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from src.student.adapter_submission_budget import KAGGLE_SINGLE_FILE_LIMIT_BYTES


REQUIRED_ADAPTER_FILES = {"adapter_config.json"}
ADAPTER_WEIGHT_CANDIDATES = ("adapter_model.safetensors", "adapter_model.bin")

# Kaggle / PEFT only need these two files at the zip root.  We deliberately do
# NOT ship any other file (optimizer state, README, tokenizer, extra safetensors
# shards) because:
#   * the upstream metric notebook recursively searches for adapter_config.json
#     and builds a vLLM LoRARequest from that directory; extra files waste the
#     1 GB single-file budget and can confuse the loader.
#   * training-only files like optimizer.pt or scheduler.pt can accidentally
#     leak if saved into the adapter dir.
SUBMISSION_ZIP_ALLOWLIST = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def validate_adapter_dir(adapter_dir: str | Path) -> list[str]:
    path = Path(adapter_dir)
    if not path.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Adapter path must be a directory: {path}")

    files = sorted(item.name for item in path.iterdir() if item.is_file())
    missing = sorted(REQUIRED_ADAPTER_FILES - set(files))
    if missing:
        raise FileNotFoundError(f"Missing required adapter files: {missing}")
    if not any(candidate in files for candidate in ADAPTER_WEIGHT_CANDIDATES):
        raise FileNotFoundError(
            f"Missing LoRA weight file. Expected one of: {', '.join(ADAPTER_WEIGHT_CANDIDATES)}"
        )
    return files


def read_adapter_config(adapter_dir: str | Path) -> dict[str, Any]:
    config_path = Path(adapter_dir) / "adapter_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def read_adapter_rank(adapter_dir: str | Path) -> int | None:
    payload = read_adapter_config(adapter_dir)
    rank = payload.get("r")
    return None if rank is None else int(rank)


def read_adapter_target_modules(adapter_dir: str | Path) -> str | list[str] | None:
    payload = read_adapter_config(adapter_dir)
    target_modules = payload.get("target_modules")
    if target_modules is None:
        return None
    if isinstance(target_modules, list):
        return [str(item) for item in target_modules]
    return str(target_modules)


def submission_zip_size_bytes(path: str | Path) -> int:
    return int(Path(path).stat().st_size)


def validate_submission_zip_size(
    path: str | Path,
    *,
    max_bytes: int = KAGGLE_SINGLE_FILE_LIMIT_BYTES,
) -> int:
    size_bytes = submission_zip_size_bytes(path)
    if size_bytes > max_bytes:
        raise ValueError(
            f"Submission zip size {size_bytes} exceeds Kaggle single-file limit {max_bytes}"
        )
    return size_bytes


def build_submission_zip(adapter_dir: str | Path, output_path: str | Path) -> Path:
    adapter_path = Path(adapter_dir)
    validate_adapter_dir(adapter_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Allowlist-only packaging.  Prefer adapter_model.safetensors over
    # adapter_model.bin; never ship both.
    included: list[str] = ["adapter_config.json"]
    if (adapter_path / "adapter_model.safetensors").is_file():
        included.append("adapter_model.safetensors")
    elif (adapter_path / "adapter_model.bin").is_file():
        included.append("adapter_model.bin")
    else:
        raise FileNotFoundError(
            f"No adapter_model.safetensors or adapter_model.bin in {adapter_path}"
        )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in included:
            src = adapter_path / name
            if not src.is_file():
                raise FileNotFoundError(f"Allowlisted file missing in adapter dir: {src}")
            archive.write(src, arcname=name)

    # Post-condition: zip roundtrip must contain exactly the allowlisted files.
    with zipfile.ZipFile(output_path, "r") as archive:
        names = archive.namelist()
    if set(names) != set(included):
        raise RuntimeError(
            f"submission.zip contents {names} do not match allowlist {included}"
        )

    validate_submission_zip_size(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a saved LoRA adapter directory into submission.zip")
    parser.add_argument("--adapter-dir", default="artifacts/adapter")
    parser.add_argument("--output", default="submission.zip")
    args = parser.parse_args()

    validate_adapter_dir(args.adapter_dir)
    build_submission_zip(args.adapter_dir, args.output)


if __name__ == "__main__":
    main()
