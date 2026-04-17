from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any


REQUIRED_ADAPTER_FILES = {"adapter_config.json"}
ADAPTER_WEIGHT_CANDIDATES = ("adapter_model.safetensors", "adapter_model.bin")


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


def build_submission_zip(adapter_dir: str | Path, output_path: str | Path) -> Path:
    adapter_path = Path(adapter_dir)
    validate_adapter_dir(adapter_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in adapter_path.iterdir():
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.name)
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
