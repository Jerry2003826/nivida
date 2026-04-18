from __future__ import annotations

import json
import shutil
from pathlib import Path


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

TOKENIZER_CACHE_DIR = Path("artifacts/_tokenizer_cache")


def import_kagglehub():
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub not installed. Install training extras: "
            "pip install kagglehub transformers"
        ) from exc
    return kagglehub


def import_transformers():
    try:
        from transformers import AutoTokenizer  # noqa: F401
        import transformers
    except ImportError as exc:
        raise SystemExit(
            "transformers not installed. Install: pip install transformers"
        ) from exc
    return transformers


def download_tokenizer_only(kagglehub, model_handle: str, cache_dir: Path) -> Path:
    """Download only tokenizer-related files via the per-file kagglehub API."""
    dest = cache_dir / model_handle.replace("/", "_")
    dest.mkdir(parents=True, exist_ok=True)

    downloaded: list[str] = []
    skipped: list[tuple[str, str]] = []

    def _pull(filename: str) -> bool:
        try:
            src_str = kagglehub.model_download(model_handle, path=filename)
        except Exception as exc:  # noqa: BLE001
            skipped.append((filename, f"{type(exc).__name__}: {exc}"))
            return False
        src_path = Path(src_str)
        if not src_path.is_file():
            skipped.append((filename, f"not a regular file: {src_path}"))
            return False
        target = dest / filename
        if src_path.resolve() != target.resolve():
            shutil.copy2(src_path, target)
        downloaded.append(filename)
        return True

    print(
        f"[probe] tokenizer-only download of {model_handle!r} "
        f"into {dest} ...",
        flush=True,
    )
    for fname in TOKENIZER_FILE_CANDIDATES:
        _pull(fname)

    config_path = dest / "tokenizer_config.json"
    if config_path.exists():
        try:
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(
                f"[probe] WARN: tokenizer_config.json unparsable ({exc}); "
                "skipping auto_map discovery",
                flush=True,
            )
            config_payload = {}
        auto_map = config_payload.get("auto_map", {}) or {}
        custom_modules: set[str] = set()
        for value in auto_map.values() if isinstance(auto_map, dict) else []:
            if isinstance(value, str) and "." in value:
                custom_modules.add(value.split(".", 1)[0] + ".py")
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and "." in item:
                        custom_modules.add(item.split(".", 1)[0] + ".py")
        for py_file in sorted(custom_modules):
            print(f"[probe] auto_map references {py_file}; downloading", flush=True)
            _pull(py_file)

    if not downloaded:
        raise SystemExit(
            f"No tokenizer files could be downloaded from {model_handle!r} via "
            "per-file API. Try --download-full-model, or verify Kaggle credentials "
            "and model T&C acceptance. Details: "
            + "; ".join(f"{n}: {why}" for n, why in skipped[:5])
        )

    print(
        f"[probe] tokenizer files downloaded ({len(downloaded)}): "
        f"{', '.join(downloaded)}",
        flush=True,
    )
    if skipped:
        print(
            f"[probe] skipped {len(skipped)} optional file(s) not present "
            "in this bundle (expected for most models)",
            flush=True,
        )
    return dest


def download_full_model_bundle(kagglehub, model_handle: str) -> Path:
    """Download the full Kaggle model bundle."""
    print(
        f"[probe] FULL download of {model_handle!r} - this may take a long time "
        "and consume tens of GB of disk.",
        flush=True,
    )
    model_path = Path(kagglehub.model_download(model_handle))
    print(f"[probe] model files cached at: {model_path}", flush=True)
    return model_path


def load_probe_tokenizer(
    model_handle: str,
    *,
    download_full_model: bool,
    cache_dir: Path,
):
    """Download what is needed and load the Nemotron tokenizer."""
    kagglehub = import_kagglehub()
    transformers = import_transformers()
    AutoTokenizer = transformers.AutoTokenizer

    if download_full_model:
        source_path = download_full_model_bundle(kagglehub, model_handle)
        mode = "full"
    else:
        source_path = download_tokenizer_only(kagglehub, model_handle, cache_dir)
        mode = "tokenizer_only"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(source_path), trust_remote_code=True
        )
    except Exception as exc_fast:  # noqa: BLE001
        print(
            f"[probe] fast tokenizer load failed ({type(exc_fast).__name__}); "
            "retrying with use_fast=False",
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(source_path), trust_remote_code=True, use_fast=False
        )
    return tokenizer, str(source_path), mode
