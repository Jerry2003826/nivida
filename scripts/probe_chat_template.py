#!/usr/bin/env python
"""Probe Nemotron tokenizer for the chat-template behaviour that docs/harness_alignment.md §6 depends on.

Run this BEFORE writing any PR0 code. It answers three blocking questions:

    1. What exact string does ``apply_chat_template(..., enable_thinking=True)`` emit?
       In particular: does the assistant seed end with ``<think>\\n``?
    2. Does the ``enable_thinking=True`` flag actually change the output string?
       Some tokenizer builds silently ignore it.
    3. Does the first public-test sample (with the harness guard appended) fit
       inside the ``max_model_len − max_tokens = 512`` prompt-token budget?

Download modes (see ``--download-full-model``):

    default  (tokenizer-only)
        Uses ``kagglehub.model_download(handle, path=<filename>)`` to pull only
        tokenizer files. Total disk ≤ 200 MB. Falls back to full download if
        the per-file API rejects every request.

    --download-full-model
        Pulls the entire Kaggle model bundle (~60 GB for Nemotron-3-Nano-30B
        BF16). Use this only if you also plan to run the model on the same
        machine right afterwards (e.g., H100 training or inference smoke).

Requirements beyond requirements.txt:
    pip install kagglehub transformers

Kaggle credentials:
    Either ``~/.kaggle/kaggle.json`` or env vars ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``.
    The first run will also prompt you to accept the model terms of use on Kaggle.

Outputs:
    artifacts/chat_template_probe.json  (structured probe result)
    stdout                              (human-readable summary)

Example:
    python scripts/probe_chat_template.py
    python scripts/probe_chat_template.py --download-full-model
    python scripts/probe_chat_template.py \\
        --model metric/nemotron-3-nano-30b-a3b-bf16/transformers/default \\
        --sample-csv 官方资料/test.csv \\
        --output artifacts/chat_template_probe.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

from src.competition.tokenizer_probe_utils import (
    TOKENIZER_CACHE_DIR,
    load_probe_tokenizer,
)


HARNESS_GUARD = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
PROMPT_TOKEN_BUDGET = 512

# Standard tokenizer file names the probe will try to pull individually.
# Any file missing on a given model is silently skipped.
_TOKENIZER_FILE_CANDIDATES: tuple[str, ...] = (
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

# Default cache root for the per-file tokenizer download path.
_TOKENIZER_CACHE_DIR = Path("artifacts/_tokenizer_cache")


def _read_first_prompt(sample_csv: Path) -> tuple[str, str]:
    """Return (sample_id, prompt_text) for the first non-header row of the csv."""
    with sample_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_value = row.get("prompt", "")
            return str(row.get("id", "") or ""), str(prompt_value or "")
    raise ValueError(f"No rows found in {sample_csv}")


def _import_kagglehub():
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub not installed. Install training extras: "
            "pip install kagglehub transformers"
        ) from exc
    return kagglehub


def _import_transformers():
    try:
        from transformers import AutoTokenizer  # noqa: F401
        import transformers
    except ImportError as exc:
        raise SystemExit(
            "transformers not installed. Install: pip install transformers"
        ) from exc
    return transformers


def _download_tokenizer_only(
    kagglehub, model_handle: str, cache_dir: Path
) -> Path:
    """Download only tokenizer-related files via per-file kagglehub API.

    Returns the local directory populated with the downloaded files, suitable
    for ``AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)``.

    Strategy:

    1. Try each known tokenizer filename via ``model_download(path=filename)``.
       Silently skip files that do not exist in this model bundle.
    2. Parse ``tokenizer_config.json::auto_map`` (if present) to discover any
       custom Python module names the tokenizer depends on, and pull those
       too. This is needed when ``trust_remote_code=True`` binds the tokenizer
       class to a module shipped inside the model directory.
    3. If nothing was downloaded, raise with a clear remediation hint.

    Total on-disk footprint is typically well under 200 MB.
    """
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
    for fname in _TOKENIZER_FILE_CANDIDATES:
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
                module_name = value.split(".", 1)[0] + ".py"
                custom_modules.add(module_name)
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


def _download_full_model(kagglehub, model_handle: str) -> Path:
    """Download the full Kaggle model bundle (~60 GB for Nemotron-3-Nano-30B)."""
    print(
        f"[probe] FULL download of {model_handle!r} — this may take a long time "
        "and consume tens of GB of disk.",
        flush=True,
    )
    model_path = Path(kagglehub.model_download(model_handle))
    print(f"[probe] model files cached at: {model_path}", flush=True)
    return model_path


def _load_tokenizer(model_handle: str, *, download_full_model: bool, cache_dir: Path):
    """Download what is needed and load the Nemotron tokenizer.

    Returns ``(tokenizer, source_dir_str, download_mode)``.

    ``download_full_model=False`` (default) downloads only tokenizer files via the
    per-file kagglehub API. ``download_full_model=True`` pulls the entire model
    bundle and is only appropriate when the caller plans to use the weights
    afterwards.
    """
    kagglehub = _import_kagglehub()
    transformers = _import_transformers()
    AutoTokenizer = transformers.AutoTokenizer

    if download_full_model:
        source_path = _download_full_model(kagglehub, model_handle)
        mode = "full"
    else:
        source_path = _download_tokenizer_only(kagglehub, model_handle, cache_dir)
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


def _try_apply_chat_template(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool | None,
) -> tuple[str | None, str | None]:
    """Call apply_chat_template; return (rendered, error_str_or_None).

    ``enable_thinking=None`` means omit the keyword (for tokenizers that
    don't support it).
    """
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _bpe_length(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _ends_with(text: str, token: str, *, allow_trailing_newline: bool) -> bool:
    stripped = text.rstrip("\n") if allow_trailing_newline else text
    return stripped.endswith(token)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Nemotron tokenizer chat template behaviour. "
            "See docs/harness_alignment.md §6."
        )
    )
    parser.add_argument(
        "--model",
        default="metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
        help="Kaggle model handle (default matches harness).",
    )
    parser.add_argument(
        "--sample-csv",
        default="官方资料/test.csv",
        type=Path,
        help="CSV containing at least an 'id' column and a 'prompt' column. "
        "The first row is used for experiment 3.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/chat_template_probe.json",
        type=Path,
    )
    parser.add_argument(
        "--hello-content",
        default="HELLO",
        help="Minimal content for experiments 1/2. Short on purpose.",
    )
    parser.add_argument(
        "--download-full-model",
        action="store_true",
        help=(
            "Pull the entire Kaggle model bundle (~60 GB for Nemotron-3-Nano-30B). "
            "Default is the tokenizer-only per-file download (≤200 MB). "
            "Use this flag only if you plan to use the weights on the same machine."
        ),
    )
    parser.add_argument(
        "--tokenizer-cache-dir",
        default=str(TOKENIZER_CACHE_DIR),
        type=Path,
        help="Local directory to populate with tokenizer files (tokenizer-only mode).",
    )
    args = parser.parse_args()

    if not args.sample_csv.exists():
        raise SystemExit(
            f"sample CSV not found: {args.sample_csv}. "
            "Provide --sample-csv pointing to a csv with 'id' and 'prompt' columns."
        )

    tokenizer, tokenizer_path, download_mode = load_probe_tokenizer(
        args.model,
        download_full_model=args.download_full_model,
        cache_dir=args.tokenizer_cache_dir,
    )

    template_str = getattr(tokenizer, "chat_template", "") or ""
    template_sha16 = hashlib.sha256(template_str.encode("utf-8")).hexdigest()[:16]
    template_mentions_enable_thinking = "enable_thinking" in template_str

    hello_messages = [{"role": "user", "content": args.hello_content}]

    # Experiment 1: apply_chat_template with enable_thinking=True
    exp1_text, exp1_err = _try_apply_chat_template(
        tokenizer, hello_messages, enable_thinking=True
    )

    # Experiment 2a: same, enable_thinking=False
    exp2a_text, exp2a_err = _try_apply_chat_template(
        tokenizer, hello_messages, enable_thinking=False
    )

    # Experiment 2b: same, enable_thinking omitted entirely (for older tokenizers)
    exp2b_text, exp2b_err = _try_apply_chat_template(
        tokenizer, hello_messages, enable_thinking=None
    )

    # Experiment 3: real public-test sample with harness guard
    sample_id, first_prompt = _read_first_prompt(args.sample_csv)
    user_content = first_prompt + HARNESS_GUARD
    real_messages = [{"role": "user", "content": user_content}]
    exp3_text, exp3_err = _try_apply_chat_template(
        tokenizer, real_messages, enable_thinking=True
    )

    def _pack(text: str | None, err: str | None) -> dict:
        if text is None:
            return {"ok": False, "error": err}
        return {
            "ok": True,
            "raw_string": text,
            "char_length": len(text),
            "token_length": _bpe_length(tokenizer, text),
            "ends_with_think_open": _ends_with(
                text, "<think>", allow_trailing_newline=True
            ),
            "ends_with_think_newline": text.endswith("<think>\n"),
            "contains_role_markers": any(
                marker in text for marker in ("<|im_start|>", "<|im_end|>", "<|begin")
            ),
        }

    exp1 = _pack(exp1_text, exp1_err)
    exp2a = _pack(exp2a_text, exp2a_err)
    exp2b = _pack(exp2b_text, exp2b_err)
    exp3 = _pack(exp3_text, exp3_err)

    # Derived conclusions
    enable_thinking_effective = (
        exp1.get("ok")
        and exp2a.get("ok")
        and exp1.get("raw_string") != exp2a.get("raw_string")
    )
    trace_as_thinking_viable = bool(
        exp1.get("ok") and exp1.get("ends_with_think_open")
    )
    first_sample_fits = bool(
        exp3.get("ok") and exp3.get("token_length", 10**9) <= PROMPT_TOKEN_BUDGET
    )
    headroom = (
        PROMPT_TOKEN_BUDGET - exp3.get("token_length", 0)
        if exp3.get("ok")
        else None
    )

    payload = {
        "model_handle": args.model,
        "download_mode": download_mode,
        "tokenizer_path": tokenizer_path,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_present": bool(template_str),
        "chat_template_sha16": template_sha16,
        "chat_template_mentions_enable_thinking": template_mentions_enable_thinking,
        "chat_template_length_chars": len(template_str),
        "experiments": {
            "exp1_hello_thinking_on": exp1,
            "exp2a_hello_thinking_off": exp2a,
            "exp2b_hello_thinking_omitted": exp2b,
            "exp3_real_sample_thinking_on": {
                **exp3,
                "sample_source_csv": str(args.sample_csv),
                "sample_id": sample_id,
                "user_content_char_length": len(user_content),
                "prompt_token_budget": PROMPT_TOKEN_BUDGET,
                "fits_in_budget": first_sample_fits,
                "headroom_tokens": headroom,
            },
        },
        "conclusions": {
            "enable_thinking_param_effective": enable_thinking_effective,
            "trace_as_thinking_viable": trace_as_thinking_viable,
            "first_public_sample_fits_budget": first_sample_fits,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Human-readable summary
    print("\n=== chat template probe summary ===")
    print(f"download mode         : {download_mode}")
    print(f"tokenizer class       : {type(tokenizer).__name__}")
    print(f"chat_template SHA16   : {template_sha16}")
    print(f"template length chars : {len(template_str)}")
    print(
        "template mentions enable_thinking: "
        f"{template_mentions_enable_thinking}"
    )
    print(
        "enable_thinking param effective  : "
        f"{enable_thinking_effective}"
    )
    print(
        "trace-as-thinking viable         : "
        f"{trace_as_thinking_viable}"
    )
    print(
        "first public sample BPE length   : "
        f"{exp3.get('token_length', 'ERR')}  "
        f"(budget={PROMPT_TOKEN_BUDGET}, headroom={headroom})"
    )
    if exp1.get("ok"):
        print("\n--- exp1 raw string (enable_thinking=True, 'HELLO' message) ---")
        print(repr(exp1["raw_string"]))
    else:
        print(f"\nexp1 FAILED: {exp1.get('error')}")
    if exp2a.get("ok"):
        print("\n--- exp2a raw string (enable_thinking=False, 'HELLO') ---")
        print(repr(exp2a["raw_string"]))
    else:
        print(f"\nexp2a FAILED: {exp2a.get('error')}")

    print(f"\nfull output: {args.output}")

    # Exit code reflects whether §6's three gating conditions are met.
    if not trace_as_thinking_viable:
        print(
            "\n[probe] WARNING: assistant seed does not end with <think>. "
            "docs/harness_alignment.md §4.2 'trace-as-thinking' plan may need revision."
        )
    if not first_sample_fits:
        print(
            "\n[probe] WARNING: first public sample already exceeds 512-token budget. "
            "Training-data prompts need stricter trimming than currently planned."
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
