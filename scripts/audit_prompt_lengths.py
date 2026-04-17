#!/usr/bin/env python
"""Audit the real BPE length of training prompts against the harness budget.

Reads one or more SFT ``.jsonl`` files, tokenizes every ``prompt`` field with a
Nemotron-compatible tokenizer, and writes a structured report to
``artifacts/prompt_length_audit.json`` (by default).

Budget comes from the official evaluator:

    max_model_len     = 4096
    max_tokens (gen)  = 3584
    prompt budget     = max_model_len - max_tokens = 512

The audit flags every sample whose prompt BPE length exceeds ``--budget``
(default 512) so those samples can be filtered upstream rather than truncated
at inference time.

Tokenizer source:

- Default: ``artifacts/_tokenizer_cache/<handle>`` (populated by
  ``scripts/probe_chat_template.py`` in tokenizer-only mode).
- Override with ``--tokenizer-path``.
- Or pass ``--download`` to force a fresh tokenizer-only download.

Example:

    python scripts/audit_prompt_lengths.py \\
        data/processed/stage2_distill_train.jsonl \\
        data/processed/stage3_repair_train.jsonl

    python scripts/audit_prompt_lengths.py \\
        data/processed/stage2_distill_train.jsonl \\
        --tokenizer-path artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default \\
        --budget 512 \\
        --output artifacts/prompt_length_audit.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

# The repo layout puts both this script and the sources at the same level;
# make sure 'src' is importable when this script is run directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.common.io import load_jsonl, write_json  # noqa: E402


_DEFAULT_CACHE = Path("artifacts/_tokenizer_cache")
_DEFAULT_MODEL_HANDLE = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
_DEFAULT_BUDGET = 512


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = max(0, math.ceil(pct * len(ordered)) - 1)
    return ordered[idx]


def _load_tokenizer(tokenizer_path: Path, download: bool, model_handle: str) -> Any:
    """Load an AutoTokenizer from disk, or download it in tokenizer-only mode."""
    if download or not tokenizer_path.exists() or not any(tokenizer_path.iterdir()):
        try:
            from scripts.probe_chat_template import _download_tokenizer_only, _import_kagglehub
        except ImportError as exc:
            raise SystemExit(
                f"Unable to import tokenizer-only downloader: {exc}. "
                "Ensure scripts/probe_chat_template.py is present."
            ) from exc
        kagglehub = _import_kagglehub()
        cache_parent = tokenizer_path.parent
        cache_parent.mkdir(parents=True, exist_ok=True)
        tokenizer_path = _download_tokenizer_only(
            kagglehub, model_handle, cache_parent
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers not installed. Install: pip install transformers"
        ) from exc

    try:
        return AutoTokenizer.from_pretrained(
            str(tokenizer_path), trust_remote_code=True
        )
    except Exception as exc_fast:  # noqa: BLE001
        print(
            f"[audit] fast tokenizer load failed ({type(exc_fast).__name__}); "
            "retrying with use_fast=False",
            flush=True,
        )
        return AutoTokenizer.from_pretrained(
            str(tokenizer_path), trust_remote_code=True, use_fast=False
        )


def audit_jsonl(
    dataset_path: Path,
    tokenizer: Any,
    *,
    budget: int,
    max_over_budget_samples: int,
) -> dict[str, Any]:
    """Return a summary dict for a single SFT jsonl file."""
    records = load_jsonl(dataset_path)
    lengths: list[int] = []
    over_budget: list[dict[str, Any]] = []
    missing_prompt = 0

    for row in records:
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            missing_prompt += 1
            continue
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        length = len(token_ids)
        lengths.append(length)
        if length > budget:
            if len(over_budget) < max_over_budget_samples:
                over_budget.append(
                    {
                        "id": row.get("id"),
                        "token_length": length,
                        "official_family": row.get("official_family"),
                        "subtype": row.get("subtype"),
                    }
                )

    count = len(lengths)
    return {
        "dataset_path": str(dataset_path),
        "num_records": len(records),
        "num_prompts_tokenized": count,
        "num_missing_prompts": missing_prompt,
        "budget": budget,
        "num_over_budget": sum(1 for length in lengths if length > budget),
        "p50": _percentile(lengths, 0.50),
        "p95": _percentile(lengths, 0.95),
        "p99": _percentile(lengths, 0.99),
        "p100": max(lengths) if lengths else 0,
        "mean": float(statistics.fmean(lengths)) if lengths else 0.0,
        "fits_budget": bool(lengths) and max(lengths) <= budget,
        "sample_over_budget_records": over_budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit real BPE prompt lengths in SFT jsonl files "
            "against the harness max_model_len - max_tokens budget."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        type=Path,
        help="One or more SFT jsonl files (each row must have a 'prompt' field).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=_DEFAULT_CACHE
        / _DEFAULT_MODEL_HANDLE.replace("/", "_"),
        help=(
            "Local directory containing the Nemotron tokenizer. Default is the "
            "cache populated by scripts/probe_chat_template.py."
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Force a fresh tokenizer-only download via kagglehub.",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL_HANDLE,
        help="Kaggle model handle for the tokenizer. Only used with --download.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=_DEFAULT_BUDGET,
        help=(
            "Prompt BPE token budget (default 512 = max_model_len 4096 - "
            "max_tokens 3584)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/prompt_length_audit.json"),
    )
    parser.add_argument(
        "--max-over-budget-samples",
        type=int,
        default=20,
        help="How many over-budget records to retain per dataset for inspection.",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        if not dataset.exists():
            raise SystemExit(f"dataset not found: {dataset}")

    tokenizer = _load_tokenizer(args.tokenizer_path, args.download, args.model)

    reports = [
        audit_jsonl(
            dataset,
            tokenizer,
            budget=args.budget,
            max_over_budget_samples=args.max_over_budget_samples,
        )
        for dataset in args.datasets
    ]

    payload = {
        "budget": args.budget,
        "tokenizer_path": str(args.tokenizer_path),
        "tokenizer_class": type(tokenizer).__name__,
        "datasets": reports,
        "overall_fits_budget": all(r["fits_budget"] for r in reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)

    print("\n=== prompt length audit ===")
    print(f"tokenizer class       : {type(tokenizer).__name__}")
    print(f"budget (prompt tokens): {args.budget}\n")
    header = f"{'dataset':<60} {'N':>7} {'p50':>6} {'p95':>6} {'p99':>6} {'p100':>6} {'over':>6}"
    print(header)
    print("-" * len(header))
    for report in reports:
        name = Path(report["dataset_path"]).name
        print(
            f"{name:<60} {report['num_prompts_tokenized']:>7} "
            f"{report['p50']:>6} {report['p95']:>6} {report['p99']:>6} "
            f"{report['p100']:>6} {report['num_over_budget']:>6}"
        )
    print(f"\nfull output: {args.output}")
    if not payload["overall_fits_budget"]:
        print(
            "\n[audit] WARNING: one or more datasets contain prompts exceeding the budget. "
            "These samples must be filtered (not truncated) before training."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
