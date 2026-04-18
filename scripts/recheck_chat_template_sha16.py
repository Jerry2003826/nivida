from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import read_json, write_json  # noqa: E402
from src.competition.harness_prompt import EXPECTED_CHAT_TEMPLATE_SHA16  # noqa: E402
from src.competition.tokenizer_probe_utils import (  # noqa: E402
    TOKENIZER_CACHE_DIR,
    load_probe_tokenizer,
)


def _load_probe_payload(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"probe payload must be a JSON object: {path}")
    return payload


def _compute_template_sha16(tokenizer) -> str:
    template = getattr(tokenizer, "chat_template", "") or ""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def recheck_chat_template_sha16(
    *,
    probe_json: str | Path,
    tokenizer_cache_dir: str | Path = TOKENIZER_CACHE_DIR,
) -> dict[str, Any]:
    probe_payload = _load_probe_payload(probe_json)
    model_handle = str(
        probe_payload.get(
            "model_handle",
            "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
        )
    )
    tokenizer_path = probe_payload.get("tokenizer_path")

    tokenizer = None
    actual_source_path: str | None = None
    if tokenizer_path:
        candidate = Path(str(tokenizer_path))
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if candidate.exists():
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(candidate),
                trust_remote_code=True,
                use_fast=False,
            )
            actual_source_path = str(candidate)

    if tokenizer is None:
        tokenizer, actual_source_path, _download_mode = load_probe_tokenizer(
            model_handle,
            download_full_model=False,
            cache_dir=Path(tokenizer_cache_dir),
        )

    current_sha16 = _compute_template_sha16(tokenizer)
    probe_sha16 = str(probe_payload.get("chat_template_sha16", ""))
    payload = {
        "model_handle": model_handle,
        "probe_json": str(probe_json),
        "tokenizer_path": actual_source_path,
        "expected_sha16": EXPECTED_CHAT_TEMPLATE_SHA16,
        "probe_sha16": probe_sha16,
        "current_sha16": current_sha16,
        "matches_expected_constant": current_sha16 == EXPECTED_CHAT_TEMPLATE_SHA16,
        "matches_probe_json": current_sha16 == probe_sha16,
    }
    if not payload["matches_expected_constant"] or not payload["matches_probe_json"]:
        raise SystemExit(
            "chat template SHA16 drift detected: "
            f"current={current_sha16}, probe={probe_sha16}, "
            f"expected={EXPECTED_CHAT_TEMPLATE_SHA16}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recheck the Nemotron chat template SHA16 before expensive training."
    )
    parser.add_argument("--probe-json", default="artifacts/chat_template_probe.json")
    parser.add_argument(
        "--tokenizer-cache-dir",
        default=str(TOKENIZER_CACHE_DIR),
        help="Fallback cache root when the probe's tokenizer_path is missing.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    payload = recheck_chat_template_sha16(
        probe_json=args.probe_json,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
    )
    if args.output:
        write_json(args.output, payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
