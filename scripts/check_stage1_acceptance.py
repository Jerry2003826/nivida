from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import read_json, write_json  # noqa: E402
from src.competition.harness_prompt import EXPECTED_CHAT_TEMPLATE_SHA16  # noqa: E402


REQUIRED_STAGE1_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "training_metadata.json",
    "last_run_summary.json",
)
PROGRESS_RE = re.compile(r"(?P<current>\d+)\s*/\s*(?P<total>\d+)")
CHECKPOINT_RE = re.compile(r"checkpoint-(?P<step>\d+)")


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def parse_stage1_log(log_path: str | Path) -> dict[str, Any]:
    source = Path(log_path)
    if not source.is_file():
        raise SystemExit(f"log file not found: {source}")

    last_progress: dict[str, int] | None = None
    last_checkpoint: str | None = None
    last_eval_line: str | None = None
    line_count = 0

    for raw_line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        line_count += 1
        for match in PROGRESS_RE.finditer(raw_line):
            current = int(match.group("current"))
            total = int(match.group("total"))
            if total > 0 and current <= total:
                last_progress = {"current": current, "total": total}
        checkpoint_match = CHECKPOINT_RE.search(raw_line)
        if checkpoint_match:
            last_checkpoint = f"checkpoint-{checkpoint_match.group('step')}"
        if "eval_" in raw_line or "eval runtime" in raw_line.lower():
            last_eval_line = raw_line.strip()

    return {
        "path": str(source),
        "num_lines": line_count,
        "last_progress": last_progress,
        "last_checkpoint": last_checkpoint,
        "last_eval_line": last_eval_line,
    }


def check_stage1_acceptance(
    *,
    adapter_dir: str | Path,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    adapter_path = Path(adapter_dir)
    if not adapter_path.is_dir():
        raise SystemExit(f"adapter_dir not found: {adapter_path}")

    file_paths = {name: adapter_path / name for name in REQUIRED_STAGE1_FILES}
    missing = [name for name, path in file_paths.items() if not path.is_file()]
    if missing:
        raise SystemExit(
            "stage1 acceptance missing required file(s): " + ", ".join(sorted(missing))
        )
    if file_paths["adapter_model.safetensors"].stat().st_size <= 0:
        raise SystemExit("adapter_model.safetensors must be non-empty")

    metadata = _load_json_object(
        file_paths["training_metadata.json"], label="training_metadata.json"
    )
    summary = _load_json_object(
        file_paths["last_run_summary.json"], label="last_run_summary.json"
    )
    adapter_config = _load_json_object(
        file_paths["adapter_config.json"], label="adapter_config.json"
    )

    try:
        adapter_rank = int(adapter_config.get("r", adapter_config.get("rank", -1)))
    except (TypeError, ValueError) as exc:
        raise SystemExit("adapter rank missing or not an integer") from exc
    if adapter_rank <= 0 or adapter_rank > 32:
        raise SystemExit(f"adapter rank must be in 1..32, got {adapter_rank}")

    target_modules = adapter_config.get("target_modules")
    if not target_modules:
        raise SystemExit("adapter_config.target_modules must be non-empty")

    preflight = _coalesce(metadata.get("preflight"), summary.get("preflight"))
    if not isinstance(preflight, dict):
        raise SystemExit("missing preflight report in training_metadata/last_run_summary")
    if preflight.get("status") != "ok":
        raise SystemExit(
            f"preflight.status must be 'ok', got {preflight.get('status')!r}"
        )
    chat_template_sha16 = preflight.get("chat_template_sha16")
    if chat_template_sha16 != EXPECTED_CHAT_TEMPLATE_SHA16:
        raise SystemExit(
            "chat_template_sha16 mismatch: "
            f"got {chat_template_sha16!r}, expected {EXPECTED_CHAT_TEMPLATE_SHA16!r}. "
            "Run scripts/probe_chat_template.py and ensure tokenizer_path matches."
        )

    dataset_stats = _coalesce(metadata.get("dataset_stats"), summary.get("dataset_stats"))
    if not isinstance(dataset_stats, dict):
        raise SystemExit("missing dataset_stats in training_metadata/last_run_summary")
    if dataset_stats.get("length_unit") != "bpe_tokens":
        raise SystemExit(
            "dataset_stats.length_unit must be 'bpe_tokens', "
            f"got {dataset_stats.get('length_unit')!r}"
        )

    num_matched_target_modules = _coalesce(
        metadata.get("num_matched_target_modules"),
        summary.get("num_matched_target_modules"),
    )
    try:
        num_matched_target_modules = int(num_matched_target_modules)
    except (TypeError, ValueError) as exc:
        raise SystemExit("num_matched_target_modules missing or not an integer") from exc
    if num_matched_target_modules <= 0:
        raise SystemExit(
            f"num_matched_target_modules must be > 0, got {num_matched_target_modules}"
        )

    num_train_records = _coalesce(
        metadata.get("num_train_records"),
        summary.get("num_train_records"),
    )
    try:
        num_train_records = int(num_train_records)
    except (TypeError, ValueError) as exc:
        raise SystemExit("num_train_records missing or not an integer") from exc
    if num_train_records <= 0:
        raise SystemExit(f"num_train_records must be > 0, got {num_train_records}")

    payload: dict[str, Any] = {
        "accepted": True,
        "adapter_dir": str(adapter_path),
        "required_files": {name: str(path) for name, path in file_paths.items()},
        "adapter_rank": adapter_rank,
        "target_modules": target_modules,
        "preflight_status": preflight["status"],
        "chat_template_sha16": chat_template_sha16,
        "expected_chat_template_sha16": EXPECTED_CHAT_TEMPLATE_SHA16,
        "dataset_length_unit": dataset_stats["length_unit"],
        "num_matched_target_modules": num_matched_target_modules,
        "num_train_records": num_train_records,
    }

    if log_path is not None:
        log_summary = parse_stage1_log(log_path)
        if log_summary["last_progress"] is None:
            raise SystemExit(
                "stage1 acceptance log check failed: no training progress line found"
            )
        payload["log"] = log_summary

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a finished stage1 adapter.")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--log-path")
    parser.add_argument("--output")
    args = parser.parse_args()

    payload = check_stage1_acceptance(
        adapter_dir=args.adapter_dir,
        log_path=args.log_path,
    )
    if args.output:
        write_json(args.output, payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
