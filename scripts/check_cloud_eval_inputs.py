from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("data/processed/cloud_eval_preflight.json")
WEIGHT_FILENAMES = ("adapter_model.safetensors", "adapter_model.bin")
REQUIRED_REPO_FILES = (
    Path("scripts/check_cloud_vllm_env.sh"),
    Path("scripts/eval_official_vllm_proxy.py"),
    Path("scripts/run_cloud_vllm_exact_eval_v3.sh"),
    Path("scripts/score_vllm_exact_eval_outputs.py"),
    Path("scripts/write_cloud_artifact_manifest.py"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _git_value(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"candidate must be name=path, got {value!r}")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise ValueError(f"candidate must be name=path, got {value!r}")
    return name, Path(raw_path)


def _resolve_eval_input(item: str, *, repo_root: Path) -> Path:
    direct = Path(item)
    if direct.is_absolute():
        return direct
    if (repo_root / direct).is_file():
        return repo_root / direct
    return repo_root / "data" / "processed" / "local_eval_manifests" / f"{item}.jsonl"


def _jsonl_summary(path: Path) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    ids: set[str] = set()
    duplicates: set[str] = set()
    rows = 0
    missing_target = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            rows += 1
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                failures.append(f"invalid JSONL at line {line_number}: {exc}")
                continue
            row_id = str(row.get("id", ""))
            if not row_id:
                failures.append(f"missing id at line {line_number}")
            elif row_id in ids:
                duplicates.add(row_id)
            else:
                ids.add(row_id)
            if "target_answer" not in row:
                missing_target += 1
    if rows <= 0:
        failures.append("JSONL has no rows")
    if duplicates:
        failures.append(f"duplicate ids: {sorted(duplicates)[:10]}")
    if missing_target:
        failures.append(f"rows missing target_answer: {missing_target}")
    return (
        {
            "rows": rows,
            "duplicate_ids": sorted(duplicates)[:20],
            "missing_target_answer_rows": missing_target,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        },
        failures,
    )


def _candidate_summary(raw: str, *, repo_root: Path) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    name, raw_path = _parse_candidate(raw)
    path = raw_path if raw_path.is_absolute() else repo_root / raw_path
    weights = [path / filename for filename in WEIGHT_FILENAMES if (path / filename).is_file()]
    config = path / "adapter_config.json"
    if not path.is_dir():
        failures.append("adapter directory missing")
    if not weights:
        failures.append("adapter weights missing")
    if not config.is_file():
        failures.append("adapter_config.json missing")
    return (
        {
            "name": name,
            "path": _rel(path, repo_root=repo_root),
            "exists": path.is_dir(),
            "adapter_config": _rel(config, repo_root=repo_root),
            "has_adapter_config": config.is_file(),
            "weights": [
                {
                    "path": _rel(weight, repo_root=repo_root),
                    "size_bytes": weight.stat().st_size,
                    "sha256": _sha256(weight),
                }
                for weight in weights
            ],
        },
        failures,
    )


def run_checks(
    *,
    repo_root: str | Path = REPO_ROOT,
    eval_inputs: list[str],
    candidates: list[str],
    config: str = "configs/train_stage2_official_balanced_answer_only.yaml",
    output_path: str | Path = DEFAULT_OUTPUT,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    failures: list[dict[str, Any]] = []

    report: dict[str, Any] = {
        "status": "dry_run" if dry_run else "pass",
        "ready_for_vllm": False if dry_run else True,
        "dry_run": dry_run,
        "timestamp_utc": _utc_now(),
        "git": {
            "branch": _git_value(root, "branch", "--show-current"),
            "commit": _git_value(root, "rev-parse", "HEAD"),
        },
        "config": config,
        "eval_inputs": [],
        "candidates": [],
        "repo_files": [],
        "failures": failures,
    }

    if dry_run:
        report["planned_eval_inputs"] = eval_inputs
        report["planned_candidates"] = candidates
        _write_json(output, report)
        return report

    config_path = root / config
    if not config_path.is_file():
        failures.append({"kind": "config", "path": config, "reason": "missing config"})
    else:
        report["config_sha256"] = _sha256(config_path)

    for repo_file in REQUIRED_REPO_FILES:
        path = root / repo_file
        item = {"path": repo_file.as_posix(), "exists": path.is_file()}
        if path.is_file():
            item["sha256"] = _sha256(path)
        else:
            failures.append({"kind": "repo_file", "path": repo_file.as_posix(), "reason": "missing"})
        report["repo_files"].append(item)

    for item in eval_inputs:
        path = _resolve_eval_input(item, repo_root=root)
        payload: dict[str, Any] = {
            "name": item,
            "path": _rel(path, repo_root=root),
            "exists": path.is_file(),
        }
        if path.is_file():
            summary, item_failures = _jsonl_summary(path)
            payload.update(summary)
            for reason in item_failures:
                failures.append({"kind": "eval_input", "name": item, "path": payload["path"], "reason": reason})
        else:
            failures.append({"kind": "eval_input", "name": item, "path": payload["path"], "reason": "missing"})
        report["eval_inputs"].append(payload)

    seen_names: set[str] = set()
    for raw in candidates:
        try:
            payload, candidate_failures = _candidate_summary(raw, repo_root=root)
        except ValueError as exc:
            failures.append({"kind": "candidate", "raw": raw, "reason": str(exc)})
            continue
        if payload["name"] in seen_names:
            candidate_failures.append("duplicate candidate name")
        seen_names.add(payload["name"])
        for reason in candidate_failures:
            failures.append(
                {
                    "kind": "candidate",
                    "name": payload["name"],
                    "path": payload["path"],
                    "reason": reason,
                }
            )
        report["candidates"].append(payload)

    if not candidates:
        failures.append({"kind": "candidate", "reason": "no candidates supplied"})
    if failures:
        report["status"] = "fail"
        report["ready_for_vllm"] = False

    _write_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU-only preflight for cloud exact-eval inputs and adapters.")
    parser.add_argument("--eval-inputs", default="smoke_6pf", help="Comma-separated manifest names or JSONL paths.")
    parser.add_argument("--candidate", action="append", default=[], help="Candidate adapter as name=path.")
    parser.add_argument("--config", default="configs/train_stage2_official_balanced_answer_only.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    report = run_checks(
        eval_inputs=_split_csv(args.eval_inputs),
        candidates=list(args.candidate),
        config=args.config,
        output_path=args.output,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
