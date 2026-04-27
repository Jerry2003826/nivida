from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rank_exact_eval_reports import _build_rows, _load_json, _parse_report_arg  # noqa: E402
from src.common.io import write_json  # noqa: E402
from src.research.candidate_registry import (  # noqa: E402
    DEFAULT_BASELINE_NAME,
    candidate_by_name,
    load_registry,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _select_baseline(reports: dict[str, Any], requested: str, registry_baseline: str) -> tuple[str, str]:
    if requested != "auto":
        if requested not in reports:
            raise SystemExit(f"Baseline {requested!r} not found. Available: {', '.join(sorted(reports))}")
        return requested, "explicit"
    for name in (registry_baseline, DEFAULT_BASELINE_NAME):
        if name in reports:
            return name, f"auto registry baseline: {name}"
    official = sorted(name for name in reports if "official_balanced" in name and "_ckpt_" not in name)
    if official:
        return official[-1], "auto official-balanced fallback"
    raise SystemExit("baseline=auto requires an official_balanced report in research mode")


def _enrich_rows(rows: list[dict[str, Any]], registry: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = candidate_by_name(registry)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        metadata = candidates.get(str(row["model"]), {})
        candidate_type = str(metadata.get("type", "unregistered"))
        submission_safe = bool(metadata.get("submission_safe", True))
        pass_gate = bool(row["pass_gate"]) and submission_safe
        gate_reason = row["gate_reason"]
        if row["pass_gate"] and not submission_safe:
            gate_reason = "candidate is marked submission_unsafe"
        enriched.append(
            {
                **row,
                "pass_gate": pass_gate,
                "submit_candidate": False,
                "candidate_type": candidate_type,
                "prompt_profile": metadata.get("prompt_profile", ""),
                "data_recipe": metadata.get("data_recipe", ""),
                "family_focus": ",".join(metadata.get("family_focus", [])),
                "gpu_required": metadata.get("gpu_required", ""),
                "submission_safe": submission_safe,
                "gate_reason": gate_reason,
            }
        )
    submit_marked = False
    for row in enriched:
        if not submit_marked and row["pass_gate"]:
            row["submit_candidate"] = True
            submit_marked = True
    return enriched


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Research Arena Candidate Ranking",
        "",
        f"- baseline: `{payload['baseline']}`",
        f"- baseline reason: `{payload['baseline_reason']}`",
        f"- public LB baseline: `{payload['baseline_public_score']}`",
        "",
        "| rank | model | type | submit | gate | official_verify | delta | boxed_valid | family_focus | prompt | data | safe |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {rank} | {model} | {candidate_type} | {submit_candidate} | {gate_reason} | "
            "{official_verify_accuracy:.4f} | {delta_vs_baseline:.4f} | "
            "{boxed_valid_rate:.4f} | {family_focus} | {prompt_profile} | "
            "{data_recipe} | {submission_safe} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank mixed research candidates with registry metadata.")
    parser.add_argument("--registry", type=Path, default=Path("configs/research_breakout_candidates.json"))
    parser.add_argument("--report", action="append", required=True, help="Candidate report as name=path.json")
    parser.add_argument("--baseline", default="auto")
    parser.add_argument("--output-json", type=Path, default=Path("data/processed/eval/research_arena_ranking.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/eval/research_arena_ranking.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("LOCAL_RESEARCH_ARENA_RANKING.md"))
    parser.add_argument("--min-overall-delta", type=float, default=0.0)
    parser.add_argument("--max-family-regression-samples", type=float, default=1.0)
    parser.add_argument("--large-family-min-n", type=int, default=24)
    args = parser.parse_args(argv)

    registry_path = args.registry if args.registry.is_absolute() else REPO_ROOT / args.registry
    registry = load_registry(registry_path)
    reports: dict[str, dict[str, Any]] = {}
    report_paths: dict[str, str] = {}
    for value in args.report:
        name, path = _parse_report_arg(value)
        reports[name] = _load_json(path)
        report_paths[name] = str(path)
    baseline, baseline_reason = _select_baseline(reports, args.baseline, str(registry["baseline"]))
    rows = _build_rows(
        reports,
        baseline_name=baseline,
        min_overall_delta=float(args.min_overall_delta),
        max_family_regression_samples=float(args.max_family_regression_samples),
        large_family_min_n=int(args.large_family_min_n),
    )
    rows = _enrich_rows(rows, registry)
    payload = {
        "registry": str(args.registry),
        "baseline": baseline,
        "baseline_reason": baseline_reason,
        "baseline_public_score": registry.get("baseline_public_score"),
        "report_paths": report_paths,
        "rows": rows,
    }
    write_json(args.output_json, payload)
    _write_csv(args.output_csv, rows)
    _write_markdown(args.output_md, payload)
    submit = next((row for row in rows if row["submit_candidate"]), None)
    print(json.dumps({"output_json": str(args.output_json), "submit_candidate": None if submit is None else submit["model"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

