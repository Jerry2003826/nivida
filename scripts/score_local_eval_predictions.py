from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTO_BASELINE_PRIORITY = (
    "adapter_stage2_thin_official_balanced_20260424_161110Z",
    "official_balanced_20260424_161110Z",
    "official_balanced",
    "b_thin",
)


def _prediction_model_name(path: Path) -> str:
    stem = path.stem
    suffix = "_predictions"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def _discover_evals(pred_dir: Path) -> list[str]:
    if not pred_dir.exists():
        return []
    evals = [path.name for path in pred_dir.iterdir() if path.is_dir()]
    if any(pred_dir.glob("*_predictions.jsonl")):
        evals.append(".")
    return sorted(evals)


def _prediction_files(pred_dir: Path, eval_name: str) -> list[Path]:
    target = pred_dir if eval_name == "." else pred_dir / eval_name
    return sorted(target.glob("*_predictions.jsonl"))


def _labels_path(manifest_dir: Path, eval_name: str) -> Path:
    if eval_name == ".":
        raise ValueError("Cannot infer labels for flat prediction directory; pass named eval subdirectories.")
    return manifest_dir / f"{eval_name}.jsonl"


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _select_baseline(reports: dict[str, str], requested: str) -> tuple[str, str]:
    if requested != "auto":
        if requested not in reports:
            raise ValueError(
                f"Requested baseline {requested!r} not found. Available models: {', '.join(sorted(reports))}"
            )
        return requested, "explicit"

    for preferred in AUTO_BASELINE_PRIORITY:
        if preferred in reports:
            return preferred, f"auto exact match: {preferred}"

    official_final = sorted(
        model
        for model in reports
        if "official_balanced" in model and "_ckpt_" not in model
    )
    if official_final:
        return official_final[-1], "auto official-balanced final adapter"

    official_any = sorted(model for model in reports if "official_balanced" in model)
    if official_any:
        return official_any[-1], "auto official-balanced checkpoint fallback"

    if "b_thin" in reports:
        return "b_thin", "auto b_thin fallback"

    raise ValueError(
        "Could not auto-select a baseline. Re-run with --baseline MODEL. "
        f"Available models: {', '.join(sorted(reports))}"
    )


def _score_eval(
    *,
    eval_name: str,
    pred_dir: Path,
    manifest_dir: Path,
    output_dir: Path,
    join: str,
    prediction_key: str | None,
) -> list[dict[str, Any]]:
    labels = _labels_path(manifest_dir, eval_name)
    if not labels.exists():
        raise FileNotFoundError(f"Missing labels manifest for {eval_name}: {labels}")
    predictions = _prediction_files(pred_dir, eval_name)
    if not predictions:
        return []

    eval_out_dir = output_dir / eval_name
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    evaluator = REPO_ROOT / "scripts" / "evaluate_predictions_exact.py"
    for prediction_path in predictions:
        model = _prediction_model_name(prediction_path)
        report_json = eval_out_dir / f"{model}_exact_report.json"
        records_csv = eval_out_dir / f"{model}_exact_records.csv"
        report_md = eval_out_dir / f"{model}_exact_report.md"
        cmd = [
            sys.executable,
            str(evaluator),
            "--predictions",
            str(prediction_path),
            "--labels",
            str(labels),
            "--join",
            join,
            "--output-json",
            str(report_json),
            "--output-csv",
            str(records_csv),
            "--output-md",
            str(report_md),
        ]
        if prediction_key:
            cmd.extend(["--prediction-key", prediction_key])
        _run(cmd)
        rows.append(
            {
                "eval": eval_name,
                "model": model,
                "predictions": str(prediction_path),
                "report_json": str(report_json),
                "records_csv": str(records_csv),
                "report_md": str(report_md),
            }
        )
    return rows


def _rank_eval(
    *,
    eval_name: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    baseline: str,
    min_overall_delta: float,
    max_family_regression_samples: float,
    large_family_min_n: int,
) -> dict[str, str] | None:
    eval_rows = [row for row in rows if row["eval"] == eval_name]
    if len(eval_rows) < 2:
        return None
    reports = {row["model"]: row["report_json"] for row in eval_rows}
    baseline_name, baseline_reason = _select_baseline(reports, baseline)
    ranker = REPO_ROOT / "scripts" / "rank_exact_eval_reports.py"
    output_json = output_dir / eval_name / "ranking.json"
    output_csv = output_dir / eval_name / "ranking.csv"
    output_md = output_dir / eval_name / "ranking.md"
    cmd = [
        sys.executable,
        str(ranker),
        "--baseline",
        baseline_name,
        "--output-json",
        str(output_json),
        "--output-csv",
        str(output_csv),
        "--output-md",
        str(output_md),
        "--min-overall-delta",
        str(min_overall_delta),
        "--max-family-regression-samples",
        str(max_family_regression_samples),
        "--large-family-min-n",
        str(large_family_min_n),
    ]
    for model, report in reports.items():
        cmd.extend(["--report", f"{model}={report}"])
    _run(cmd)
    return {
        "eval": eval_name,
        "baseline": baseline_name,
        "baseline_reason": baseline_reason,
        "ranking_json": str(output_json),
        "ranking_csv": str(output_csv),
        "ranking_md": str(output_md),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch score cloud-generated local-eval predictions and emit candidate rankings."
    )
    parser.add_argument("--pred-dir", type=Path, default=Path("data/processed/local_eval_predictions_v3"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/processed/local_eval_manifests"))
    parser.add_argument("--eval", action="append", dest="evals")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/eval/exact_reports_v3"))
    parser.add_argument("--join", choices=["id", "order"], default="id")
    parser.add_argument("--prediction-key")
    parser.add_argument(
        "--baseline",
        default="auto",
        help="Baseline model name, or 'auto' to prefer official-balanced final then b_thin.",
    )
    parser.add_argument("--rank-eval", action="append")
    parser.add_argument("--skip-ranking", action="store_true")
    parser.add_argument("--min-overall-delta", type=float, default=0.0)
    parser.add_argument("--max-family-regression-samples", type=float, default=1.0)
    parser.add_argument("--large-family-min-n", type=int, default=24)
    args = parser.parse_args()

    evals = args.evals or _discover_evals(args.pred_dir)
    if not evals:
        raise SystemExit(f"No prediction eval directories found under {args.pred_dir}")

    scored_rows: list[dict[str, Any]] = []
    for eval_name in evals:
        scored_rows.extend(
            _score_eval(
                eval_name=eval_name,
                pred_dir=args.pred_dir,
                manifest_dir=args.manifest_dir,
                output_dir=args.output_dir,
                join=args.join,
                prediction_key=args.prediction_key,
            )
        )

    rankings: list[dict[str, str]] = []
    if not args.skip_ranking:
        for eval_name in args.rank_eval or ["combined_balanced_48pf"]:
            ranking = _rank_eval(
                eval_name=eval_name,
                rows=scored_rows,
                output_dir=args.output_dir,
                baseline=args.baseline,
                min_overall_delta=args.min_overall_delta,
                max_family_regression_samples=args.max_family_regression_samples,
                large_family_min_n=args.large_family_min_n,
            )
            if ranking is not None:
                rankings.append(ranking)

    summary = {
        "pred_dir": str(args.pred_dir),
        "manifest_dir": str(args.manifest_dir),
        "output_dir": str(args.output_dir),
        "num_reports": len(scored_rows),
        "reports": scored_rows,
        "rankings": rankings,
    }
    summary_path = args.output_dir / "score_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "num_reports": len(scored_rows)}, indent=2))


if __name__ == "__main__":
    main()
