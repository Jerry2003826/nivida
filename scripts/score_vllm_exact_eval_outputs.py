from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = Path("data/processed/local_eval_manifests")
AUTO_BASELINE_PRIORITY = (
    "adapter_stage2_thin_official_balanced_20260424_161110Z",
    "official_balanced_20260424_161110Z",
    "official_balanced",
    "b_thin",
)


def _parse_name_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected name=path, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"Empty name in {value!r}")
    return name, Path(path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_label_path(eval_name: str) -> Path:
    return DEFAULT_MANIFEST_DIR / f"{eval_name}.jsonl"


def _discover_raw_predictions(root: Path, eval_filter: set[str] | None) -> dict[str, dict[str, Path]]:
    discovered: dict[str, dict[str, Path]] = {}
    for raw_path in sorted(root.glob("*/*/raw/repeat_0.jsonl")):
        parts = raw_path.relative_to(root).parts
        if len(parts) < 4:
            continue
        eval_name, model = parts[0], parts[1]
        if eval_filter is not None and eval_name not in eval_filter:
            continue
        discovered.setdefault(eval_name, {})[model] = raw_path
    return discovered


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _select_baseline(reports: dict[str, Path], requested: str) -> tuple[str, str]:
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

    return next(iter(reports)), "auto first available fallback"


def _score_one(
    *,
    raw_path: Path,
    label_path: Path,
    output_dir: Path,
    prediction_key: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    _run(
        [
            sys.executable,
            "scripts/evaluate_predictions_exact.py",
            "--predictions",
            str(raw_path),
            "--labels",
            str(label_path),
            "--prediction-key",
            prediction_key,
            "--join",
            "id",
            "--output-json",
            str(report_path),
            "--output-csv",
            str(output_dir / "records.csv"),
            "--output-md",
            str(output_dir / "report.md"),
        ]
    )
    return report_path


def _rank_eval(
    *,
    eval_name: str,
    reports: dict[str, Path],
    baseline: str,
    output_root: Path,
    research_registry: Path | None,
) -> dict[str, str] | None:
    if not reports:
        return None
    baseline_name, baseline_reason = _select_baseline(reports, baseline)
    output_dir = output_root / eval_name
    rank_script = "scripts/rank_exact_eval_reports.py"
    cmd = [
        sys.executable,
        rank_script,
        "--baseline",
        baseline_name,
        "--output-json",
        str(output_dir / "ranking.json"),
        "--output-csv",
        str(output_dir / "ranking.csv"),
        "--output-md",
        str(output_dir / "ranking.md"),
    ]
    if research_registry is not None and research_registry.exists():
        rank_script = "scripts/rank_research_candidates.py"
        cmd = [
            sys.executable,
            rank_script,
            "--registry",
            str(research_registry),
            "--baseline",
            baseline_name,
            "--output-json",
            str(output_dir / "ranking.json"),
            "--output-csv",
            str(output_dir / "ranking.csv"),
            "--output-md",
            str(output_dir / "ranking.md"),
        ]
    for model, report_path in sorted(reports.items()):
        cmd.extend(["--report", f"{model}={report_path}"])
    _run(cmd)
    return {
        "baseline": baseline_name,
        "baseline_reason": baseline_reason,
        "rank_script": rank_script,
        "ranking": str(output_dir / "ranking.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score vLLM raw generation outputs with evaluate_predictions_exact.py and rank candidates."
    )
    parser.add_argument("--predictions-root", type=Path, default=Path("data/processed/vllm_exact_eval_v3"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/eval/vllm_exact_eval_v3"))
    parser.add_argument("--label", action="append", type=_parse_name_path, default=[])
    parser.add_argument("--eval-name", action="append", help="Optional eval set filter. Repeatable.")
    parser.add_argument(
        "--baseline",
        default="auto",
        help="Baseline model name, or 'auto' to prefer official-balanced final then b_thin.",
    )
    parser.add_argument(
        "--research-registry",
        type=Path,
        default=Path("configs/research_breakout_candidates.json"),
        help="Optional registry for research-arena ranking. Falls back to exact ranking if missing.",
    )
    parser.add_argument("--prediction-key", default="generation")
    args = parser.parse_args()

    eval_filter = None if not args.eval_name else set(args.eval_name)
    research_registry = (
        args.research_registry
        if args.research_registry.is_absolute()
        else REPO_ROOT / args.research_registry
    )
    label_paths = dict(args.label)
    discovered = _discover_raw_predictions(args.predictions_root, eval_filter)
    if not discovered:
        raise SystemExit(f"No vLLM repeat_0 raw predictions found under {args.predictions_root}")

    manifest: dict[str, dict[str, object]] = {"evals": {}}
    for eval_name, model_paths in sorted(discovered.items()):
        label_path = label_paths.get(eval_name, _default_label_path(eval_name))
        if not label_path.exists():
            raise SystemExit(
                f"Missing labels for eval {eval_name!r}: {label_path}. "
                f"Pass --label {eval_name}=path/to/labels.jsonl"
            )
        reports: dict[str, Path] = {}
        for model, raw_path in sorted(model_paths.items()):
            reports[model] = _score_one(
                raw_path=raw_path,
                label_path=label_path,
                output_dir=args.output_root / eval_name / model,
                prediction_key=args.prediction_key,
            )
        ranking_info = _rank_eval(
            eval_name=eval_name,
            reports=reports,
            baseline=args.baseline,
            output_root=args.output_root,
            research_registry=research_registry,
        )
        ranking_path = None if ranking_info is None else Path(ranking_info["ranking"])
        ranking = None if ranking_path is None else _load_json(ranking_path)
        submit_candidate = None
        if ranking:
            submit_candidate = next(
                (row["model"] for row in ranking["rows"] if row.get("submit_candidate")),
                None,
            )
        manifest["evals"][eval_name] = {
            "labels": str(label_path),
            "reports": {model: str(path) for model, path in sorted(reports.items())},
            "ranking": None if ranking_path is None else str(ranking_path),
            "baseline": None if ranking_info is None else ranking_info["baseline"],
            "baseline_reason": None if ranking_info is None else ranking_info["baseline_reason"],
            "rank_script": None if ranking_info is None else ranking_info["rank_script"],
            "submit_candidate": submit_candidate,
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "score_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_manifest": str(manifest_path), "evals": sorted(discovered)}, indent=2))


if __name__ == "__main__":
    main()
