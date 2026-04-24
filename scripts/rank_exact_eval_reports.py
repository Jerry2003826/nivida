from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import pstdev
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_report_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty report name in {value!r}")
        return name, Path(path)
    path = Path(value)
    return path.stem, path


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _overall(report: dict[str, Any], key: str, default: float = 0.0) -> float:
    return _float(dict(report.get("overall", {})).get(key), default)


def _family_rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    family = report.get("family", {})
    if not isinstance(family, dict):
        return {}
    return {str(name): dict(row) for name, row in family.items() if isinstance(row, dict)}


def _subtype_rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    subtype = report.get("subtype_official_verify_accuracy", {})
    if not isinstance(subtype, dict):
        return {}
    return {str(name): dict(row) for name, row in subtype.items() if isinstance(row, dict)}


def _family_accuracy_std(report: dict[str, Any]) -> float:
    values = [
        _float(row.get("official_verify_accuracy"))
        for row in _family_rows(report).values()
        if int(row.get("n", 0) or 0) > 0
    ]
    return pstdev(values) if len(values) > 1 else 0.0


def _subtype_accuracy_std(report: dict[str, Any]) -> float:
    values = [
        _float(row.get("accuracy"))
        for row in _subtype_rows(report).values()
        if int(row.get("n", 0) or 0) > 0
    ]
    return pstdev(values) if len(values) > 1 else 0.0


def _worst_family_delta(
    report: dict[str, Any],
    baseline: dict[str, Any],
    *,
    large_family_min_n: int,
) -> tuple[str, float, float]:
    report_families = _family_rows(report)
    baseline_families = _family_rows(baseline)
    worst_family = ""
    worst_delta_samples = 0.0
    worst_delta_accuracy = 0.0
    first = True
    for family, baseline_row in baseline_families.items():
        report_row = report_families.get(family)
        if report_row is None:
            continue
        baseline_n = int(baseline_row.get("n", 0) or 0)
        report_n = int(report_row.get("n", 0) or 0)
        n = min(baseline_n, report_n)
        if n < large_family_min_n:
            continue
        baseline_acc = _float(baseline_row.get("official_verify_accuracy"))
        report_acc = _float(report_row.get("official_verify_accuracy"))
        delta_accuracy = report_acc - baseline_acc
        delta_samples = delta_accuracy * n
        if first or delta_samples < worst_delta_samples:
            worst_family = family
            worst_delta_samples = delta_samples
            worst_delta_accuracy = delta_accuracy
            first = False
    return worst_family, worst_delta_samples, worst_delta_accuracy


def _worst_subtype_delta(
    report: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[str, float, float]:
    report_subtypes = _subtype_rows(report)
    baseline_subtypes = _subtype_rows(baseline)
    worst_subtype = ""
    worst_delta_samples = 0.0
    worst_delta_accuracy = 0.0
    first = True
    for subtype, baseline_row in baseline_subtypes.items():
        report_row = report_subtypes.get(subtype)
        if report_row is None:
            continue
        baseline_n = int(baseline_row.get("n", 0) or 0)
        report_n = int(report_row.get("n", 0) or 0)
        n = min(baseline_n, report_n)
        if n <= 0:
            continue
        baseline_acc = _float(baseline_row.get("accuracy"))
        report_acc = _float(report_row.get("accuracy"))
        delta_accuracy = report_acc - baseline_acc
        delta_samples = delta_accuracy * n
        if first or delta_samples < worst_delta_samples:
            worst_subtype = subtype
            worst_delta_samples = delta_samples
            worst_delta_accuracy = delta_accuracy
            first = False
    return worst_subtype, worst_delta_samples, worst_delta_accuracy


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, str]:
    return (
        -_float(row["official_verify_accuracy"]),
        -_float(row["boxed_valid_rate"]),
        _float(row["avg_prediction_words"]),
        _float(row["family_accuracy_std"]),
        _float(row["subtype_accuracy_std"]),
        str(row["model"]),
    )


def _gate_reason(
    *,
    model: str,
    baseline_name: str,
    overall_delta: float,
    min_overall_delta: float,
    worst_family: str,
    worst_family_delta_samples: float,
    max_family_regression_samples: float,
) -> tuple[bool, str]:
    if model == baseline_name:
        return False, "baseline"
    if overall_delta <= min_overall_delta:
        return (
            False,
            f"overall_delta {overall_delta:.4f} <= required {min_overall_delta:.4f}",
        )
    if worst_family and worst_family_delta_samples < -max_family_regression_samples:
        return (
            False,
            f"{worst_family} regressed {worst_family_delta_samples:.2f} samples",
        )
    return True, "passes overall and family-regression gates"


def _build_rows(
    reports: dict[str, dict[str, Any]],
    *,
    baseline_name: str,
    min_overall_delta: float,
    max_family_regression_samples: float,
    large_family_min_n: int,
) -> list[dict[str, Any]]:
    baseline = reports[baseline_name]
    baseline_overall = _overall(baseline, "official_verify_accuracy")
    rows: list[dict[str, Any]] = []
    for model, report in reports.items():
        official_acc = _overall(report, "official_verify_accuracy")
        local_comp_acc = _overall(report, "local_competition_accuracy")
        boxed_rate = _overall(report, "boxed_valid_rate")
        avg_words = _overall(report, "avg_prediction_words")
        family_std = _family_accuracy_std(report)
        subtype_std = _subtype_accuracy_std(report)
        worst_family, worst_delta_samples, worst_delta_accuracy = _worst_family_delta(
            report,
            baseline,
            large_family_min_n=large_family_min_n,
        )
        worst_subtype, worst_subtype_delta_samples, worst_subtype_delta_accuracy = _worst_subtype_delta(
            report,
            baseline,
        )
        overall_delta = official_acc - baseline_overall
        pass_gate, gate_reason = _gate_reason(
            model=model,
            baseline_name=baseline_name,
            overall_delta=overall_delta,
            min_overall_delta=min_overall_delta,
            worst_family=worst_family,
            worst_family_delta_samples=worst_delta_samples,
            max_family_regression_samples=max_family_regression_samples,
        )
        rows.append(
            {
                "rank": 0,
                "model": model,
                "submit_candidate": False,
                "pass_gate": pass_gate,
                "gate_reason": gate_reason,
                "official_verify_accuracy": official_acc,
                "delta_vs_baseline": overall_delta,
                "boxed_valid_rate": boxed_rate,
                "local_competition_accuracy": local_comp_acc,
                "avg_prediction_words": avg_words,
                "family_accuracy_std": family_std,
                "subtype_accuracy_std": subtype_std,
                "worst_family": worst_family,
                "worst_family_delta_accuracy": worst_delta_accuracy,
                "worst_family_delta_samples": worst_delta_samples,
                "worst_subtype": worst_subtype,
                "worst_subtype_delta_accuracy": worst_subtype_delta_accuracy,
                "worst_subtype_delta_samples": worst_subtype_delta_samples,
            }
        )
    rows.sort(key=_sort_key)
    submit_marked = False
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
        if not submit_marked and row["pass_gate"]:
            row["submit_candidate"] = True
            submit_marked = True
    return rows


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = payload["rows"]
    lines = [
        "# Exact Eval Candidate Ranking",
        "",
        f"- baseline: `{payload['baseline']}`",
        f"- min overall delta: `{payload['settings']['min_overall_delta']}`",
        f"- max family regression samples: `{payload['settings']['max_family_regression_samples']}`",
        f"- large family min n: `{payload['settings']['large_family_min_n']}`",
        "",
        "| rank | model | submit | gate | official_verify | delta | boxed_valid | avg_words | family_std | subtype_std | worst_family | worst_family_samples | worst_subtype | worst_subtype_samples |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {rank} | {model} | {submit_candidate} | {gate_reason} | "
            "{official_verify_accuracy:.4f} | {delta_vs_baseline:.4f} | "
            "{boxed_valid_rate:.4f} | {avg_prediction_words:.2f} | "
            "{family_accuracy_std:.4f} | {subtype_accuracy_std:.4f} | "
            "{worst_family} | {worst_family_delta_samples:.2f} | "
            "{worst_subtype} | {worst_subtype_delta_samples:.2f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank exact-eval JSON reports and mark the first candidate that passes submit gates."
    )
    parser.add_argument("--report", action="append", required=True, help="Candidate report as name=path.json")
    parser.add_argument("--baseline", help="Baseline candidate name. Defaults to the first --report name.")
    parser.add_argument("--output-json", type=Path, default=Path("data/processed/eval/exact_eval_ranking.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/eval/exact_eval_ranking.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("LOCAL_EXACT_EVAL_RANKING.md"))
    parser.add_argument("--min-overall-delta", type=float, default=0.0)
    parser.add_argument("--max-family-regression-samples", type=float, default=1.0)
    parser.add_argument("--large-family-min-n", type=int, default=24)
    args = parser.parse_args()

    reports: dict[str, dict[str, Any]] = {}
    report_paths: dict[str, str] = {}
    for value in args.report:
        name, path = _parse_report_arg(value)
        if name in reports:
            raise SystemExit(f"Duplicate report name: {name}")
        reports[name] = _load_json(path)
        report_paths[name] = str(path)
    if not reports:
        raise SystemExit("No reports supplied.")

    baseline_name = args.baseline or next(iter(reports))
    if baseline_name not in reports:
        raise SystemExit(f"Baseline {baseline_name!r} not found. Available: {', '.join(reports)}")

    rows = _build_rows(
        reports,
        baseline_name=baseline_name,
        min_overall_delta=float(args.min_overall_delta),
        max_family_regression_samples=float(args.max_family_regression_samples),
        large_family_min_n=int(args.large_family_min_n),
    )
    payload = {
        "baseline": baseline_name,
        "report_paths": report_paths,
        "settings": {
            "min_overall_delta": float(args.min_overall_delta),
            "max_family_regression_samples": float(args.max_family_regression_samples),
            "large_family_min_n": int(args.large_family_min_n),
        },
        "rows": rows,
    }
    _write_json(args.output_json, payload)
    _write_csv(args.output_csv, rows)
    _write_markdown(args.output_md, payload)
    submit = next((row for row in rows if row["submit_candidate"]), None)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "submit_candidate": None if submit is None else submit["model"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
