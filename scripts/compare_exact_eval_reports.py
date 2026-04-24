from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_input(text: str) -> tuple[str, Path]:
    if "=" not in text:
        path = Path(text)
        return path.stem, path
    name, path_text = text.split("=", 1)
    return name, Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple exact-eval JSON reports.")
    parser.add_argument("--report", action="append", required=True, help="name=path.json or path.json. Repeatable.")
    parser.add_argument("--baseline")
    parser.add_argument("--output-md", type=Path, default=Path("EXACT_EVAL_COMPARISON.md"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/eval/exact_eval_comparison.csv"))
    args = parser.parse_args()

    loaded: dict[str, dict[str, Any]] = {}
    for item in args.report:
        name, path = parse_input(item)
        loaded[name] = load_report(path)
    baseline_name = args.baseline or next(iter(loaded))
    baseline = loaded[baseline_name]

    rows: list[dict[str, Any]] = []
    for name, report in loaded.items():
        overall = report.get("overall", {})
        base_overall = baseline.get("overall", {})
        rows.append(
            {
                "scope": "overall",
                "family": "ALL",
                "model": name,
                "n": report.get("num_joined", 0),
                "official_verify_accuracy": overall.get("official_verify_accuracy", 0.0),
                "delta_vs_baseline": overall.get("official_verify_accuracy", 0.0)
                - base_overall.get("official_verify_accuracy", 0.0),
                "boxed_valid_rate": overall.get("boxed_valid_rate", 0.0),
                "local_competition_accuracy": overall.get("local_competition_accuracy", 0.0),
            }
        )
        families = sorted(set(report.get("family", {})) | set(baseline.get("family", {})))
        for family in families:
            current = report.get("family", {}).get(family, {})
            base = baseline.get("family", {}).get(family, {})
            rows.append(
                {
                    "scope": "family",
                    "family": family,
                    "model": name,
                    "n": current.get("n", 0),
                    "official_verify_accuracy": current.get("official_verify_accuracy", 0.0),
                    "delta_vs_baseline": current.get("official_verify_accuracy", 0.0)
                    - base.get("official_verify_accuracy", 0.0),
                    "boxed_valid_rate": current.get("boxed_valid_rate", 0.0),
                    "local_competition_accuracy": current.get("local_competition_accuracy", 0.0),
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Exact Eval Comparison",
        "",
        f"Baseline: `{baseline_name}`",
        "",
        "## Overall",
        "",
        "| model | n | official_verify | delta | boxed_valid | local_competition |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["scope"] != "overall":
            continue
        lines.append(
            f"| {row['model']} | {row['n']} | {row['official_verify_accuracy']:.4f} | "
            f"{row['delta_vs_baseline']:+.4f} | {row['boxed_valid_rate']:.4f} | "
            f"{row['local_competition_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Family",
            "",
            "| family | model | n | official_verify | delta | boxed_valid |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row["scope"] != "family":
            continue
        lines.append(
            f"| {row['family']} | {row['model']} | {row['n']} | "
            f"{row['official_verify_accuracy']:.4f} | {row['delta_vs_baseline']:+.4f} | "
            f"{row['boxed_valid_rate']:.4f} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_md": str(args.output_md), "output_csv": str(args.output_csv), "reports": len(loaded)}, indent=2))


if __name__ == "__main__":
    main()
