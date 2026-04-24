from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render route-selection stability JSON as CSV/Markdown.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/route_probe/selection_stability_aggregate.json"),
    )
    parser.add_argument(
        "--safe-intersections",
        type=Path,
        default=Path("data/processed/route_probe/safe_intersections_no_public_source_norm.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("SELECTION_STABILITY_SUMMARY.md"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/route_probe/selection_stability_summary.csv"),
    )
    args = parser.parse_args()

    payload = _load(args.input)
    safe = _load(args.safe_intersections) if args.safe_intersections.is_file() else {}

    csv_rows: list[dict[str, Any]] = []
    for row in payload.get("effective_raw_count_weights", []):
        csv_rows.append(
            {
                "section": "effective_raw_count_weights",
                "name": row["source"],
                "comparison": "",
                "nominal_weight": row["nominal_weight"],
                "raw_count_share": row["effective_raw_count_share"],
                "num_rows": row["num_rows"],
                "max_new_tokens": row["max_new_tokens"],
                "mean_overlap": "",
                "min_overlap": "",
                "max_overlap": "",
                "mean_jaccard": "",
                "mean_jsd": "",
                "mean_leverage": "",
                "max_leverage": "",
            }
        )
    for pair in payload.get("pairs", []):
        csv_rows.append(
            {
                "section": "pair_summary",
                "name": f"{pair['a']} vs {pair['b']}",
                "comparison": f"{pair['a']} vs {pair['b']}",
                "nominal_weight": "",
                "raw_count_share": "",
                "num_rows": "",
                "max_new_tokens": "",
                "mean_overlap": pair["mean_overlap"],
                "min_overlap": pair["min_overlap"],
                "max_overlap": pair["max_overlap"],
                "mean_jaccard": pair["mean_jaccard"],
                "mean_jsd": pair["mean_jsd"],
                "mean_leverage": "",
                "max_leverage": "",
            }
        )
    for section in ("source_leverage_raw_mixed", "source_leverage_source_normalized_mixed"):
        for row in payload.get(section, []):
            csv_rows.append(
                {
                    "section": section,
                    "name": row["removed_source"],
                    "comparison": f"remove {row['removed_source']}",
                    "nominal_weight": "",
                    "raw_count_share": "",
                    "num_rows": "",
                    "max_new_tokens": "",
                    "mean_overlap": "",
                    "min_overlap": "",
                    "max_overlap": "",
                    "mean_jaccard": "",
                    "mean_jsd": "",
                    "mean_leverage": row["mean_leverage"],
                    "max_leverage": row["max_leverage"],
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# Selection Stability Summary",
        "",
        "This is a local-only route selection audit. It does not recommend a submission by itself.",
        "",
        "## Effective Raw-Count Shares",
        "",
        "| source | nominal_weight | raw_count_share | rows | max_new_tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("effective_raw_count_weights", []):
        lines.append(
            f"| {row['source']} | {_fmt(row['nominal_weight'], 2)} | "
            f"{_fmt(row['effective_raw_count_share'])} | {row['num_rows']} | "
            f"{row['max_new_tokens']} |"
        )

    lines.extend(
        [
            "",
            "## Pair Summaries",
            "",
            "| comparison | mean_overlap | min | max | mean_jaccard | mean_jsd | overlap_hist |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for pair in payload.get("pairs", []):
        lines.append(
            f"| {pair['a']} vs {pair['b']} | {_fmt(pair['mean_overlap'], 2)} / 8 | "
            f"{pair['min_overlap']} | {pair['max_overlap']} | "
            f"{_fmt(pair['mean_jaccard'], 3)} | {_fmt(pair['mean_jsd'])} | "
            f"`{json.dumps(pair['overlap_hist'], ensure_ascii=False)}` |"
        )

    lines.extend(
        [
            "",
            "## Source Leverage",
            "",
            "Leverage is `8 - overlap(full_mix_top8, remove_source_top8)`.",
            "",
            "| mix | removed_source | mean_leverage | max_leverage | leverage_hist |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for section, label in (
        ("source_leverage_raw_mixed", "raw_mixed"),
        ("source_leverage_source_normalized_mixed", "source_normalized_mixed"),
    ):
        for row in payload.get(section, []):
            lines.append(
                f"| {label} | {row['removed_source']} | {_fmt(row['mean_leverage'], 2)} | "
                f"{row['max_leverage']} | `{json.dumps(row['leverage_hist'], ensure_ascii=False)}` |"
            )

    if safe:
        lines.extend(
            [
                "",
                "## Safe Intersection Filter",
                "",
                f"- modules checked: `{safe.get('num_modules')}`",
                f"- modules with norm-route intersection: `{safe.get('num_modules_with_intersection')}`",
                f"- recommended modules after filters: `{safe.get('num_recommended_modules')}`",
                f"- recommended expert slots: `{safe.get('num_recommended_expert_slots')}`",
                "",
                "Recommended rows are documented in `SAFE_EXPERT_INTERSECTIONS_no_public_source_norm.md`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Bottom Line",
            "",
            "Raw-count mixing materially overweights visible public in route counts, but source-normalizing does not fully flip top8 selection. The safer interpretation is a combined problem: route-count weighting is misaligned with the intended source mixture, and routed-expert-to-shared-expert transplant remains structurally risky.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output_md": str(args.output_md), "output_csv": str(args.output_csv), "rows": len(csv_rows)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
