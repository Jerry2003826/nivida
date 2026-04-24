from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(raw_counts: dict[str, Any]) -> dict[int, float]:
    counts = {int(key): float(value) for key, value in raw_counts.items() if float(value) > 0}
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def topk(dist: dict[int, float], k: int) -> list[int]:
    return [expert for expert, _ in sorted(dist.items(), key=lambda item: item[1], reverse=True)[:k]]


def entropy_norm(dist: dict[int, float]) -> float:
    probs = [value for value in normalize({str(k): v for k, v in dist.items()}).values() if value > 0]
    if len(probs) <= 1:
        return 0.0
    return -sum(p * math.log(p) for p in probs) / math.log(len(probs))


def route_layer_stats(route_report: dict[str, Any], *, top_k: int) -> dict[int, dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {}
    for row in route_report.get("layers", []):
        layer = int(row["layer"])
        dist = normalize(row.get("topk_counts") or row.get("top1_counts") or {})
        route_top = topk(dist, top_k)
        stats[layer] = {
            "route_top": route_top,
            "topk_mass": sum(dist.get(expert, 0.0) for expert in route_top),
            "entropy_norm": entropy_norm(dist),
            "dist": dist,
        }
    return stats


def public_leverage_by_layer(stability_report: dict[str, Any]) -> dict[int, int]:
    for row in stability_report.get("source_leverage_source_normalized_mixed", []):
        if row.get("removed_source") == "public_visible":
            return {int(layer_row["layer"]): int(layer_row["leverage"]) for layer_row in row.get("layers", [])}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Find safer norm-route expert intersections.")
    parser.add_argument(
        "--norm-report",
        type=Path,
        default=Path("artifacts/adapter_stage2_thin_expertmean_shared_top8_s1/expertmean_shared_report.json"),
    )
    parser.add_argument(
        "--route-report",
        type=Path,
        default=Path("data/processed/route_probe/mixed_source_norm_no_public_hard40_all40_train20.json"),
    )
    parser.add_argument(
        "--stability-report",
        type=Path,
        default=Path("data/processed/route_probe/selection_stability_aggregate.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/processed/route_probe/safe_intersections_no_public_source_norm.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("SAFE_EXPERT_INTERSECTIONS_no_public_source_norm.md"),
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-topk-mass", type=float, default=0.35)
    parser.add_argument("--max-entropy", type=float, default=0.87)
    parser.add_argument("--max-public-leverage", type=int, default=1)
    args = parser.parse_args()

    norm_report = load_json(args.norm_report)
    route_report = load_json(args.route_report)
    stability_report = load_json(args.stability_report)
    route_stats = route_layer_stats(route_report, top_k=args.top_k)
    public_leverage = public_leverage_by_layer(stability_report)

    rows: list[dict[str, Any]] = []
    for module_row in norm_report.get("modified_modules", []):
        layer = int(module_row["layer"])
        module = str(module_row["module"])
        norm_top = [int(expert) for expert in module_row.get("selected_experts", [])]
        stats = route_stats.get(layer)
        if stats is None:
            continue
        route_top = stats["route_top"]
        intersection = [expert for expert in norm_top if expert in set(route_top)]
        leverage = int(public_leverage.get(layer, 0))
        passes_layer_filter = (
            stats["topk_mass"] >= args.min_topk_mass
            and stats["entropy_norm"] <= args.max_entropy
            and leverage <= args.max_public_leverage
        )
        rows.append(
            {
                "layer": layer,
                "module": module,
                "norm_top": norm_top,
                "route_top": route_top,
                "intersection": intersection,
                "intersection_size": len(intersection),
                "route_topk_mass": stats["topk_mass"],
                "route_entropy_norm": stats["entropy_norm"],
                "public_visible_leverage": leverage,
                "passes_layer_filter": passes_layer_filter,
                "recommended_experts": intersection if passes_layer_filter else [],
            }
        )

    recommended = [row for row in rows if row["recommended_experts"]]
    payload = {
        "norm_report": str(args.norm_report),
        "route_report": str(args.route_report),
        "stability_report": str(args.stability_report),
        "filters": {
            "top_k": args.top_k,
            "min_topk_mass": args.min_topk_mass,
            "max_entropy": args.max_entropy,
            "max_public_leverage": args.max_public_leverage,
        },
        "num_modules": len(rows),
        "num_modules_with_intersection": sum(1 for row in rows if row["intersection"]),
        "num_recommended_modules": len(recommended),
        "num_recommended_expert_slots": sum(len(row["recommended_experts"]) for row in recommended),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Safe Expert Intersections: no-public source-normalized route",
        "",
        "This is a local-only diagnostic. It does not imply a submission should be made.",
        "",
        "## Summary",
        "",
        f"- modules checked: `{payload['num_modules']}`",
        f"- modules with norm-route intersection: `{payload['num_modules_with_intersection']}`",
        f"- recommended modules after layer filters: `{payload['num_recommended_modules']}`",
        f"- recommended expert slots: `{payload['num_recommended_expert_slots']}`",
        "",
        "Layer filters:",
        "",
        f"- route top8 mass >= `{args.min_topk_mass}`",
        f"- route normalized entropy <= `{args.max_entropy}`",
        f"- public visible leverage <= `{args.max_public_leverage}`",
        "",
        "## Recommended Rows",
        "",
        "| layer | module | experts | top8_mass | entropy | public_leverage |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in recommended:
        experts = ", ".join(str(expert) for expert in row["recommended_experts"])
        lines.append(
            f"| {row['layer']} | {row['module']} | {experts} | "
            f"{row['route_topk_mass']:.3f} | {row['route_entropy_norm']:.3f} | "
            f"{row['public_visible_leverage']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use this only as a candidate filter. The next required step is local parsed-exact inference eval.",
            "If the recommended set is small, keep it small; do not force top8 per layer.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "num_recommended_modules": payload["num_recommended_modules"],
                "num_recommended_expert_slots": payload["num_recommended_expert_slots"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
