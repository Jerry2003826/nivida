from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a small, auditable safe shared-expert candidate plan."
    )
    parser.add_argument(
        "--safe-intersections",
        type=Path,
        default=Path("data/processed/route_probe/safe_intersections_no_public_source_norm.json"),
    )
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--max-experts-per-module", type=int, default=3)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/processed/route_probe/safe_shared_candidate_plan_scale025.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("SAFE_SHARED_CANDIDATE_PLAN_scale025.md"),
    )
    args = parser.parse_args()

    payload = load_json(args.safe_intersections)
    candidate_rows: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        experts = list(row.get("recommended_experts") or [])[: args.max_experts_per_module]
        if not experts:
            continue
        candidate_rows.append(
            {
                "layer": int(row["layer"]),
                "module": str(row["module"]),
                "experts": [int(expert) for expert in experts],
                "scale": args.scale,
                "route_topk_mass": float(row["route_topk_mass"]),
                "route_entropy_norm": float(row["route_entropy_norm"]),
                "public_visible_leverage": int(row["public_visible_leverage"]),
            }
        )
    out = {
        "kind": "safe_shared_candidate_plan",
        "source": str(args.safe_intersections),
        "scale": args.scale,
        "max_experts_per_module": args.max_experts_per_module,
        "num_modules": len(candidate_rows),
        "num_expert_slots": sum(len(row["experts"]) for row in candidate_rows),
        "rows": candidate_rows,
        "notes": [
            "This plan is not a submission recommendation.",
            "It should be evaluated locally with parsed exact metrics before any Kaggle submission.",
            "Visible public is diagnostic-only and is not used as a selection source.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Safe Shared Candidate Plan",
        "",
        "This is a small candidate plan, not a submission recommendation.",
        "",
        f"- scale: `{args.scale}`",
        f"- modules: `{out['num_modules']}`",
        f"- expert slots: `{out['num_expert_slots']}`",
        "",
        "| layer | module | experts | scale | top8_mass | entropy | public_leverage |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in candidate_rows:
        experts = ", ".join(str(expert) for expert in row["experts"])
        lines.append(
            f"| {row['layer']} | {row['module']} | {experts} | {row['scale']:.3f} | "
            f"{row['route_topk_mass']:.3f} | {row['route_entropy_norm']:.3f} | "
            f"{row['public_visible_leverage']} |"
        )
    lines.extend(
        [
            "",
            "Next step: build only if local per-example route v3 still supports these intersections, then run local parsed-exact inference eval before any submission.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "modules": out["num_modules"]}, indent=2))


if __name__ == "__main__":
    main()
