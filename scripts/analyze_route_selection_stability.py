from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SOURCES = {
    "public_visible": ("public_test35.json", 0.25),
    "official_hard": ("official_hard.json", 0.30),
    "official_all": ("official_all.json", 0.30),
    "stage2_train": ("stage2_train.json", 0.15),
}

DEFAULT_NO_PUBLIC_SOURCES = {
    "official_hard": ("official_hard.json", 0.40),
    "official_all": ("official_all.json", 0.40),
    "stage2_train": ("stage2_train.json", 0.20),
}


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_source_specs(
    specs: list[str] | None,
    defaults: dict[str, tuple[str, float]],
) -> dict[str, tuple[str, float]]:
    if not specs:
        return defaults
    parsed: dict[str, tuple[str, float]] = {}
    for spec in specs:
        name, rest = spec.split("=", 1)
        path_text, weight_text = rest.rsplit(":", 1)
        parsed[name] = (path_text, float(weight_text))
    return parsed


def layer_counts(report: dict[str, Any]) -> dict[int, dict[int, float]]:
    out: dict[int, dict[int, float]] = {}
    for row in report.get("layers", []):
        out[int(row["layer"])] = {
            int(k): float(v)
            for k, v in (row.get("topk_counts") or {}).items()
        }
    return out


def normalize(counts: dict[int, float]) -> dict[int, float]:
    total = float(sum(v for v in counts.values() if v > 0))
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in counts.items() if v > 0}


def topk(dist: dict[int, float], k: int = 8) -> list[int]:
    return [expert for expert, _ in sorted(dist.items(), key=lambda item: item[1], reverse=True)[:k]]


def overlap(a: list[int], b: list[int]) -> int:
    return len(set(a) & set(b))


def hist(values: list[int]) -> dict[str, int]:
    counter = Counter(values)
    return {str(k): int(counter[k]) for k in sorted(counter)}


def jsd(p: dict[int, float], q: dict[int, float]) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    p_norm = normalize({k: p.get(k, 0.0) for k in keys})
    q_norm = normalize({k: q.get(k, 0.0) for k in keys})
    m = {k: 0.5 * (p_norm.get(k, 0.0) + q_norm.get(k, 0.0)) for k in keys}

    def kl(a: dict[int, float], b: dict[int, float]) -> float:
        total = 0.0
        for key, av in a.items():
            if av > 0:
                total += av * math.log(av / max(b.get(key, 0.0), 1e-12), 2)
        return total

    return 0.5 * kl(p_norm, m) + 0.5 * kl(q_norm, m)


def mix_raw(
    source_layers: dict[str, dict[int, dict[int, float]]],
    weights: dict[str, float],
) -> dict[int, dict[int, float]]:
    layers = sorted({layer for src in source_layers.values() for layer in src})
    mixed: dict[int, dict[int, float]] = {}
    for layer in layers:
        acc: Counter[int] = Counter()
        for name, weight in weights.items():
            for expert, count in source_layers[name].get(layer, {}).items():
                acc[expert] += weight * count
        mixed[layer] = dict(acc)
    return mixed


def mix_source_normalized(
    source_layers: dict[str, dict[int, dict[int, float]]],
    weights: dict[str, float],
) -> dict[int, dict[int, float]]:
    total_weight = sum(weights.values())
    norm_weights = {k: v / total_weight for k, v in weights.items() if total_weight > 0}
    layers = sorted({layer for src in source_layers.values() for layer in src})
    mixed: dict[int, dict[int, float]] = {}
    for layer in layers:
        acc: Counter[int] = Counter()
        for name, weight in norm_weights.items():
            dist = normalize(source_layers[name].get(layer, {}))
            for expert, prob in dist.items():
                acc[expert] += weight * prob
        mixed[layer] = dict(acc)
    return mixed


def summarize_pair(
    a_name: str,
    a_layers: dict[int, dict[int, float]],
    b_name: str,
    b_layers: dict[int, dict[int, float]],
    *,
    k: int,
) -> dict[str, Any]:
    rows = []
    overlaps = []
    jensen = []
    for layer in sorted(set(a_layers) & set(b_layers)):
        a_top = topk(a_layers[layer], k)
        b_top = topk(b_layers[layer], k)
        ov = overlap(a_top, b_top)
        overlaps.append(ov)
        j = jsd(a_layers[layer], b_layers[layer])
        jensen.append(j)
        rows.append(
            {
                "layer": layer,
                "overlap": ov,
                "jaccard": 0.0 if not (set(a_top) | set(b_top)) else ov / len(set(a_top) | set(b_top)),
                "jsd": j,
                f"{a_name}_top{k}": a_top,
                f"{b_name}_top{k}": b_top,
            }
        )
    return {
        "a": a_name,
        "b": b_name,
        "mean_overlap": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "min_overlap": min(overlaps) if overlaps else 0,
        "max_overlap": max(overlaps) if overlaps else 0,
        "overlap_hist": hist(overlaps),
        "mean_jaccard": sum(row["jaccard"] for row in rows) / len(rows) if rows else 0.0,
        "mean_jsd": sum(jensen) / len(jensen) if jensen else 0.0,
        "max_jsd": max(jensen) if jensen else 0.0,
        "layers": rows,
    }


def source_leverage(
    full_name: str,
    full_layers: dict[int, dict[int, float]],
    source_layers: dict[str, dict[int, dict[int, float]]],
    weights: dict[str, float],
    *,
    k: int,
    source_norm: bool,
) -> list[dict[str, Any]]:
    out = []
    for removed in weights:
        keep_weights = {name: weight for name, weight in weights.items() if name != removed}
        reduced = (
            mix_source_normalized(source_layers, keep_weights)
            if source_norm
            else mix_raw(source_layers, keep_weights)
        )
        overlaps = []
        rows = []
        for layer in sorted(full_layers):
            full_top = topk(full_layers[layer], k)
            reduced_top = topk(reduced[layer], k)
            ov = overlap(full_top, reduced_top)
            overlaps.append(ov)
            rows.append(
                {
                    "layer": layer,
                    "leverage": k - ov,
                    "overlap": ov,
                    f"{full_name}_top{k}": full_top,
                    f"without_{removed}_top{k}": reduced_top,
                }
            )
        leverages = [k - value for value in overlaps]
        out.append(
            {
                "removed_source": removed,
                "mean_leverage": sum(leverages) / len(leverages) if leverages else 0.0,
                "max_leverage": max(leverages) if leverages else 0,
                "leverage_hist": hist(leverages),
                "layers": rows,
            }
        )
    return out


def effective_raw_weight_summary(
    reports: dict[str, dict[str, Any]],
    source_layers: dict[str, dict[int, dict[int, float]]],
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    weighted_totals = []
    for name, weight in weights.items():
        totals = [sum(layer_counts.values()) for layer_counts in source_layers[name].values()]
        mean_total = sum(totals) / len(totals) if totals else 0.0
        weighted = weight * mean_total
        weighted_totals.append(weighted)
        rows.append(
            {
                "source": name,
                "nominal_weight": weight,
                "num_rows": reports[name].get("num_rows"),
                "max_new_tokens": reports[name].get("max_new_tokens"),
                "mean_layer_topk_total": mean_total,
                "weighted_mean_layer_topk_total": weighted,
            }
        )
    denom = sum(weighted_totals)
    for row, weighted in zip(rows, weighted_totals):
        row["effective_raw_count_share"] = 0.0 if denom <= 0 else weighted / denom
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze aggregate route-selection stability.")
    parser.add_argument("--route-dir", type=Path, default=Path("data/processed/route_probe"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/route_probe/selection_stability_aggregate.json"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--source",
        action="append",
        help="Source as name=filename:weight, relative to --route-dir unless absolute. Repeatable.",
    )
    parser.add_argument(
        "--no-public-source",
        action="append",
        help=(
            "No-public source as name=filename:weight, relative to --route-dir unless absolute. "
            "Repeatable. Defaults to official_hard/official_all/stage2_train."
        ),
    )
    args = parser.parse_args()

    source_specs = parse_source_specs(args.source, DEFAULT_SOURCES)
    no_public_specs = parse_source_specs(args.no_public_source, DEFAULT_NO_PUBLIC_SOURCES)
    all_specs = {**source_specs, **no_public_specs}
    reports = {
        name: load_report(Path(filename) if Path(filename).is_absolute() else args.route_dir / filename)
        for name, (filename, _weight) in all_specs.items()
    }
    weights = {name: weight for name, (_filename, weight) in source_specs.items()}
    no_public_weights = {name: weight for name, (_filename, weight) in no_public_specs.items()}
    source_layers = {name: layer_counts(report) for name, report in reports.items()}

    raw_mixed = mix_raw(source_layers, weights)
    source_norm_mixed = mix_source_normalized(source_layers, weights)
    no_public = mix_source_normalized(source_layers, no_public_weights)

    named_layers = {
        "raw_mixed": raw_mixed,
        "source_normalized_mixed": source_norm_mixed,
        "no_public_source_normalized": no_public,
        **source_layers,
    }

    candidate_pair_names = [
        ("raw_mixed", "source_normalized_mixed"),
        ("raw_mixed", "no_public_source_normalized"),
        ("source_normalized_mixed", "no_public_source_normalized"),
        ("public_visible", "official_all"),
        ("public_visible", "official_hard"),
        ("official_hard", "official_all"),
    ]
    pair_names = [
        (a, b)
        for a, b in candidate_pair_names
        if a in named_layers and b in named_layers
    ]
    pairs = [
        summarize_pair(a, named_layers[a], b, named_layers[b], k=args.top_k)
        for a, b in pair_names
    ]

    payload = {
        "top_k": args.top_k,
        "limitations": [
            "Existing probe reports are aggregate-by-source only, not per-example.",
            "Per-example normalized route selection requires rerunning the route probe with per-row counters.",
            "Generation-only versus prompt-only routing cannot be separated from public_test35.json because that run combined prompt and generation counts.",
        ],
        "effective_raw_count_weights": effective_raw_weight_summary(reports, source_layers, weights),
        "pairs": pairs,
        "source_leverage_raw_mixed": source_leverage(
            "raw_mixed", raw_mixed, source_layers, weights, k=args.top_k, source_norm=False
        ),
        "source_leverage_source_normalized_mixed": source_leverage(
            "source_normalized_mixed", source_norm_mixed, source_layers, weights, k=args.top_k, source_norm=True
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pairs": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()
