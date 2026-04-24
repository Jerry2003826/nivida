from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _add_counts(dst: Counter[int], raw_counts: dict[str, Any], weight: float) -> None:
    for key, value in raw_counts.items():
        count = float(value)
        if count > 0:
            dst[int(key)] += weight * count


def _normalize_counts(raw_counts: dict[str, Any]) -> dict[int, float]:
    counts = {int(key): float(value) for key, value in raw_counts.items() if float(value) > 0}
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def _source_layer_distributions(payload: dict[str, Any], normalization: str) -> dict[int, dict[int, float]]:
    if normalization == "raw":
        return {
            int(row["layer"]): {
                int(key): float(value)
                for key, value in (row.get("topk_counts") or row.get("top1_counts") or {}).items()
                if float(value) > 0
            }
            for row in payload.get("layers", [])
        }
    if normalization == "source":
        return {
            int(row["layer"]): _normalize_counts(row.get("topk_counts") or row.get("top1_counts") or {})
            for row in payload.get("layers", [])
        }
    if normalization != "example":
        raise ValueError(f"unknown normalization={normalization!r}")
    examples = payload.get("examples")
    if not examples:
        raise ValueError(
            "normalization='example' requires probe reports generated with --record-examples"
        )
    by_layer: dict[int, Counter[int]] = defaultdict(Counter)
    by_layer_examples: Counter[int] = Counter()
    for example in examples:
        seen_layers: set[int] = set()
        for row in example.get("layers", []):
            layer = int(row["layer"])
            dist = _normalize_counts(row.get("topk_counts") or row.get("top1_counts") or {})
            if not dist:
                continue
            for expert, prob in dist.items():
                by_layer[layer][expert] += prob
            seen_layers.add(layer)
        for layer in seen_layers:
            by_layer_examples[layer] += 1
    return {
        layer: {
            expert: score / max(1, by_layer_examples[layer])
            for expert, score in counts.items()
        }
        for layer, counts in by_layer.items()
    }


def mix_reports(
    inputs: list[tuple[Path, float]],
    output: Path,
    *,
    normalization: str = "raw",
) -> dict[str, Any]:
    by_layer: dict[int, Counter[int]] = defaultdict(Counter)
    sources: list[dict[str, Any]] = []

    for path, weight in inputs:
        payload = _load(path)
        sources.append(
            {
                "path": str(path),
                "weight": weight,
                "num_rows": payload.get("num_rows"),
                "top_k": payload.get("top_k"),
                "max_new_tokens": payload.get("max_new_tokens"),
                "count_scope": payload.get("count_scope"),
                "record_examples": payload.get("record_examples"),
            }
        )
        layer_distributions = _source_layer_distributions(payload, normalization)
        for layer, counts in layer_distributions.items():
            _add_counts(by_layer[layer], {str(k): v for k, v in counts.items()}, weight)

    layers: list[dict[str, Any]] = []
    for layer, counts in sorted(by_layer.items()):
        total = float(sum(counts.values()))
        layers.append(
            {
                "layer": layer,
                "topk_total": total,
                "topk_counts": {
                    str(expert): float(count)
                    for expert, count in counts.most_common()
                },
                "topk_head": [
                    {
                        "expert": int(expert),
                        "count": float(count),
                        "fraction": 0.0 if total <= 0 else float(count / total),
                    }
                    for expert, count in counts.most_common(16)
                ],
            }
        )

    payload = {
        "kind": "mixed_route_report",
        "normalization": normalization,
        "sources": sources,
        "layers": layers,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mix multiple Nemotron route probe reports into one weighted route report."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Report plus weight in the form path:weight. Repeatable.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--normalization",
        choices=["raw", "source", "example"],
        default="raw",
        help=(
            "raw mixes route counts directly; source normalizes each source-layer first; "
            "example averages per-example layer distributions inside each source first."
        ),
    )
    args = parser.parse_args()

    parsed: list[tuple[Path, float]] = []
    for item in args.input:
        path_text, weight_text = item.rsplit(":", 1)
        parsed.append((Path(path_text), float(weight_text)))
    payload = mix_reports(parsed, args.output, normalization=args.normalization)
    print(json.dumps({"output": str(args.output), "num_layers": len(payload["layers"])}, indent=2))


if __name__ == "__main__":
    main()
