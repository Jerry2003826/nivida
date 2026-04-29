from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition.metrics import competition_correct  # noqa: E402
from src.competition.schema import PuzzleExample  # noqa: E402
from src.teacher.chain_search import ChainSearchEngine  # noqa: E402
from src.teacher.family_tagger import apply_family_tags  # noqa: E402


DEFAULT_INPUTS = [
    Path("data/processed/local_eval_manifests/combined_balanced_48pf.jsonl"),
    Path("data/processed/local_eval_manifests/proxy_all_balanced_64pf.jsonl"),
    Path("data/processed/local_eval_manifests/hard_triad_full.jsonl"),
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        return max(len(left), len(right))
    return sum(a != b for a, b in zip(left, right))


def classify_bit_candidate_gap(
    *,
    top_correct: bool,
    oracle_rank: int | None,
    top_support_full: bool,
) -> str:
    if not top_support_full:
        return "support_incomplete"
    if top_correct and oracle_rank == 1:
        return "low_risk_top1"
    if oracle_rank is not None:
        return "ranker_miss_oracle_hit"
    return "operator_gap_oracle_miss"


def boolean_expression_features(expressions: list[dict[str, Any]]) -> dict[str, Any]:
    ops = [str(expr.get("op", "")) for expr in expressions]
    complexities = [int(expr.get("complexity", 0) or 0) for expr in expressions]
    total = sum(complexities)
    return {
        "expression_ops": ",".join(ops),
        "expression_complexity_total": total,
        "expression_complexity_avg": 0.0 if not complexities else total / len(complexities),
        "expression_complexity_max": 0 if not complexities else max(complexities),
    }


def _candidate_expression_features(candidate: Any) -> dict[str, Any]:
    for step in getattr(candidate, "steps", []):
        if getattr(step, "op_name", "") != "binary_boolean_expr":
            continue
        params = getattr(step, "params", {})
        expressions = params.get("expressions", []) if isinstance(params, dict) else []
        if isinstance(expressions, list):
            return boolean_expression_features(
                [expr for expr in expressions if isinstance(expr, dict)]
            )
    return boolean_expression_features([])


def bit_operator_family_from_steps(*, steps: str, expression_ops: str = "") -> str:
    signature = f"{steps},{expression_ops}".lower()
    expression_tokens = {
        token.strip()
        for token in expression_ops.lower().replace(">", ",").split(",")
        if token.strip()
    }
    if "binary_boolean_expr" in signature or bool(
        expression_tokens & {"xor", "and", "or", "not", "choice"}
    ):
        return "boolean_template"
    if "binary_affine_transform" in signature or "affine" in signature or "gf2" in signature:
        return "affine_gf2"
    if "rotate" in signature or "rotation" in signature:
        return "rotation"
    if "reverse_bits" in signature or "reversal" in signature:
        return "reversal"
    if "swap_nibbles" in signature or "nibble" in signature or "byte" in signature:
        return "nibble_byte_transform"
    if "binary_permutation" in signature or "permutation" in signature:
        return "plain_permutation"
    return "unknown"


def oracle_rank_bucket(oracle_rank: int | None) -> str:
    if oracle_rank is None:
        return "miss"
    if oracle_rank == 1:
        return "top1"
    if oracle_rank <= 3:
        return "rank2_3"
    return "rank4_plus"


def support_leave_one_out_stability(
    *,
    top_support_full: bool,
    support_full_candidate_count: int,
    top_complexity_penalty: float,
    top_expression_complexity_total: int,
) -> str:
    if not top_support_full:
        return "unstable_support"
    if support_full_candidate_count <= 1:
        return "unique_support_fit"
    if top_complexity_penalty <= 0.35 and top_expression_complexity_total <= 8:
        return "stable_low_complexity"
    return "ambiguous_support_fit"


def filter_diagnostics(
    rows: list[dict[str, Any]],
    *,
    failure_class: str | None = None,
    subtype: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    filtered = rows
    if failure_class:
        filtered = [row for row in filtered if str(row.get("risk_class")) == failure_class]
    if subtype:
        filtered = [row for row in filtered if str(row.get("subtype")) == subtype]
    if limit is not None and limit >= 0:
        filtered = filtered[:limit]
    return filtered


def _support_full(candidate: Any, example: PuzzleExample) -> bool:
    if candidate is None or len(candidate.predictions) != len(example.parsed_examples):
        return False
    return all(
        competition_correct(prediction, pair.output)
        for prediction, pair in zip(candidate.predictions, example.parsed_examples)
    )


def diagnose_example(engine: ChainSearchEngine, row: dict[str, Any], *, top_k: int) -> dict[str, Any] | None:
    example = PuzzleExample.from_dict(row)
    apply_family_tags([example])
    if example.metadata.official_family != "bit" or example.metadata.subtype != "bit_permutation":
        return None

    candidates = engine.solve_example(example, top_k=top_k)
    target = "" if example.target_answer is None else str(example.target_answer)
    oracle_rank: int | None = None
    support_full_count = 0
    candidate_rows: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates, start=1):
        prediction = "" if candidate.query_prediction is None else str(candidate.query_prediction)
        query_correct = bool(target) and competition_correct(prediction, target)
        candidate_support_full = _support_full(candidate, example)
        if candidate_support_full:
            support_full_count += 1
        if oracle_rank is None and candidate_support_full and query_correct:
            oracle_rank = index
        candidate_rows.append(
            {
                "rank": index,
                "prediction": prediction,
                "query_correct": query_correct,
                "support_full": candidate_support_full,
                "hamming_to_target": "" if not target else hamming_distance(prediction, target),
                "hamming_to_query": hamming_distance(prediction, example.query),
                "score": float(candidate.score),
                "exact_ratio": float(candidate.exact_ratio),
                "steps": ">".join(step.op_name for step in candidate.steps),
                "complexity_penalty": float(candidate.debug.get("complexity_penalty", 0.0)),
                **_candidate_expression_features(candidate),
            }
        )
        candidate_rows[-1]["operator_family"] = bit_operator_family_from_steps(
            steps=str(candidate_rows[-1]["steps"]),
            expression_ops=str(candidate_rows[-1]["expression_ops"]),
        )

    top = candidate_rows[0] if candidate_rows else None
    oracle = candidate_rows[oracle_rank - 1] if oracle_rank is not None and oracle_rank <= len(candidate_rows) else None
    top_prediction = "" if top is None else str(top["prediction"])
    oracle_prediction = "" if oracle is None else str(oracle["prediction"])
    top_correct = bool(top and top["query_correct"])
    top_support_full = bool(top and top["support_full"])
    risk_class = classify_bit_candidate_gap(
        top_correct=top_correct,
        oracle_rank=oracle_rank,
        top_support_full=top_support_full,
    )
    stability_bucket = support_leave_one_out_stability(
        top_support_full=top_support_full,
        support_full_candidate_count=support_full_count,
        top_complexity_penalty=0.0 if top is None else float(top["complexity_penalty"]),
        top_expression_complexity_total=0 if top is None else int(top["expression_complexity_total"]),
    )
    return {
        "id": example.id,
        "family": example.metadata.official_family,
        "subtype": example.metadata.subtype,
        "query": example.query,
        "target": target,
        "num_pairs": len(example.parsed_examples),
        "top_prediction": top_prediction,
        "top_steps": "" if top is None else str(top["steps"]),
        "top_operator_family": "unknown" if top is None else str(top["operator_family"]),
        "top_hamming_to_target": "" if top is None else top["hamming_to_target"],
        "top_expression_ops": "" if top is None else top["expression_ops"],
        "top_expression_complexity_total": 0 if top is None else top["expression_complexity_total"],
        "top_complexity_penalty": 0.0 if top is None else top["complexity_penalty"],
        "oracle_prediction": oracle_prediction,
        "oracle_steps": "" if oracle is None else str(oracle["steps"]),
        "oracle_operator_family": "unknown" if oracle is None else str(oracle["operator_family"]),
        "oracle_hamming_to_target": "" if oracle is None else oracle["hamming_to_target"],
        "oracle_expression_ops": "" if oracle is None else oracle["expression_ops"],
        "oracle_expression_complexity_total": 0 if oracle is None else oracle["expression_complexity_total"],
        "top_oracle_hamming": "" if not top_prediction or not oracle_prediction else hamming_distance(top_prediction, oracle_prediction),
        "top_query_correct": top_correct,
        "top_support_full": top_support_full,
        "oracle_rank": oracle_rank,
        "oracle_rank_bucket": oracle_rank_bucket(oracle_rank),
        "support_full_candidate_count": support_full_count,
        "support_leave_one_out_stability": stability_bucket,
        "support_leave_one_out_stable": stability_bucket in {"unique_support_fit", "stable_low_complexity"},
        "risk_class": risk_class,
        "candidates": candidate_rows,
    }


def run_diagnostic(inputs: list[Path], *, top_k: int, beam_width: int, max_depth: int) -> list[dict[str, Any]]:
    engine = ChainSearchEngine(beam_width=beam_width, max_depth=max_depth)
    diagnostics: list[dict[str, Any]] = []
    for path in inputs:
        for row in _load_jsonl(path):
            diagnostic = diagnose_example(engine, row, top_k=top_k)
            if diagnostic is None:
                continue
            diagnostic["path"] = str(path)
            diagnostics.append(diagnostic)
    return diagnostics


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat: dict[str, Any] = {}
        for key, value in row.items():
            if key == "candidates":
                continue
            if isinstance(value, (list, dict)):
                flat[key] = json.dumps(value, ensure_ascii=False)
            else:
                flat[key] = value
        flat_rows.append(flat)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Bit Permutation Diagnostic",
        "",
        "| manifest | risk_class | n | top1_acc | oracle_at_k | support_full_top1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    manifests = sorted({str(row.get("path", "")) for row in rows})
    for manifest in manifests:
        manifest_rows = [row for row in rows if str(row.get("path", "")) == manifest]
        risk_classes = sorted({str(row["risk_class"]) for row in manifest_rows})
        for risk_class in risk_classes:
            group = [row for row in manifest_rows if row["risk_class"] == risk_class]
            lines.append(
                f"| `{manifest}` | {risk_class} | {len(group)} | "
                f"{_rate([bool(row['top_query_correct']) for row in group]):.4f} | "
                f"{_rate([row['oracle_rank'] is not None for row in group]):.4f} | "
                f"{_rate([bool(row['top_support_full']) for row in group]):.4f} |"
            )

    lines.extend(["", "## Top Step Buckets", ""])
    top_steps = Counter(str(row["top_steps"]) for row in rows)
    for steps, count in top_steps.most_common(12):
        lines.append(f"- `{steps}`: {count}")

    lines.extend(["", "## Ranker Misses", ""])
    misses = [row for row in rows if row["risk_class"] == "ranker_miss_oracle_hit"][:20]
    for row in misses:
        lines.append(
            f"- `{row['id']}` oracle_rank=`{row['oracle_rank']}` target=`{row['target']}` "
            f"top=`{row['top_prediction']}` oracle=`{row['oracle_prediction']}` "
            f"top_oracle_hamming=`{row['top_oracle_hamming']}` "
            f"top_ops=`{row['top_expression_ops']}` oracle_ops=`{row['oracle_expression_ops']}` "
            f"steps=`{row['top_steps']}`"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose bit_permutation candidate and ranker gaps.")
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--failure-class", help="Optional risk_class filter applied to diagnostic outputs.")
    parser.add_argument("--subtype", default="bit_permutation", help="Optional subtype filter applied to outputs.")
    parser.add_argument("--limit", type=int, help="Optional max rows after filters; use 0 for an empty smoke output.")
    parser.add_argument("--output-json", type=Path, default=Path("data/processed/bit_permutation_diagnostic.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/bit_permutation_diagnostic.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/bit_permutation_diagnostic_latest.md"))
    args = parser.parse_args()

    all_rows = run_diagnostic(
        args.input or DEFAULT_INPUTS,
        top_k=args.top_k,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
    )
    rows = filter_diagnostics(
        all_rows,
        failure_class=args.failure_class,
        subtype=args.subtype,
        limit=args.limit,
    )
    _write_json(
        args.output_json,
        {
            "settings": {
                "top_k": args.top_k,
                "beam_width": args.beam_width,
                "max_depth": args.max_depth,
                "failure_class": args.failure_class,
                "subtype": args.subtype,
                "limit": args.limit,
                "num_rows_before_filters": len(all_rows),
            },
            "rows": rows,
        },
    )
    _write_csv(args.output_csv, rows)
    _write_markdown(args.output_md, rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "output_md": str(args.output_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
