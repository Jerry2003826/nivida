from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_bit_permutation import (  # noqa: E402
    DEFAULT_INPUTS as BIT_DEFAULT_INPUTS,
    _load_jsonl as load_bit_jsonl,
    diagnose_example as diagnose_bit_example,
)
from scripts.diagnose_equation_template import (  # noqa: E402
    DEFAULT_INPUTS as EQUATION_DEFAULT_INPUTS,
    diagnose_example as diagnose_equation_example,
)
from src.teacher.chain_search import ChainSearchEngine  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("data/processed/solver_breakout_v2")
DEFAULT_OUTPUT_MD = Path("docs/solver_breakout_v2_latest.md")


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _safe_override_possible(family: str, row: dict[str, Any]) -> bool:
    risk = str(row.get("risk_class", ""))
    if family == "equation_template":
        return risk == "low_risk_support_stable" and bool(row.get("top_query_correct"))
    if family == "bit_permutation":
        return risk == "low_risk_top1" and bool(row.get("top_support_full"))
    return False


def _group_summary(family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    top1 = sum(bool(row.get("top_query_correct")) for row in rows)
    oracle = sum(row.get("oracle_rank") is not None for row in rows)
    support_full_candidates = [
        int(row.get("support_full_candidate_count") or row.get("ambiguity_count") or 0)
        for row in rows
    ]
    safe_override = sum(_safe_override_possible(family, row) for row in rows)
    ranker_miss = sum(str(row.get("risk_class")) == "ranker_miss_oracle_hit" for row in rows)
    operator_gap = sum(str(row.get("risk_class")) == "operator_gap_oracle_miss" for row in rows)
    theoretical_gain_ceiling = sum(
        row.get("oracle_rank") is not None and not bool(row.get("top_query_correct"))
        for row in rows
    )
    return {
        "n": n,
        "top1_correct_count": top1,
        "top1_accuracy": _rate(top1, n),
        "oracle_at_k_count": oracle,
        "oracle_at_k": _rate(oracle, n),
        "support_full_candidate_count_avg": (
            sum(support_full_candidates) / len(support_full_candidates)
            if support_full_candidates
            else 0.0
        ),
        "safe_override_possible_count": safe_override,
        "safe_override_possible_rate": _rate(safe_override, n),
        "ranker_miss_oracle_hit_count": ranker_miss,
        "operator_gap_oracle_miss_count": operator_gap,
        "theoretical_gain_ceiling_count": theoretical_gain_ceiling,
        "theoretical_gain_ceiling_rate": _rate(theoretical_gain_ceiling, n),
    }


def summarize_rows(family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    manifests = sorted({str(row.get("path", "")) for row in rows})
    subtypes = sorted({str(row.get("subtype", "")) for row in rows})
    return {
        "overall": _group_summary(family, rows),
        "by_manifest": {
            manifest: _group_summary(
                family,
                [row for row in rows if str(row.get("path", "")) == manifest],
            )
            for manifest in manifests
        },
        "by_subtype": {
            subtype: _group_summary(
                family,
                [row for row in rows if str(row.get("subtype", "")) == subtype],
            )
            for subtype in subtypes
        },
        "ranker_miss_oracle_hit_examples": [
            _example_payload(row)
            for row in rows
            if str(row.get("risk_class")) == "ranker_miss_oracle_hit"
        ][:20],
        "operator_gap_examples": [
            _example_payload(row)
            for row in rows
            if str(row.get("risk_class")) == "operator_gap_oracle_miss"
        ][:20],
        "operator_gap_clusters": operator_gap_clusters(family, rows),
    }


def operator_gap_clusters(family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    gap_rows = [row for row in rows if str(row.get("risk_class")) == "operator_gap_oracle_miss"]
    if family == "equation_template":
        return _equation_gap_clusters(gap_rows)
    if family == "bit_permutation":
        return _bit_gap_clusters(gap_rows)
    return {"n": len(gap_rows)}


def _equation_gap_clusters(rows: list[dict[str, Any]]) -> dict[str, Any]:
    support_coverage = Counter(
        _coverage_bucket(row.get("ranker_support_key_coverage"))
        for row in rows
    )
    provenance = Counter(
        "unseen_target_literal" if int(row.get("provenance_unseen") or 0) > 0 else "target_literals_seen"
        for row in rows
    )
    expressibility = Counter(
        "target_expressible" if bool(row.get("target_expressible")) else "target_not_expressible"
        for row in rows
    )
    query_key = Counter(
        "seen_query_key" if bool(row.get("ranker_query_key_seen_any")) else "unseen_query_key"
        for row in rows
    )
    literal_reuse = Counter(
        "literal_reuse_risk" if bool(row.get("ranker_literal_reuse_risk")) else "no_literal_reuse_risk"
        for row in rows
    )
    return {
        "n": len(rows),
        "support_key_coverage": dict(sorted(support_coverage.items())),
        "target_literal_provenance": dict(sorted(provenance.items())),
        "target_expressibility": dict(sorted(expressibility.items())),
        "query_key_seen": dict(sorted(query_key.items())),
        "literal_reuse": dict(sorted(literal_reuse.items())),
    }


def _bit_gap_clusters(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "top_operator_family": _counter_dict(row.get("top_operator_family", "unknown") for row in rows),
        "oracle_operator_family": _counter_dict(row.get("oracle_operator_family", "unknown") for row in rows),
        "oracle_rank_bucket": _counter_dict(row.get("oracle_rank_bucket", "miss") for row in rows),
        "support_stability": _counter_dict(row.get("support_leave_one_out_stability", "unknown") for row in rows),
        "top_hamming_to_target": _counter_dict(_hamming_bucket(row.get("top_hamming_to_target")) for row in rows),
        "top_oracle_hamming": _counter_dict(_hamming_bucket(row.get("top_oracle_hamming")) for row in rows),
        "top_complexity_penalty": _counter_dict(_penalty_bucket(row.get("top_complexity_penalty")) for row in rows),
        "top_expression_complexity": _counter_dict(
            _complexity_bucket(row.get("top_expression_complexity_total")) for row in rows
        ),
    }


def _counter_dict(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _coverage_bucket(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if numeric >= 0.99:
        return "1.00"
    if numeric >= 0.75:
        return "0.75-0.99"
    if numeric >= 0.50:
        return "0.50-0.75"
    if numeric > 0.0:
        return "0.01-0.50"
    return "0"


def _hamming_bucket(value: Any) -> str:
    if value in ("", None):
        return "unknown"
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if numeric == 0:
        return "0"
    if numeric == 1:
        return "1"
    if numeric <= 3:
        return "2-3"
    return "4+"


def _penalty_bucket(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if numeric <= 0.1:
        return "0-0.10"
    if numeric <= 0.35:
        return "0.10-0.35"
    if numeric <= 0.75:
        return "0.35-0.75"
    return "0.75+"


def _complexity_bucket(value: Any) -> str:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if numeric <= 0:
        return "0"
    if numeric <= 4:
        return "1-4"
    if numeric <= 8:
        return "5-8"
    return "9+"


def _example_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "path": row.get("path"),
        "subtype": row.get("subtype"),
        "risk_class": row.get("risk_class"),
        "oracle_rank": row.get("oracle_rank"),
        "query": row.get("query"),
        "target": row.get("target"),
        "top_prediction": row.get("top_prediction"),
        "top_steps": row.get("top_steps", ""),
        "top_oracle_hamming": row.get("top_oracle_hamming", ""),
        "top_operator_family": row.get("top_operator_family", ""),
    }


def collect_diagnostics(
    inputs: list[Path],
    *,
    limit: int | None,
    top_k_equation: int,
    top_k_bit: int,
    bit_beam_width: int,
    bit_max_depth: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    equation_engine = ChainSearchEngine(beam_width=24, max_depth=4)
    bit_engine = ChainSearchEngine(beam_width=bit_beam_width, max_depth=bit_max_depth)
    equation_rows: list[dict[str, Any]] = []
    bit_rows: list[dict[str, Any]] = []
    for path in inputs:
        source_rows = load_bit_jsonl(path)
        if limit is not None and limit >= 0:
            source_rows = source_rows[:limit]
        for row in source_rows:
            equation = diagnose_equation_example(equation_engine, row, top_k=top_k_equation)
            if equation is not None:
                equation["path"] = str(path)
                equation_rows.append(equation)
            bit = diagnose_bit_example(bit_engine, row, top_k=top_k_bit)
            if bit is not None:
                bit["path"] = str(path)
                bit_rows.append(bit)
    return equation_rows, bit_rows


def build_report(
    *,
    inputs: list[Path],
    limit: int | None,
    top_k_equation: int,
    top_k_bit: int,
    bit_beam_width: int,
    bit_max_depth: int,
) -> dict[str, Any]:
    equation_rows, bit_rows = collect_diagnostics(
        inputs,
        limit=limit,
        top_k_equation=top_k_equation,
        top_k_bit=top_k_bit,
        bit_beam_width=bit_beam_width,
        bit_max_depth=bit_max_depth,
    )
    return {
        "schema_version": 1,
        "settings": {
            "inputs": [path.as_posix() for path in inputs],
            "limit": limit,
            "top_k_equation": top_k_equation,
            "top_k_bit": top_k_bit,
            "bit_beam_width": bit_beam_width,
            "bit_max_depth": bit_max_depth,
        },
        "equation_template": summarize_rows("equation_template", equation_rows),
        "bit_permutation": summarize_rows("bit_permutation", bit_rows),
        "rows": {
            "equation_template": equation_rows,
            "bit_permutation": bit_rows,
        },
    }


def write_report(output_dir: Path, output_md: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", _summary_only(report))
    _write_json(output_dir / "equation_template_rows.json", report["rows"]["equation_template"])
    _write_json(output_dir / "bit_permutation_rows.json", report["rows"]["bit_permutation"])
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(report), encoding="utf-8")


def _summary_only(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report["schema_version"],
        "settings": report["settings"],
        "equation_template": report["equation_template"],
        "bit_permutation": report["bit_permutation"],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _md_literal(value: Any) -> str:
    return json.dumps("" if value is None else str(value), ensure_ascii=False)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Solver Breakout v2",
        "",
        "CPU-only upper-bound and ranker-gap report for weak families.",
        "",
        "| family | n | top1_acc | oracle@k | safe_override | ranker_miss | operator_gap | gain_ceiling |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in ("equation_template", "bit_permutation"):
        overall = report[family]["overall"]
        lines.append(
            f"| {family} | {overall['n']} | {overall['top1_accuracy']:.4f} | "
            f"{overall['oracle_at_k']:.4f} | {overall['safe_override_possible_rate']:.4f} | "
            f"{overall['ranker_miss_oracle_hit_count']} | {overall['operator_gap_oracle_miss_count']} | "
            f"{overall['theoretical_gain_ceiling_rate']:.4f} |"
        )

    for family in ("equation_template", "bit_permutation"):
        lines.extend(["", f"## {family}", ""])
        for manifest, summary in report[family]["by_manifest"].items():
            lines.append(
                f"- `{manifest}`: n=`{summary['n']}`, top1=`{summary['top1_accuracy']:.4f}`, "
                f"oracle@k=`{summary['oracle_at_k']:.4f}`, gain_ceiling=`{summary['theoretical_gain_ceiling_rate']:.4f}`"
            )
        lines.extend(["", "### Ranker Miss Examples", ""])
        for row in report[family]["ranker_miss_oracle_hit_examples"][:10]:
            lines.append(
                f"- `{row['id']}` oracle_rank=`{row['oracle_rank']}` "
                f"target={_md_literal(row['target'])} top={_md_literal(row['top_prediction'])} "
                f"family=`{row.get('top_operator_family', '')}`"
            )
        lines.extend(["", "### Operator Gap Examples", ""])
        for row in report[family]["operator_gap_examples"][:10]:
            lines.append(
                f"- `{row['id']}` target={_md_literal(row['target'])} "
                f"top={_md_literal(row['top_prediction'])} "
                f"risk=`{row['risk_class']}`"
            )
        clusters = report[family].get("operator_gap_clusters", {})
        lines.extend(["", "### Operator Gap Clusters", ""])
        if not clusters or int(clusters.get("n", 0) or 0) == 0:
            lines.append("- no operator-gap rows")
        else:
            for key, value in clusters.items():
                if key == "n":
                    continue
                lines.append(f"- `{key}`: `{json.dumps(value, ensure_ascii=False, sort_keys=True)}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CPU-only solver breakout v2 diagnostics.")
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--limit", type=int, help="Optional per-manifest raw-row limit; use 0 for schema smoke.")
    parser.add_argument("--top-k-equation", type=int, default=10)
    parser.add_argument("--top-k-bit", type=int, default=5)
    parser.add_argument("--bit-beam-width", type=int, default=12)
    parser.add_argument("--bit-max-depth", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args(argv)

    inputs = args.input or list(dict.fromkeys([*EQUATION_DEFAULT_INPUTS, *BIT_DEFAULT_INPUTS]))
    report = build_report(
        inputs=inputs,
        limit=args.limit,
        top_k_equation=args.top_k_equation,
        top_k_bit=args.top_k_bit,
        bit_beam_width=args.bit_beam_width,
        bit_max_depth=args.bit_max_depth,
    )
    write_report(args.output_dir, args.output_md, report)
    print(
        json.dumps(
            {
                "status": "done",
                "output_dir": args.output_dir.as_posix(),
                "output_md": args.output_md.as_posix(),
                "equation_rows": len(report["rows"]["equation_template"]),
                "bit_rows": len(report["rows"]["bit_permutation"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
