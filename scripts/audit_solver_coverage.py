from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition.metrics import competition_correct  # noqa: E402
from src.competition.schema import PuzzleExample  # noqa: E402
from src.teacher.chain_search import ChainSearchEngine  # noqa: E402
from src.teacher.program_signature import canonicalize_candidate  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _support_accuracy(candidate: Any, example: PuzzleExample) -> float:
    if candidate is None or len(candidate.predictions) != len(example.parsed_examples):
        return 0.0
    checks = [
        competition_correct(prediction, pair.output)
        for prediction, pair in zip(candidate.predictions, example.parsed_examples)
    ]
    return _rate(checks)


def _classify(
    *,
    candidate: Any | None,
    support_accuracy: float,
    query_correct: bool,
    target: str | None,
) -> str:
    if candidate is None:
        return "no_candidate"
    if support_accuracy < 1.0:
        return "support_incomplete"
    if getattr(candidate, "query_prediction", None) is None:
        return "no_query_prediction"
    if target in (None, ""):
        return "missing_target"
    if query_correct:
        return "query_correct"
    return "query_wrong_after_support_fit"


def _audit_example(engine: ChainSearchEngine, row: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    example = PuzzleExample.from_dict(row)
    family = example.metadata.official_family or "unknown"
    subtype = example.metadata.subtype or "unknown"
    candidates = engine.solve_example(example, top_k=top_k)
    top = candidates[0] if candidates else None
    support_acc = _support_accuracy(top, example)
    query_prediction = "" if top is None or top.query_prediction is None else str(top.query_prediction)
    target = example.target_answer
    query_ok = False if target is None else competition_correct(query_prediction, target)
    signature = ""
    signature_bucket = ""
    steps = ""
    confidence = 0.0
    score = 0.0
    exact_ratio = 0.0
    if top is not None:
        program = canonicalize_candidate(top, family, subtype=subtype)
        signature = program.signature
        signature_bucket = program.signature_bucket
        steps = ">".join(program.steps)
        confidence = float(top.confidence)
        score = float(top.score)
        exact_ratio = float(top.exact_ratio)
    failure_class = _classify(
        candidate=top,
        support_accuracy=support_acc,
        query_correct=query_ok,
        target=target,
    )
    return {
        "id": example.id,
        "family": family,
        "subtype": subtype,
        "target": "" if target is None else str(target),
        "query": example.query,
        "query_prediction": query_prediction,
        "query_correct": query_ok,
        "support_accuracy": support_acc,
        "candidate_exact_ratio": exact_ratio,
        "confidence": confidence,
        "score": score,
        "failure_class": failure_class,
        "signature": signature,
        "signature_bucket": signature_bucket,
        "steps": steps,
        "num_pairs": len(example.parsed_examples),
    }


def _group_summary(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[str(record.get(key, "unknown"))].append(record)
    summary = {}
    for name, rows in sorted(buckets.items()):
        summary[name] = {
            "n": len(rows),
            "query_accuracy": _rate([bool(row["query_correct"]) for row in rows]),
            "support_full_rate": _rate([float(row["support_accuracy"]) >= 1.0 for row in rows]),
            "avg_support_accuracy": sum(float(row["support_accuracy"]) for row in rows) / len(rows),
            "failure_classes": dict(Counter(str(row["failure_class"]) for row in rows).most_common()),
            "top_signature_buckets": dict(Counter(str(row["signature_bucket"]) for row in rows if row.get("signature_bucket")).most_common(8)),
        }
    return summary


def _audit_file(
    path: Path,
    *,
    beam_width: int,
    max_depth: int,
    top_k: int,
    max_rows: int | None,
) -> dict[str, Any]:
    rows = _load_jsonl(path)
    if max_rows is not None:
        rows = rows[:max_rows]
    engine = ChainSearchEngine(beam_width=beam_width, max_depth=max_depth)
    records = [_audit_example(engine, row, top_k=top_k) for row in rows]
    return {
        "path": str(path),
        "settings": {
            "beam_width": beam_width,
            "max_depth": max_depth,
            "top_k": top_k,
            "max_rows": max_rows,
        },
        "overall": {
            "n": len(records),
            "query_accuracy": _rate([bool(row["query_correct"]) for row in records]),
            "support_full_rate": _rate([float(row["support_accuracy"]) >= 1.0 for row in records]),
            "avg_support_accuracy": sum(float(row["support_accuracy"]) for row in records) / len(records) if records else 0.0,
            "failure_classes": dict(Counter(str(row["failure_class"]) for row in records).most_common()),
        },
        "family": _group_summary(records, "family"),
        "subtype": _group_summary(records, "subtype"),
        "records": records,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Solver Coverage Audit",
        "",
        "This measures how much of the labeled local eval set is already explainable by the rule/search system.",
        "",
    ]
    for report in payload["reports"]:
        overall = report["overall"]
        lines.extend(
            [
                f"## `{report['path']}`",
                "",
                f"- rows: `{overall['n']}`",
                f"- query accuracy: `{overall['query_accuracy']:.4f}`",
                f"- support-full rate: `{overall['support_full_rate']:.4f}`",
                f"- avg support accuracy: `{overall['avg_support_accuracy']:.4f}`",
                f"- failure classes: `{json.dumps(overall['failure_classes'], ensure_ascii=False)}`",
                "",
                "| family | n | query_acc | support_full | avg_support | failures | top signature buckets |",
                "| --- | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for family, row in report["family"].items():
            lines.append(
                f"| {family} | {row['n']} | {row['query_accuracy']:.4f} | "
                f"{row['support_full_rate']:.4f} | {row['avg_support_accuracy']:.4f} | "
                f"`{json.dumps(row['failure_classes'], ensure_ascii=False)}` | "
                f"`{json.dumps(row['top_signature_buckets'], ensure_ascii=False)}` |"
            )
        lines.extend(["", "### Weakest Subtypes", ""])
        subtype_rows = sorted(
            report["subtype"].items(),
            key=lambda item: (item[1]["query_accuracy"], -item[1]["n"], item[0]),
        )
        lines.extend(
            [
                "| subtype | n | query_acc | support_full | failures |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for subtype, row in subtype_rows[:12]:
            lines.append(
                f"| {subtype} | {row['n']} | {row['query_accuracy']:.4f} | "
                f"{row['support_full_rate']:.4f} | "
                f"`{json.dumps(row['failure_classes'], ensure_ascii=False)}` |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit rule/search solver coverage by family and subtype.")
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="Labeled JSONL manifest. Repeatable. Defaults to the three local eval manifests.",
    )
    parser.add_argument("--beam-width", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output-json", type=Path, default=Path("data/processed/solver_coverage_audit.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/solver_coverage_records.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/solver_coverage_audit_latest.md"))
    args = parser.parse_args()

    inputs = args.input or [
        Path("data/processed/local_eval_manifests/combined_balanced_48pf.jsonl"),
        Path("data/processed/local_eval_manifests/proxy_all_balanced_64pf.jsonl"),
        Path("data/processed/local_eval_manifests/hard_triad_full.jsonl"),
    ]
    reports = [
        _audit_file(
            path,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            top_k=args.top_k,
            max_rows=args.max_rows,
        )
        for path in inputs
        if path.is_file()
    ]
    if not reports:
        raise SystemExit("No input files found.")
    payload = {"reports": reports}
    _write_json(args.output_json, payload)
    csv_rows = []
    for report in reports:
        for row in report["records"]:
            csv_rows.append({"path": report["path"], **row})
    _write_csv(args.output_csv, csv_rows)
    _write_markdown(args.output_md, payload)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "output_md": str(args.output_md),
                "files": len(reports),
                "records": len(csv_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
