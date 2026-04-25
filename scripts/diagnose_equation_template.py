from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition.metrics import competition_correct  # noqa: E402
from src.competition.schema import PuzzleExample  # noqa: E402
from src.teacher.atomic_ops import OperatorTemplateOp, PositionTransducerOp  # noqa: E402
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


def target_char_provenance(row_text: dict[str, Any]) -> dict[str, int]:
    query = str(row_text.get("query", ""))
    support_inputs = "".join(str(value) for value in row_text.get("support_inputs", []))
    support_outputs = "".join(str(value) for value in row_text.get("support_outputs", []))
    target = str(row_text.get("target", ""))
    counts = {
        "from_query": 0,
        "from_support_inputs": 0,
        "from_support_outputs": 0,
        "unseen": 0,
    }
    for char in target:
        if char in query:
            counts["from_query"] += 1
        elif char in support_inputs:
            counts["from_support_inputs"] += 1
        elif char in support_outputs:
            counts["from_support_outputs"] += 1
        else:
            counts["unseen"] += 1
    return counts


def classify_template_risk(
    *,
    oracle_rank: int | None,
    ambiguity_count: int,
    has_unseen_literal: bool,
    support_full: bool,
    target_expressible: bool = False,
    target_uses_unseen_query_key: bool = False,
) -> str:
    if not support_full:
        return "operator_gap_oracle_miss"
    if oracle_rank == 1 and ambiguity_count <= 3 and not has_unseen_literal:
        return "low_risk_support_stable"
    if oracle_rank is not None and oracle_rank > 1:
        return "ranker_miss_oracle_hit"
    if has_unseen_literal:
        return "unseen_literal_high_risk"
    if target_uses_unseen_query_key:
        return "unseen_key_template_miss"
    if target_expressible:
        return "expressible_oracle_miss"
    return "operator_gap_oracle_miss"


def _support_full(candidate: Any, example: PuzzleExample) -> bool:
    if candidate is None or len(candidate.predictions) != len(example.parsed_examples):
        return False
    return all(
        competition_correct(prediction, pair.output)
        for prediction, pair in zip(candidate.predictions, example.parsed_examples)
    )


def target_expressibility(
    *,
    examples: list[tuple[str, str]],
    query: str,
    target: str,
) -> dict[str, bool]:
    augmented_examples = [*examples, (query, target)]
    template_params = OperatorTemplateOp().candidate_params(augmented_examples)
    seen_query_key = False
    if template_params:
        for params in template_params:
            key_position = int(params["key_position"])
            if key_position >= len(query):
                continue
            support_key_chars = {
                input_text[key_position]
                for input_text, _ in examples
                if key_position < len(input_text)
            }
            if query[key_position] in support_key_chars:
                seen_query_key = True
                break
    operator_template = bool(template_params)
    position_transducer = bool(PositionTransducerOp().candidate_params(augmented_examples))
    return {
        "operator_template": operator_template,
        "operator_template_seen_query_key": seen_query_key,
        "operator_template_unseen_query_key": operator_template and not seen_query_key,
        "position_transducer": position_transducer,
        "any": operator_template or position_transducer,
    }


def diagnose_example(engine: ChainSearchEngine, row: dict[str, Any], *, top_k: int) -> dict[str, Any] | None:
    example = PuzzleExample.from_dict(row)
    apply_family_tags([example])
    if example.metadata.official_family != "equation" or example.metadata.subtype != "equation_template":
        return None

    candidates = engine.solve_example(example, top_k=top_k)
    target = "" if example.target_answer is None else str(example.target_answer)
    oracle_rank = None
    candidate_rows: list[dict[str, Any]] = []
    support_full_count = 0

    for index, candidate in enumerate(candidates, start=1):
        prediction = "" if candidate.query_prediction is None else str(candidate.query_prediction)
        query_correct = bool(target) and competition_correct(prediction, target)
        candidate_support_full = _support_full(candidate, example)
        if candidate_support_full:
            support_full_count += 1
        if query_correct and oracle_rank is None:
            oracle_rank = index
        candidate_rows.append(
            {
                "rank": index,
                "prediction": prediction,
                "query_correct": query_correct,
                "support_full": candidate_support_full,
                "score": float(candidate.score),
                "exact_ratio": float(candidate.exact_ratio),
                "steps": ">".join(step.op_name for step in candidate.steps),
                "debug": candidate.debug,
            }
        )

    provenance = target_char_provenance(
        {
            "support_inputs": [pair.input for pair in example.parsed_examples],
            "support_outputs": [pair.output for pair in example.parsed_examples],
            "query": example.query,
            "target": target,
        }
    )
    expressibility = target_expressibility(
        examples=[(pair.input, pair.output) for pair in example.parsed_examples],
        query=example.query,
        target=target,
    )
    top_support_full = bool(candidates) and _support_full(candidates[0], example)
    risk = classify_template_risk(
        oracle_rank=oracle_rank,
        ambiguity_count=support_full_count,
        has_unseen_literal=provenance["unseen"] > 0,
        support_full=top_support_full,
        target_expressible=expressibility["any"],
        target_uses_unseen_query_key=expressibility["operator_template_unseen_query_key"],
    )
    return {
        "id": example.id,
        "family": example.metadata.official_family,
        "subtype": example.metadata.subtype,
        "target": target,
        "query": example.query,
        "num_pairs": len(example.parsed_examples),
        "top_prediction": "" if not candidates or candidates[0].query_prediction is None else str(candidates[0].query_prediction),
        "top_query_correct": oracle_rank == 1,
        "oracle_rank": oracle_rank,
        "ambiguity_count": support_full_count,
        "risk_class": risk,
        "target_expressible": expressibility["any"],
        "target_expressible_operator_template": expressibility["operator_template"],
        "target_expressible_operator_template_seen_query_key": expressibility["operator_template_seen_query_key"],
        "target_expressible_operator_template_unseen_query_key": expressibility[
            "operator_template_unseen_query_key"
        ],
        "target_expressible_position_transducer": expressibility["position_transducer"],
        **{f"provenance_{key}": value for key, value in provenance.items()},
        "candidates": candidate_rows,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat_rows = [{key: value for key, value in row.items() if key != "candidates"} for row in rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Equation Template Diagnostic",
        "",
        "| manifest | risk_class | n | top1_acc | oracle_at_k | target_expressible | unseen_literal_rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    manifests = sorted({str(row.get("path", "")) for row in rows})
    for manifest in manifests:
        manifest_rows = [row for row in rows if str(row.get("path", "")) == manifest]
        risk_classes = sorted({str(row["risk_class"]) for row in manifest_rows})
        for risk_class in risk_classes:
            group = [row for row in manifest_rows if row["risk_class"] == risk_class]
            top1 = _rate([bool(row["top_query_correct"]) for row in group])
            oracle = _rate([row["oracle_rank"] is not None for row in group])
            expressible = _rate([bool(row["target_expressible"]) for row in group])
            unseen = sum(int(row["provenance_unseen"]) > 0 for row in group)
            lines.append(
                f"| `{manifest}` | {risk_class} | {len(group)} | {top1:.4f} | "
                f"{oracle:.4f} | {expressible:.4f} | {unseen} |"
            )

    lines.extend(["", "## Target Expressibility", ""])
    expressible_count = sum(bool(row["target_expressible"]) for row in rows)
    template_count = sum(bool(row["target_expressible_operator_template"]) for row in rows)
    position_count = sum(bool(row["target_expressible_position_transducer"]) for row in rows)
    lines.append(f"- current ops can fit support+query target: `{expressible_count} / {len(rows)}`")
    lines.append(f"- via `operator_template`: `{template_count}`")
    seen_key_count = sum(bool(row["target_expressible_operator_template_seen_query_key"]) for row in rows)
    unseen_key_count = sum(bool(row["target_expressible_operator_template_unseen_query_key"]) for row in rows)
    lines.append(f"- via `operator_template` with query key seen in support: `{seen_key_count}`")
    lines.append(f"- via `operator_template` with query key unseen in support: `{unseen_key_count}`")
    lines.append(f"- via `position_transducer`: `{position_count}`")

    lines.extend(["", "## Top Misses", ""])
    misses = [row for row in rows if not row["top_query_correct"]][:20]
    for row in misses:
        lines.append(
            f"- `{row['id']}` risk=`{row['risk_class']}` oracle_rank=`{row['oracle_rank']}` "
            f"target=`{row['target']}` top=`{row['top_prediction']}`"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_diagnostic(inputs: list[Path], *, top_k: int) -> list[dict[str, Any]]:
    engine = ChainSearchEngine(beam_width=24, max_depth=4)
    diagnostics: list[dict[str, Any]] = []
    for path in inputs:
        for row in _load_jsonl(path):
            diagnostic = diagnose_example(engine, row, top_k=top_k)
            if diagnostic is None:
                continue
            diagnostic["path"] = str(path)
            diagnostics.append(diagnostic)
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose equation_template ambiguity and oracle-at-K.")
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=Path("data/processed/equation_template_diagnostic.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/equation_template_diagnostic.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/equation_template_diagnostic_latest.md"))
    args = parser.parse_args()

    rows = run_diagnostic(args.input or DEFAULT_INPUTS, top_k=args.top_k)
    _write_json(args.output_json, {"settings": {"top_k": args.top_k}, "rows": rows})
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
