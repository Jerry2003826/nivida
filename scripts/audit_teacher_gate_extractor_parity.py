from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_equation_family import ANSWER_TYPES, classify_answer_type
from src.common.io import load_jsonl, write_json
from src.competition.metrics import competition_correct as _repo_verify_bidirectional
from src.competition.official_metric_contract import (
    current_contract_fingerprint,
    verify as _official_verify,
)
from src.competition.schema import PuzzleExample
from src.teacher.chain_search import ChainSearchEngine
from src.teacher.program_signature import annotate_example_from_candidates


STRICT_CONF_MIN = 0.80
STRICT_COVERAGE_MIN = 1.0
SILVER_CONF_MIN = 0.65
SILVER_COVERAGE_MIN = 0.67
HARD_TRIAD_FAMILIES: frozenset[str] = frozenset({"bit", "cipher", "equation"})
KNOWN_SYNTH_SOURCES: tuple[str, ...] = (
    "chain_search",
    "program_signature",
    "pseudo_labeler",
    "hardcase_miner",
    "synth_generator",
    "unknown",
)
CANONICAL_CHAIN_SEARCH = {
    "beam_width": 8,
    "max_depth": 2,
    "top_k": 2,
}


@dataclass(slots=True)
class EvaluatedExample:
    id: str
    source_pool: str
    family: str
    answer_type: str
    synth_source: str
    annotation_source: str
    target_answer: str
    query_prediction: str | None
    program_signature: str | None
    teacher_confidence: float
    repo_support_coverage: float
    official_support_coverage: float
    repo_solver_verifiable: bool
    official_solver_verifiable: bool
    strict_repo: bool
    strict_official: bool
    silver_repo: bool
    silver_official: bool
    synth_repo: bool
    synth_official: bool
    mismatched_support_pairs: list[dict[str, Any]]


def _normalize_family(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) or {}
    return str(
        row.get("official_family")
        or row.get("family")
        or metadata.get("official_family")
        or metadata.get("family")
        or "unknown"
    )


def _normalize_program_signature(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata", {}) or {}
    value = row.get("program_signature")
    if value is None:
        value = metadata.get("program_signature")
    if value in (None, ""):
        return None
    return str(value)


def _normalize_teacher_confidence(row: dict[str, Any]) -> float:
    metadata = row.get("metadata", {}) or {}
    value = row.get("teacher_confidence")
    if value is None:
        value = metadata.get("teacher_confidence")
    return float(value or 0.0)


def _normalize_synth_source(row: dict[str, Any], input_path: str | Path) -> str:
    metadata = row.get("metadata", {}) or {}
    raw_source = str(
        metadata.get("source")
        or metadata.get("extras", {}).get("source")
        or ""
    ).strip().lower()
    alias_map = {
        "synthetic": "synth_generator",
        "synth": "synth_generator",
    }
    if raw_source in alias_map:
        raw_source = alias_map[raw_source]
    if raw_source in KNOWN_SYNTH_SOURCES:
        return raw_source

    lowered_name = str(input_path).lower()
    for candidate in KNOWN_SYNTH_SOURCES:
        if candidate == "unknown":
            continue
        if candidate in lowered_name:
            return candidate
    if "synth" in lowered_name:
        return "synth_generator"
    return "unknown"


def _support_pairs_from_row(row: dict[str, Any]) -> tuple[list[dict[str, str]] | None, str | None]:
    metadata = row.get("metadata", {}) or {}
    extras = metadata.get("extras", {}) or {}
    support_pairs = extras.get("support_pairs")
    if isinstance(support_pairs, list):
        normalized: list[dict[str, str]] = []
        for pair in support_pairs:
            if not isinstance(pair, dict):
                continue
            normalized.append(
                {
                    "input": str(pair.get("input", "")),
                    "target": str(pair.get("target", "")),
                    "prediction": str(pair.get("prediction", "")),
                }
            )
        return normalized, None if extras.get("query_prediction") is None else str(extras.get("query_prediction"))

    top_candidates = extras.get("chain_search_top_k") or extras.get("top_candidates")
    parsed_examples = row.get("parsed_examples") or row.get("train_pairs") or []
    if isinstance(top_candidates, list) and top_candidates:
        candidate = top_candidates[0]
        if isinstance(candidate, dict):
            predictions = candidate.get("predictions") or candidate.get("support_predictions")
            if isinstance(predictions, list) and len(predictions) == len(parsed_examples):
                normalized = []
                for pair, prediction in zip(parsed_examples, predictions):
                    if not isinstance(pair, dict):
                        continue
                    normalized.append(
                        {
                            "input": str(pair.get("input", "")),
                            "target": str(pair.get("output", pair.get("target", ""))),
                            "prediction": str(prediction),
                        }
                    )
                query_prediction = candidate.get("query_prediction")
                return normalized, None if query_prediction is None else str(query_prediction)
    return None, None


def _rerun_annotation(row: dict[str, Any]) -> tuple[list[dict[str, str]], str | None, str | None, float]:
    example = PuzzleExample.from_dict(row)
    engine = ChainSearchEngine(
        beam_width=CANONICAL_CHAIN_SEARCH["beam_width"],
        max_depth=CANONICAL_CHAIN_SEARCH["max_depth"],
    )
    candidates = engine.solve_example(example, top_k=CANONICAL_CHAIN_SEARCH["top_k"])
    annotate_example_from_candidates(example, candidates)
    top = candidates[0] if candidates else None
    predictions = [] if top is None else list(top.predictions)
    support_pairs = [
        {
            "input": pair.input,
            "target": pair.output,
            "prediction": str(prediction),
        }
        for pair, prediction in zip(example.parsed_examples, predictions)
    ]
    query_prediction = None if top is None or top.query_prediction is None else str(top.query_prediction)
    return (
        support_pairs,
        query_prediction,
        example.metadata.program_signature,
        float(example.metadata.teacher_confidence or 0.0),
    )


def _strict_gate(
    *,
    teacher_confidence: float,
    support_coverage: float,
    solver_verifiable: bool,
    program_signature: str | None,
) -> bool:
    return (
        teacher_confidence >= STRICT_CONF_MIN
        and solver_verifiable
        and support_coverage >= STRICT_COVERAGE_MIN
    )


def _silver_gate(
    *,
    family: str,
    teacher_confidence: float,
    support_coverage: float,
    solver_verifiable: bool,
    program_signature: str | None,
) -> bool:
    return (
        family in HARD_TRIAD_FAMILIES
        and teacher_confidence >= SILVER_CONF_MIN
        and support_coverage >= SILVER_COVERAGE_MIN
        and (program_signature is not None or solver_verifiable)
    )


def _synth_gate(
    *,
    target_answer: str,
    query_prediction: str | None,
    solver_verifiable: bool,
    program_signature: str | None,
) -> bool:
    return bool(target_answer) and query_prediction is not None and solver_verifiable and program_signature is not None


def _rate(numerator: int, denominator: int, *, null_if_zero: bool = False) -> float | None:
    if denominator == 0:
        return None if null_if_zero else 0.0
    return numerator / denominator


def _group_stats(rows: list[EvaluatedExample]) -> dict[str, Any]:
    strict_base = [row for row in rows if row.source_pool == "train" and row.strict_repo]
    strict_repo_only = [row for row in strict_base if not row.strict_official]
    silver_base = [
        row
        for row in rows
        if row.source_pool == "train" and row.silver_repo and not row.strict_repo
    ]
    silver_repo_only = [row for row in silver_base if not row.silver_official]
    synth_base = [row for row in rows if row.source_pool == "synth" and row.synth_repo]
    synth_repo_only = [row for row in synth_base if not row.synth_official]
    return {
        "count": len(rows),
        "strict_repo_accepted": len(strict_base),
        "strict_pollution_rate": _rate(len(strict_repo_only), len(strict_base), null_if_zero=True),
        "silver_pollution_rate": _rate(len(silver_repo_only), len(silver_base), null_if_zero=True),
        "synth_pollution_rate": _rate(len(synth_repo_only), len(synth_base), null_if_zero=True),
    }


def _example_payload(example: EvaluatedExample) -> dict[str, Any]:
    return {
        "id": example.id,
        "family": example.family,
        "answer_type": example.answer_type,
        "target_answer": example.target_answer,
        "query_prediction": example.query_prediction,
        "repo_support_coverage": example.repo_support_coverage,
        "official_support_coverage": example.official_support_coverage,
        "repo_solver_verifiable": example.repo_solver_verifiable,
        "official_solver_verifiable": example.official_solver_verifiable,
        "program_signature": example.program_signature,
        "teacher_confidence": example.teacher_confidence,
        "mismatched_support_pairs": example.mismatched_support_pairs,
    }


def _stratified_limit(rows: list[EvaluatedExample], limit: int) -> list[EvaluatedExample]:
    buckets: dict[tuple[str, str, str], list[EvaluatedExample]] = {}
    for row in sorted(rows, key=lambda item: (item.family, item.answer_type, item.synth_source, item.id)):
        key = (row.family, row.answer_type, row.synth_source)
        buckets.setdefault(key, []).append(row)

    selected: list[EvaluatedExample] = []
    indices = {key: 0 for key in buckets}
    ordered_keys = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for key in ordered_keys:
            bucket = buckets[key]
            idx = indices[key]
            if idx >= len(bucket):
                continue
            selected.append(bucket[idx])
            indices[key] += 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def _decision_hint(strict_pollution_rate: float) -> dict[str, Any]:
    if strict_pollution_rate > 0.02:
        action = "rebuild_stage2"
        rationale = f"strict_pollution_rate ({strict_pollution_rate:.3f}) > 0.02 threshold"
    elif strict_pollution_rate >= 0.005:
        action = "family_targeted_fix"
        rationale = f"strict_pollution_rate ({strict_pollution_rate:.3f}) is between 0.005 and 0.02"
    else:
        action = "record_only"
        rationale = f"strict_pollution_rate ({strict_pollution_rate:.3f}) < 0.005 threshold"
    return {
        "global_strict_pollution_rate": strict_pollution_rate,
        "action": action,
        "action_rationale": rationale,
        "action_choices": {
            "below_0p5_percent": "record_only",
            "0p5_to_2_percent": "family_targeted_fix",
            "above_2_percent": "rebuild_stage2",
        },
    }


def _evaluate_rows(
    *,
    rows: list[dict[str, Any]],
    source_pool: str,
    input_path: str | Path,
    allow_rerun_chain_search: bool,
) -> list[EvaluatedExample]:
    evaluated: list[EvaluatedExample] = []
    for row in rows:
        family = _normalize_family(row)
        target_answer = str(row.get("target_answer", ""))
        program_signature = _normalize_program_signature(row)
        teacher_confidence = _normalize_teacher_confidence(row)

        support_pairs, query_prediction = _support_pairs_from_row(row)
        annotation_source = "cached"
        if support_pairs is None:
            if not allow_rerun_chain_search:
                raise ValueError(
                    "Missing support_pairs/query_prediction in metadata.extras. "
                    "Re-run with --allow-rerun-chain-search to rebuild annotations."
                )
            support_pairs, query_prediction, program_signature, teacher_confidence = _rerun_annotation(row)
            annotation_source = "rerun"

        repo_matches: list[bool] = []
        official_matches: list[bool] = []
        mismatched_support_pairs: list[dict[str, Any]] = []
        for pair in support_pairs:
            prediction = str(pair.get("prediction", ""))
            target = str(pair.get("target", ""))
            repo_match = _repo_verify_bidirectional(prediction, target)
            official_match = _official_verify(target, prediction)
            repo_matches.append(repo_match)
            official_matches.append(official_match)
            if repo_match != official_match:
                mismatched_support_pairs.append(
                    {
                        "input": str(pair.get("input", "")),
                        "target": target,
                        "prediction": prediction,
                        "repo_match": repo_match,
                        "official_match": official_match,
                    }
                )

        repo_support_coverage = _rate(sum(repo_matches), len(repo_matches)) or 0.0
        official_support_coverage = _rate(sum(official_matches), len(official_matches)) or 0.0
        repo_solver_verifiable = bool(repo_matches) and all(repo_matches) and query_prediction is not None
        official_solver_verifiable = bool(official_matches) and all(official_matches) and query_prediction is not None

        strict_repo = _strict_gate(
            teacher_confidence=teacher_confidence,
            support_coverage=repo_support_coverage,
            solver_verifiable=repo_solver_verifiable,
            program_signature=program_signature,
        )
        strict_official = _strict_gate(
            teacher_confidence=teacher_confidence,
            support_coverage=official_support_coverage,
            solver_verifiable=official_solver_verifiable,
            program_signature=program_signature,
        )
        silver_repo = _silver_gate(
            family=family,
            teacher_confidence=teacher_confidence,
            support_coverage=repo_support_coverage,
            solver_verifiable=repo_solver_verifiable,
            program_signature=program_signature,
        )
        silver_official = _silver_gate(
            family=family,
            teacher_confidence=teacher_confidence,
            support_coverage=official_support_coverage,
            solver_verifiable=official_solver_verifiable,
            program_signature=program_signature,
        )
        synth_repo = _synth_gate(
            target_answer=target_answer,
            query_prediction=query_prediction,
            solver_verifiable=repo_solver_verifiable,
            program_signature=program_signature,
        )
        synth_official = _synth_gate(
            target_answer=target_answer,
            query_prediction=query_prediction,
            solver_verifiable=official_solver_verifiable,
            program_signature=program_signature,
        )

        evaluated.append(
            EvaluatedExample(
                id=str(row.get("id", "")),
                source_pool=source_pool,
                family=family,
                answer_type=classify_answer_type(target_answer),
                synth_source=_normalize_synth_source(row, input_path),
                annotation_source=annotation_source,
                target_answer=target_answer,
                query_prediction=query_prediction,
                program_signature=program_signature,
                teacher_confidence=teacher_confidence,
                repo_support_coverage=float(repo_support_coverage),
                official_support_coverage=float(official_support_coverage),
                repo_solver_verifiable=repo_solver_verifiable,
                official_solver_verifiable=official_solver_verifiable,
                strict_repo=strict_repo,
                strict_official=strict_official,
                silver_repo=silver_repo,
                silver_official=silver_official,
                synth_repo=synth_repo,
                synth_official=synth_official,
                mismatched_support_pairs=mismatched_support_pairs,
            )
        )
    return evaluated


def run_audit(
    *,
    train_jsonl: str | Path,
    synth_jsonl: str | Path,
    stage2_report: str | Path | None,
    output: str | Path,
    samples_to_include: int,
    allow_rerun_chain_search: bool,
) -> dict[str, Any]:
    train_path = Path(train_jsonl)
    if not train_path.exists():
        print(
            "Pending: stage2 annotated teacher JSONL is missing. "
            "Generate stage2_official_train_no_hard_valid.jsonl and rerun this audit."
        )
        payload = {"status": "pending_stage2_teacher_annotation"}
        write_json(output, payload)
        return payload

    synth_path = Path(synth_jsonl)
    train_rows = load_jsonl(train_path)
    synth_rows = load_jsonl(synth_path) if synth_path.exists() else []

    evaluated = _evaluate_rows(
        rows=train_rows,
        source_pool="train",
        input_path=train_path,
        allow_rerun_chain_search=allow_rerun_chain_search,
    ) + _evaluate_rows(
        rows=synth_rows,
        source_pool="synth",
        input_path=synth_path,
        allow_rerun_chain_search=allow_rerun_chain_search,
    )

    train_examples = [row for row in evaluated if row.source_pool == "train"]
    synth_examples = [row for row in evaluated if row.source_pool == "synth"]

    strict_repo_base = [row for row in train_examples if row.strict_repo]
    strict_repo_only = [row for row in strict_repo_base if not row.strict_official]
    silver_repo_base = [
        row
        for row in train_examples
        if row.silver_repo and not row.strict_repo
    ]
    silver_repo_only = [row for row in silver_repo_base if not row.silver_official]
    synth_repo_base = [row for row in synth_examples if row.synth_repo]
    synth_repo_only = [row for row in synth_repo_base if not row.synth_official]

    family_keys = ("bit", "gravity", "unit", "cipher", "numeral", "equation")
    by_family = {
        family: _group_stats([row for row in evaluated if row.family == family])
        for family in family_keys
    }

    by_synth_source = {
        source: _group_stats([row for row in evaluated if row.synth_source == source])
        for source in KNOWN_SYNTH_SOURCES
        if any(row.synth_source == source for row in evaluated)
    }

    by_answer_type = {
        answer_type: _group_stats([row for row in evaluated if row.answer_type == answer_type])
        for answer_type in ANSWER_TYPES
    }

    by_annotation_source = {
        annotation_source: _group_stats(
            [row for row in evaluated if row.annotation_source == annotation_source]
        )
        for annotation_source in ("cached", "rerun")
    }

    strict_pollution_rate = _rate(len(strict_repo_only), len(strict_repo_base)) or 0.0
    payload = {
        "inputs": {
            "train_jsonl": str(train_path),
            "synth_jsonl": str(synth_path),
            "stage2_report": None if stage2_report is None else str(stage2_report),
            "contract_fingerprint": current_contract_fingerprint().to_dict(),
        },
        "thresholds": {
            "strict": {"conf_min": STRICT_CONF_MIN, "coverage_min": STRICT_COVERAGE_MIN},
            "silver": {"conf_min": SILVER_CONF_MIN, "coverage_min": SILVER_COVERAGE_MIN},
        },
        "summary": {
            "num_train_examples": len(train_examples),
            "num_synth_examples": len(synth_examples),
            "strict_repo_accepted": len(strict_repo_base),
            "strict_official_accepted": sum(row.strict_official for row in train_examples),
            "strict_repo_only": len(strict_repo_only),
            "strict_pollution_rate": strict_pollution_rate,
            "silver_repo_accepted": len(silver_repo_base),
            "silver_official_accepted": sum(
                row.silver_official and not row.strict_repo
                for row in train_examples
            ),
            "silver_repo_only": len(silver_repo_only),
            "silver_pollution_rate": _rate(len(silver_repo_only), len(silver_repo_base)) or 0.0,
            "synth_repo_accepted": len(synth_repo_base),
            "synth_official_accepted": sum(row.synth_official for row in synth_examples),
            "synth_repo_only": len(synth_repo_only),
            "synth_pollution_rate": _rate(len(synth_repo_only), len(synth_repo_base)) or 0.0,
        },
        "by_family": by_family,
        "by_synth_source": by_synth_source,
        "by_answer_type": by_answer_type,
        "by_annotation_source": by_annotation_source,
        "decision_hint": _decision_hint(strict_pollution_rate),
        "examples": {
            "strict_repo_only": [
                _example_payload(row)
                for row in _stratified_limit(strict_repo_only, samples_to_include)
            ],
            "silver_repo_only": [
                _example_payload(row)
                for row in _stratified_limit(silver_repo_only, samples_to_include)
            ],
            "synth_repo_only": [
                _example_payload(row)
                for row in _stratified_limit(synth_repo_only, samples_to_include)
            ],
        },
    }
    write_json(output, payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit repo-vs-official teacher gate parity.")
    parser.add_argument("--train-jsonl", default="data/processed/stage2_official_train_no_hard_valid.jsonl")
    parser.add_argument("--synth-jsonl", default="data/synthetic/synth_hard_triads.jsonl")
    parser.add_argument("--stage2-report")
    parser.add_argument("--output", default="data/processed/audit_teacher_gate_extractor_parity.json")
    parser.add_argument("--samples-to-include", type=int, default=50)
    parser.add_argument("--allow-rerun-chain-search", action="store_true")
    parser.add_argument("--strict-strict-gate", action="store_true", default=True)
    args = parser.parse_args(argv)

    run_audit(
        train_jsonl=args.train_jsonl,
        synth_jsonl=args.synth_jsonl,
        stage2_report=args.stage2_report,
        output=args.output,
        samples_to_include=max(0, int(args.samples_to_include)),
        allow_rerun_chain_search=bool(args.allow_rerun_chain_search),
    )


if __name__ == "__main__":
    main()
