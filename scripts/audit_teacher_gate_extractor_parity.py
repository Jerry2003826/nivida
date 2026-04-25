from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_equation_family import ANSWER_TYPES, classify_answer_type
from src.common.io import load_jsonl, read_json, write_json
from src.competition.metrics import competition_correct as _repo_verify_bidirectional
from src.competition.official_metric_contract import (
    current_contract_fingerprint,
    verify as _official_verify,
)
from src.competition.schema import PuzzleExample
from src.student.sft_dataset_builder import (
    HARD_TRIAD_FAMILIES,
    _select_official_stage2_silver,
    _select_official_stage2_strict,
    _select_synth_stage2,
)
from src.teacher.chain_search import ChainSearchEngine
from src.teacher.program_signature import annotate_example_from_candidates
from src.teacher.stage2_annotation_provenance import (
    stage2_provenance_matches_local,
)


KNOWN_SYNTH_SOURCES: tuple[str, ...] = (
    "chain_search",
    "program_signature",
    "pseudo_labeler",
    "hardcase_miner",
    "synth_generator",
    "unknown",
)


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
    strict_repo_reason: str | None
    strict_official: bool
    strict_official_reason: str | None
    silver_repo: bool
    silver_official: bool
    synth_repo: bool
    synth_official: bool
    mismatched_support_pairs: list[dict[str, Any]]
    repo_example: PuzzleExample
    official_example: PuzzleExample


def _metadata_value(example: PuzzleExample, key: str, default: Any = None) -> Any:
    extras = example.metadata.extras or {}
    if key in extras and extras[key] not in (None, ""):
        return extras[key]
    if hasattr(example.metadata, key):
        value = getattr(example.metadata, key)
        if value not in (None, ""):
            return value
    return default


def _clone_example(example: PuzzleExample) -> PuzzleExample:
    return PuzzleExample.from_dict(example.to_dict())


def _normalize_family_from_example(example: PuzzleExample, row: dict[str, Any]) -> str:
    value = (
        _metadata_value(example, "official_family")
        or row.get("official_family")
        or row.get("family")
        or "unknown"
    )
    return str(value)


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


def _example_from_row(row: dict[str, Any], *, source_pool: str) -> PuzzleExample:
    example = PuzzleExample.from_dict(row)
    metadata = row.get("metadata", {}) or {}

    if not example.raw_prompt and row.get("prompt"):
        example.raw_prompt = str(row.get("prompt", ""))
    if not example.official_instruction and row.get("official_instruction"):
        example.official_instruction = str(row.get("official_instruction", ""))
    if not example.query:
        example.query = str(row.get("query", row.get("query_input", "")) or "")
    if example.target_answer is None and row.get("target_answer") is not None:
        example.target_answer = str(row.get("target_answer"))
    if not example.metadata.official_family:
        family = row.get("official_family") or row.get("family") or metadata.get("official_family")
        if family not in (None, ""):
            example.metadata.official_family = str(family)
    if not example.metadata.program_signature:
        signature = row.get("program_signature") or metadata.get("program_signature")
        if signature not in (None, ""):
            example.metadata.program_signature = str(signature)
    if example.metadata.teacher_confidence is None:
        confidence = row.get("teacher_confidence")
        if confidence is None:
            confidence = metadata.get("teacher_confidence")
        if confidence is not None:
            example.metadata.teacher_confidence = float(confidence)

    if source_pool == "synth":
        example.metadata.source = "synthetic"

    return example


def _support_pairs_from_row(
    row: dict[str, Any],
) -> tuple[list[dict[str, str]] | None, str | None]:
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
        query_prediction = extras.get("query_prediction")
        return normalized, None if query_prediction is None else str(query_prediction)

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


def _row_needs_rerun(row: dict[str, Any]) -> bool:
    support_pairs, _query_prediction = _support_pairs_from_row(row)
    return support_pairs is None


def _rerun_annotation(
    example: PuzzleExample,
    *,
    settings: Mapping[str, Any],
) -> tuple[list[dict[str, str]], str | None]:
    engine = ChainSearchEngine(
        beam_width=int(settings["beam_width"]),
        max_depth=int(settings["max_depth"]),
    )
    candidates = engine.solve_example(example, top_k=int(settings["top_k"]))
    probe = _clone_example(example)
    annotate_example_from_candidates(probe, candidates)
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
    return support_pairs, query_prediction


def _counterfactual_with_official_verify(
    example: PuzzleExample,
    *,
    support_coverage: float,
    solver_verifiable: bool,
) -> PuzzleExample:
    clone = _clone_example(example)
    clone.metadata.extras = {
        **dict(clone.metadata.extras or {}),
        "support_coverage": float(support_coverage),
        "solver_verifiable": bool(solver_verifiable),
    }
    return clone


def _rate(numerator: int, denominator: int, *, null_if_zero: bool = False) -> float | None:
    if denominator == 0:
        return None if null_if_zero else 0.0
    return numerator / denominator


def _group_stats(rows: list[EvaluatedExample]) -> dict[str, Any]:
    strict_base = [row for row in rows if row.source_pool == "train" and row.strict_repo]
    strict_repo_only = [row for row in strict_base if not row.strict_official]
    silver_base = [row for row in rows if row.source_pool == "train" and row.silver_repo]
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
        "strict_repo_reason": example.strict_repo_reason,
        "strict_official_reason": example.strict_official_reason,
        "annotation_source": example.annotation_source,
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


def _rerun_settings_from_provenance(
    *,
    stage2_provenance_json: str | Path | None,
    train_jsonl: str | Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    provenance_path = (
        Path(stage2_provenance_json)
        if stage2_provenance_json is not None
        else Path(f"{train_jsonl}.provenance.json")
    )
    if not provenance_path.exists():
        payload = {
            "status": "insufficient_evidence",
            "reason": "stage2 provenance missing or mismatched",
            "required": {"provenance_path": str(provenance_path)},
            "found": None,
        }
        return None, payload

    provenance = read_json(provenance_path)
    ok, required, found = stage2_provenance_matches_local(
        provenance,
        output_path=train_jsonl,
    )
    if not ok:
        payload = {
            "status": "insufficient_evidence",
            "reason": "stage2 provenance missing or mismatched",
            "required": required,
            "found": {
                **found,
                "provenance_path": str(provenance_path),
                "code_commit": provenance.get("code_commit"),
                "input_jsonl_sha256": provenance.get("input_jsonl_sha256"),
                "output_jsonl_sha256": provenance.get("output_jsonl_sha256"),
                "created_at_utc": provenance.get("created_at_utc"),
            },
        }
        return None, payload

    settings = {
        "beam_width": int(provenance["beam_width"]),
        "max_depth": int(provenance["max_depth"]),
        "top_k": int(provenance["top_k"]),
        "annotation_engine": str(provenance["annotation_engine"]),
        "operator_priority_hash": str(provenance["operator_priority_hash"]),
        "provenance_path": str(provenance_path),
        "code_commit": provenance.get("code_commit"),
        "input_jsonl_sha256": provenance.get("input_jsonl_sha256"),
        "output_jsonl_sha256": provenance.get("output_jsonl_sha256"),
        "created_at_utc": provenance.get("created_at_utc"),
    }
    return settings, None


def _evaluate_rows(
    *,
    rows: list[dict[str, Any]],
    source_pool: str,
    input_path: str | Path,
    allow_rerun_chain_search: bool,
    rerun_settings: Mapping[str, Any] | None,
) -> list[EvaluatedExample]:
    evaluated: list[EvaluatedExample] = []
    for row in rows:
        example = _example_from_row(row, source_pool=source_pool)
        family = _normalize_family_from_example(example, row)
        target_answer = str("" if example.target_answer is None else example.target_answer)
        program_signature = (
            None
            if _metadata_value(example, "program_signature") in (None, "")
            else str(_metadata_value(example, "program_signature"))
        )
        teacher_confidence = float(_metadata_value(example, "teacher_confidence", 0.0) or 0.0)
        repo_support_coverage = float(example.metadata.extras.get("support_coverage", 0.0) or 0.0)
        repo_solver_verifiable = bool(example.metadata.extras.get("solver_verifiable"))

        support_pairs, query_prediction = _support_pairs_from_row(row)
        annotation_source = "cached"
        if support_pairs is None:
            if not allow_rerun_chain_search:
                raise ValueError(
                    "Missing support_pairs/query_prediction in metadata.extras. "
                    "Re-run with --allow-rerun-chain-search to rebuild annotations."
                )
            if rerun_settings is None:
                raise ValueError("rerun_settings is required when --allow-rerun-chain-search is enabled.")
            support_pairs, query_prediction = _rerun_annotation(example, settings=rerun_settings)
            annotation_source = "rerun"

        official_matches: list[bool] = []
        mismatched_support_pairs: list[dict[str, Any]] = []
        for pair in support_pairs:
            prediction = str(pair.get("prediction", ""))
            target = str(pair.get("target", ""))
            repo_match = _repo_verify_bidirectional(prediction, target)
            official_match = _official_verify(target, prediction)
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

        official_support_coverage = _rate(sum(official_matches), len(official_matches)) or 0.0
        official_solver_verifiable = bool(official_matches) and all(official_matches) and query_prediction is not None

        strict_repo, strict_repo_reason = _select_official_stage2_strict(example)
        official_example = _counterfactual_with_official_verify(
            example,
            support_coverage=float(official_support_coverage),
            solver_verifiable=official_solver_verifiable,
        )
        strict_official, strict_official_reason = _select_official_stage2_strict(official_example)
        silver_repo = _select_official_stage2_silver(example) and not strict_repo
        silver_official = _select_official_stage2_silver(official_example) and not strict_official

        evaluated.append(
            EvaluatedExample(
                id=str(row.get("id", example.id)),
                source_pool=source_pool,
                family=family,
                answer_type=classify_answer_type(target_answer),
                synth_source=_normalize_synth_source(row, input_path),
                annotation_source=annotation_source,
                target_answer=target_answer,
                query_prediction=query_prediction,
                program_signature=program_signature,
                teacher_confidence=teacher_confidence,
                repo_support_coverage=repo_support_coverage,
                official_support_coverage=float(official_support_coverage),
                repo_solver_verifiable=repo_solver_verifiable,
                official_solver_verifiable=official_solver_verifiable,
                strict_repo=bool(strict_repo),
                strict_repo_reason=strict_repo_reason,
                strict_official=bool(strict_official),
                strict_official_reason=strict_official_reason,
                silver_repo=bool(silver_repo),
                silver_official=bool(silver_official),
                synth_repo=False,
                synth_official=False,
                mismatched_support_pairs=mismatched_support_pairs,
                repo_example=example,
                official_example=official_example,
            )
        )
    return evaluated


def _populate_synth_selection(
    *,
    train_examples: list[EvaluatedExample],
    synth_examples: list[EvaluatedExample],
) -> None:
    repo_strict_signatures = {
        row.program_signature
        for row in train_examples
        if row.strict_repo and row.program_signature
    }
    official_strict_signatures = {
        row.program_signature
        for row in train_examples
        if row.strict_official and row.program_signature
    }
    for row in synth_examples:
        row.synth_repo = _select_synth_stage2(
            row.repo_example,
            official_signatures=repo_strict_signatures,
        )
        row.synth_official = _select_synth_stage2(
            row.official_example,
            official_signatures=official_strict_signatures,
        )


def run_audit(
    *,
    train_jsonl: str | Path,
    synth_jsonl: str | Path,
    stage2_report: str | Path | None,
    stage2_provenance_json: str | Path | None,
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

    rerun_settings: dict[str, Any] | None = None
    if allow_rerun_chain_search and any(_row_needs_rerun(row) for row in train_rows + synth_rows):
        rerun_settings, insufficient_payload = _rerun_settings_from_provenance(
            stage2_provenance_json=stage2_provenance_json,
            train_jsonl=train_path,
        )
        if insufficient_payload is not None:
            write_json(output, insufficient_payload)
            return insufficient_payload

    train_examples = _evaluate_rows(
        rows=train_rows,
        source_pool="train",
        input_path=train_path,
        allow_rerun_chain_search=allow_rerun_chain_search,
        rerun_settings=rerun_settings,
    )
    synth_examples = _evaluate_rows(
        rows=synth_rows,
        source_pool="synth",
        input_path=synth_path,
        allow_rerun_chain_search=allow_rerun_chain_search,
        rerun_settings=rerun_settings,
    )
    _populate_synth_selection(
        train_examples=train_examples,
        synth_examples=synth_examples,
    )

    evaluated = train_examples + synth_examples

    strict_repo_base = [row for row in train_examples if row.strict_repo]
    strict_repo_only = [row for row in strict_repo_base if not row.strict_official]
    silver_repo_base = [row for row in train_examples if row.silver_repo]
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

    strict_pollution_rate = _rate(len(strict_repo_only), len(strict_repo_base), null_if_zero=True)
    summary = {
        "num_train_examples": len(train_examples),
        "num_synth_examples": len(synth_examples),
        "strict_repo_accepted": len(strict_repo_base),
        "strict_official_accepted": sum(row.strict_official for row in train_examples),
        "strict_repo_only": len(strict_repo_only),
        "strict_pollution_rate": strict_pollution_rate,
        "silver_repo_accepted": len(silver_repo_base),
        "silver_official_accepted": sum(row.silver_official for row in train_examples),
        "silver_repo_only": len(silver_repo_only),
        "silver_pollution_rate": _rate(len(silver_repo_only), len(silver_repo_base), null_if_zero=True),
        "synth_repo_accepted": len(synth_repo_base),
        "synth_official_accepted": sum(row.synth_official for row in synth_examples),
        "synth_repo_only": len(synth_repo_only),
        "synth_pollution_rate": _rate(len(synth_repo_only), len(synth_repo_base), null_if_zero=True),
    }

    decision_hint: dict[str, Any]
    if len(strict_repo_base) == 0:
        decision_hint = {
            "action": "insufficient_evidence",
            "reason": "strict_repo_accepted == 0 (no samples passed repo strict gate)",
        }
    else:
        assert strict_pollution_rate is not None
        decision_hint = _decision_hint(strict_pollution_rate)

    payload = {
        "inputs": {
            "train_jsonl": str(train_path),
            "synth_jsonl": str(synth_path),
            "stage2_report": None if stage2_report is None else str(stage2_report),
            "stage2_provenance_json": None if rerun_settings is None else str(rerun_settings["provenance_path"]),
            "contract_fingerprint": current_contract_fingerprint().to_dict(),
        },
        "selection_contract": {
            "strict_selector": "src.student.sft_dataset_builder._select_official_stage2_strict",
            "silver_selector": "src.student.sft_dataset_builder._select_official_stage2_silver",
            "synth_selector": "src.student.sft_dataset_builder._select_synth_stage2",
            "hard_triad_families": list(HARD_TRIAD_FAMILIES),
        },
        "summary": summary,
        "by_family": by_family,
        "by_synth_source": by_synth_source,
        "by_answer_type": by_answer_type,
        "by_annotation_source": by_annotation_source,
        "decision_hint": decision_hint,
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
    parser.add_argument("--stage2-provenance-json")
    parser.add_argument("--output", default="data/processed/audit_teacher_gate_extractor_parity.json")
    parser.add_argument("--samples-to-include", type=int, default=50)
    parser.add_argument("--allow-rerun-chain-search", action="store_true")
    args = parser.parse_args(argv)

    run_audit(
        train_jsonl=args.train_jsonl,
        synth_jsonl=args.synth_jsonl,
        stage2_report=args.stage2_report,
        stage2_provenance_json=args.stage2_provenance_json,
        output=args.output,
        samples_to_include=max(0, int(args.samples_to_include)),
        allow_rerun_chain_search=bool(args.allow_rerun_chain_search),
    )


if __name__ == "__main__":
    main()
