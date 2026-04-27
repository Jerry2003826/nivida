from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.io import load_jsonl, write_json, write_jsonl
from src.competition.schema import PuzzleExample
from src.student.format_guard import wrap_boxed
from src.teacher.chain_search import ChainSearchEngine
from src.teacher.program_signature import normalize_family_alias


DEFAULT_OVERRIDE_FAMILIES = {"equation", "bit"}
HIGH_RISK_TEMPLATE_CLASSES = {
    "expressible_oracle_miss",
    "operator_gap_oracle_miss",
    "unseen_key_template_miss",
    "unseen_literal_high_risk",
}


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _family(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    return normalize_family_alias(
        row.get("official_family")
        or row.get("family")
        or metadata.get("official_family")
        or metadata.get("family")
        or "unknown"
    )


def _subtype(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    return str(row.get("subtype") or metadata.get("subtype") or "")


def _template_risk_class(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    extras = metadata.get("extras") if isinstance(metadata.get("extras"), dict) else {}
    return str(
        row.get("template_risk_class")
        or metadata.get("template_risk_class")
        or extras.get("template_risk_class")
        or ""
    )


def _row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("id", row.get("sample_id", index)))


def _prediction_text(row: dict[str, Any], prediction_key: str) -> str:
    value = row.get(prediction_key)
    if value is None:
        value = row.get("generation", row.get("prediction", ""))
    return "" if value is None else str(value)


def solver_override_for_label(
    label_row: dict[str, Any],
    *,
    engine: ChainSearchEngine,
    min_confidence: float = 0.88,
    min_support_coverage: float = 1.0,
    top_k: int = 5,
    override_families: set[str] = DEFAULT_OVERRIDE_FAMILIES,
) -> dict[str, Any]:
    family = _family(label_row)
    if family not in override_families:
        return {"override": False, "reason": f"family {family} not enabled"}
    if _subtype(label_row) == "equation_template" and _template_risk_class(label_row) in HIGH_RISK_TEMPLATE_CLASSES:
        return {"override": False, "reason": "equation_template high-risk class"}
    try:
        example = PuzzleExample.from_dict(label_row)
    except Exception as exc:
        return {"override": False, "reason": f"label parse failed: {exc}"}
    candidates = engine.solve_example(example, top_k=top_k)
    if not candidates:
        return {"override": False, "reason": "no solver candidate"}
    best = candidates[0]
    prediction = "" if best.query_prediction is None else str(best.query_prediction)
    support_coverage = float(best.exact_ratio)
    confidence = float(best.confidence)
    if not prediction:
        return {"override": False, "reason": "empty solver prediction"}
    if support_coverage < min_support_coverage:
        return {
            "override": False,
            "reason": "support below threshold",
            "solver_prediction": prediction,
            "solver_confidence": confidence,
            "support_coverage": support_coverage,
        }
    if confidence < min_confidence:
        return {
            "override": False,
            "reason": "confidence below threshold",
            "solver_prediction": prediction,
            "solver_confidence": confidence,
            "support_coverage": support_coverage,
        }
    return {
        "override": True,
        "reason": "high-confidence solver/verifier override",
        "solver_prediction": prediction,
        "solver_confidence": confidence,
        "support_coverage": support_coverage,
        "solver_steps": [step.op_name for step in best.steps],
        "solver_debug": best.debug,
    }


def apply_solver_assisted_finalizer(
    *,
    predictions_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    prediction_key: str = "generation",
    min_confidence: float = 0.88,
    min_support_coverage: float = 1.0,
    beam_width: int = 10,
    max_depth: int = 4,
    top_k: int = 5,
    override_families: set[str] = DEFAULT_OVERRIDE_FAMILIES,
) -> dict[str, Any]:
    predictions = load_jsonl(predictions_path)
    labels = load_jsonl(labels_path)
    labels_by_id = {_row_id(row, index): row for index, row in enumerate(labels)}
    engine = ChainSearchEngine(beam_width=beam_width, max_depth=max_depth)

    output_rows: list[dict[str, Any]] = []
    override_count = 0
    missing_labels: list[str] = []
    for index, prediction_row in enumerate(predictions):
        row_id = _row_id(prediction_row, index)
        label_row = labels_by_id.get(row_id)
        raw_generation = _prediction_text(prediction_row, prediction_key)
        if label_row is None:
            missing_labels.append(row_id)
            output_rows.append(
                {
                    **prediction_row,
                    "generation": raw_generation,
                    "raw_generation": raw_generation,
                    "finalizer_action": "keep_model_missing_label",
                }
            )
            continue
        decision = solver_override_for_label(
            label_row,
            engine=engine,
            min_confidence=min_confidence,
            min_support_coverage=min_support_coverage,
            top_k=top_k,
            override_families=override_families,
        )
        if decision.get("override"):
            override_count += 1
            final_generation = wrap_boxed(str(decision["solver_prediction"]))
            action = "solver_override"
        else:
            final_generation = raw_generation
            action = "keep_model"
        output_rows.append(
            {
                **prediction_row,
                "generation": final_generation,
                "raw_generation": raw_generation,
                "finalizer_action": action,
                "solver_decision": decision,
            }
        )

    write_jsonl(output_path, output_rows)
    report = {
        "predictions": str(predictions_path),
        "labels": str(labels_path),
        "output": str(output_path),
        "prediction_key": prediction_key,
        "num_predictions": len(predictions),
        "num_labels": len(labels),
        "num_overrides": override_count,
        "override_rate": override_count / len(predictions) if predictions else 0.0,
        "missing_label_ids": missing_labels,
        "settings": {
            "min_confidence": min_confidence,
            "min_support_coverage": min_support_coverage,
            "beam_width": beam_width,
            "max_depth": max_depth,
            "top_k": top_k,
            "override_families": sorted(override_families),
        },
    }
    if report_path is not None:
        write_json(report_path, report)
    return report

