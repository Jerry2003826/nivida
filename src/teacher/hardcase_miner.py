from __future__ import annotations

from typing import Any, Iterable

from src.teacher.error_taxonomy import HARD_TRIAD_FAMILIES, classify_error, hardcase_reason


def _normalise_rows(rows_or_payload: Iterable[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(rows_or_payload, dict):
        return list(rows_or_payload.get("records", []))
    return list(rows_or_payload)


def _margin(row: dict[str, Any]) -> float:
    debug = row.get("debug") or {}
    return float(
        row.get("margin")
        or row.get("top1_top2_margin")
        or debug.get("margin_to_best")
        or 0.0
    )


def mine_hard_cases(
    rows_or_payload: Iterable[dict[str, Any]] | dict[str, Any],
    *,
    max_items: int = 32,
) -> list[dict[str, Any]]:
    rows = []
    for row in _normalise_rows(rows_or_payload):
        enriched = dict(row)
        enriched["error_type"] = classify_error(enriched)
        enriched["hardcase_reason"] = hardcase_reason(enriched)
        if enriched.get("program_signature") and not enriched.get("target_signature"):
            enriched["target_signature"] = enriched["program_signature"]
        rows.append(enriched)
    ranked = sorted(
        rows,
        key=lambda row: (
            row.get("competition_correct", False),
            row.get("boxed_valid", False),
            0 if str(row.get("official_family", row.get("family", "unknown"))) in HARD_TRIAD_FAMILIES else 1,
            float(row.get("teacher_confidence", row.get("confidence", 0.0)) or 0.0),
            _margin(row),
            str(row.get("id", "")),
        ),
    )
    return ranked[:max_items]
