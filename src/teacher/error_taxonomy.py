from __future__ import annotations

from typing import Any


HARD_TRIAD_FAMILIES = {"cipher", "bit", "equation"}


def classify_error(row: dict[str, Any]) -> str:
    if row.get("competition_correct", False):
        if row.get("numeric", False) and not row.get("exact", False):
            return "numeric_only"
        return "correct"
    if not row.get("boxed_valid", True):
        return "format_error"
    target_family = row.get("official_family") or row.get("target_family")
    predicted_family = row.get("predicted_family")
    if predicted_family and target_family and predicted_family != target_family:
        return "wrong_family"
    target_subtype = row.get("subtype") or row.get("target_subtype")
    predicted_subtype = row.get("predicted_subtype")
    if predicted_subtype and target_subtype and predicted_subtype != target_subtype:
        return "wrong_subtype"
    target_signature = row.get("target_signature") or row.get("program_signature")
    predicted_signature = row.get("predicted_signature")
    if predicted_signature and target_signature and predicted_signature != target_signature:
        return "wrong_signature"
    confidence = float(row.get("teacher_confidence", row.get("confidence", 1.0)) or 0.0)
    if confidence < 0.5:
        return "low_confidence"
    return "search_miss"


def hardcase_reason(row: dict[str, Any]) -> str:
    error_type = classify_error(row)
    if error_type == "format_error":
        return "format"
    if error_type == "numeric_only":
        return "numeric_margin"
    if error_type == "wrong_family":
        return "wrong_family"
    if error_type in {"wrong_subtype", "wrong_signature", "search_miss"}:
        return "wrong_rule"
    return "low_confidence"
