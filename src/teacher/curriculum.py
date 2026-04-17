from __future__ import annotations

from typing import Any


def _difficulty_label(row: dict[str, Any]) -> str:
    if row.get("competition_correct", False):
        return "easy"
    if row.get("numeric", False):
        return "medium"
    confidence = float(row.get("teacher_confidence", row.get("confidence", 0.0)))
    family = str(row.get("official_family", row.get("family", "unknown")))
    if family in {"equation", "bit"}:
        return "hard"
    if confidence >= 0.75:
        return "medium"
    return "hard"


def assign_curriculum_bucket(row: dict[str, Any]) -> str:
    family = str(row.get("official_family", row.get("family", "unknown")))
    subtype = row.get("subtype")
    if subtype:
        return f"{family}:{subtype}:{_difficulty_label(row)}"
    return f"{family}:{_difficulty_label(row)}"


def build_curriculum(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        bucket = assign_curriculum_bucket(row)
        buckets.setdefault(bucket, []).append(row)
    return dict(sorted(buckets.items()))
