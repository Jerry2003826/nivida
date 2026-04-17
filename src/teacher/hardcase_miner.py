from __future__ import annotations

from typing import Any, Iterable


_HARDCASE_FAMILY_PRIORITY = {
    "equation": 0,
    "bit": 0,
    "gravity": 1,
    "unit": 1,
    "cipher": 1,
    "numeral": 1,
}


def _normalise_rows(rows_or_payload: Iterable[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(rows_or_payload, dict):
        return list(rows_or_payload.get("records", []))
    return list(rows_or_payload)


def mine_hard_cases(
    rows_or_payload: Iterable[dict[str, Any]] | dict[str, Any],
    *,
    max_items: int = 32,
) -> list[dict[str, Any]]:
    rows = _normalise_rows(rows_or_payload)
    ranked = sorted(
        rows,
        key=lambda row: (
            row.get("competition_correct", False),
            row.get("boxed_valid", False),
            _HARDCASE_FAMILY_PRIORITY.get(str(row.get("official_family", row.get("family", "unknown"))), 2),
            -float(row.get("teacher_confidence", row.get("confidence", 0.0))),
            -len(row.get("steps", [])),
            str(row.get("id", "")),
        ),
    )
    return ranked[:max_items]
