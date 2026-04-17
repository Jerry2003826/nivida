from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, getcontext
from fractions import Fraction
from typing import Any, Iterable

from src.common.text_normalise import normalize_for_exact_match
from src.competition.answer_extract import extract_single_boxed_answer


getcontext().prec = 50

_LATEX_FRAC_RE = re.compile(r"^\\frac\{([^{}]+)\}\{([^{}]+)\}$")
_PLAIN_FRAC_RE = re.compile(r"^([+-]?\d+)\/([+-]?\d+)$")


def unwrap_boxed(text: str | None) -> str:
    if text is None:
        return ""
    if BOXED_TOKEN in text:
        result = extract_single_boxed_answer(text)
        if result.is_valid and result.answer is not None:
            return result.answer
    return text


def canonicalize_answer(text: str | None) -> str:
    return normalize_for_exact_match(unwrap_boxed(text))


def parse_numeric_value(text: str | None) -> Decimal | None:
    """Parse common numeric answer forms including fractions and base prefixes."""
    candidate = canonicalize_answer(text).replace(",", "").replace(" ", "")
    if not candidate:
        return None

    frac_match = _LATEX_FRAC_RE.match(candidate)
    if frac_match:
        numerator = parse_numeric_value(frac_match.group(1))
        denominator = parse_numeric_value(frac_match.group(2))
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator

    plain_fraction = _PLAIN_FRAC_RE.match(candidate)
    if plain_fraction:
        try:
            return Decimal(Fraction(int(plain_fraction.group(1)), int(plain_fraction.group(2))).numerator) / Decimal(
                Fraction(int(plain_fraction.group(1)), int(plain_fraction.group(2))).denominator
            )
        except ZeroDivisionError:
            return None

    try:
        if candidate.lower().startswith(("0x", "-0x", "+0x")):
            return Decimal(int(candidate, 16))
        if candidate.lower().startswith(("0b", "-0b", "+0b")):
            return Decimal(int(candidate, 2))
        if candidate.lower().startswith(("0o", "-0o", "+0o")):
            return Decimal(int(candidate, 8))
        return Decimal(candidate)
    except (InvalidOperation, ValueError):
        return None


def exact_match(prediction: str | None, target: str | None) -> bool:
    return canonicalize_answer(prediction) == canonicalize_answer(target)


def numeric_match(prediction: str | None, target: str | None, tolerance: float = 1e-9) -> bool:
    pred_value = parse_numeric_value(prediction)
    target_value = parse_numeric_value(target)
    if pred_value is None or target_value is None:
        return False
    return abs(pred_value - target_value) <= Decimal(str(tolerance))


@dataclass(slots=True)
class EvaluationRecord:
    example_id: str
    prediction: str
    target: str
    boxed_valid: bool
    exact: bool
    numeric: bool
    extracted_answer: str | None


def evaluate_predictions(
    rows: Iterable[dict[str, Any]],
    *,
    prediction_key: str = "prediction",
    target_key: str = "target_answer",
    id_key: str = "id",
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    records: list[EvaluationRecord] = []
    family_buckets: dict[str, list[bool]] = {}
    for row in rows:
        prediction = row.get(prediction_key, "")
        target = row.get(target_key, "")
        boxed = extract_single_boxed_answer(prediction)
        extracted = boxed.answer if boxed.is_valid else None
        exact = exact_match(prediction, target)
        numeric = numeric_match(prediction, target, tolerance=tolerance)
        records.append(
            EvaluationRecord(
                example_id=str(row.get(id_key, len(records))),
                prediction=str(prediction),
                target=str(target),
                boxed_valid=boxed.is_valid,
                exact=exact,
                numeric=numeric,
                extracted_answer=extracted,
            )
        )
        for family in row.get("family_tags", []):
            family_buckets.setdefault(str(family), []).append(exact)

    total = len(records) or 1
    boxed_rate = sum(record.boxed_valid for record in records) / total
    exact_rate = sum(record.exact for record in records) / total
    numeric_rate = sum(record.numeric for record in records) / total
    invalid_rate = 1.0 - boxed_rate
    avg_output_tokens = sum(len(record.prediction.split()) for record in records) / total

    return {
        "num_examples": len(records),
        "boxed_rate": boxed_rate,
        "exact_match_rate": exact_rate,
        "numeric_match_rate": numeric_rate,
        "invalid_output_rate": invalid_rate,
        "avg_output_tokens": avg_output_tokens,
        "family_wise_accuracy": {
            family: sum(values) / max(1, len(values))
            for family, values in sorted(family_buckets.items())
        },
        "records": [
            {
                "id": record.example_id,
                "prediction": record.prediction,
                "target": record.target,
                "boxed_valid": record.boxed_valid,
                "exact": record.exact,
                "numeric": record.numeric,
                "extracted_answer": record.extracted_answer,
            }
            for record in records
        ],
    }


BOXED_TOKEN = r"\boxed{"
