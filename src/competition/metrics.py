from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, getcontext
from fractions import Fraction
from typing import Any, Iterable

from src.common.text_normalise import normalize_for_exact_match
from src.competition.answer_extract import extract_single_boxed_answer


getcontext().prec = 50

BOXED_TOKEN = r"\boxed{"
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
            fraction = Fraction(int(plain_fraction.group(1)), int(plain_fraction.group(2)))
        except ZeroDivisionError:
            return None
        return Decimal(fraction.numerator) / Decimal(fraction.denominator)

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


def competition_numeric_match(
    prediction: str | None,
    target: str | None,
    *,
    rel_tol: float = 1e-2,
    abs_tol: float = 1e-5,
) -> bool:
    pred_value = parse_numeric_value(prediction)
    target_value = parse_numeric_value(target)
    if pred_value is None or target_value is None:
        return False
    diff = abs(pred_value - target_value)
    abs_tol_decimal = Decimal(str(abs_tol))
    rel_tol_decimal = Decimal(str(rel_tol))
    scale = max(abs(pred_value), abs(target_value))
    return diff <= max(abs_tol_decimal, rel_tol_decimal * scale)


def competition_correct(
    prediction: str | None,
    target: str | None,
    *,
    rel_tol: float = 1e-2,
    abs_tol: float = 1e-5,
) -> bool:
    if exact_match(prediction, target):
        return True
    return competition_numeric_match(prediction, target, rel_tol=rel_tol, abs_tol=abs_tol)


def numeric_match(prediction: str | None, target: str | None, tolerance: float = 1e-2) -> bool:
    return competition_numeric_match(prediction, target, rel_tol=tolerance, abs_tol=1e-5)


@dataclass(slots=True)
class EvaluationRecord:
    example_id: str
    prediction: str
    target: str
    boxed_valid: bool
    exact: bool
    numeric: bool
    competition_correct: bool
    extracted_answer: str | None
    official_family: str | None
    subtype: str | None


def evaluate_predictions(
    rows: Iterable[dict[str, Any]],
    *,
    prediction_key: str = "prediction",
    target_key: str = "target_answer",
    id_key: str = "id",
    numeric_rel_tolerance: float = 1e-2,
    numeric_abs_tolerance: float = 1e-5,
) -> dict[str, Any]:
    records: list[EvaluationRecord] = []
    family_exact_buckets: dict[str, list[bool]] = {}
    family_competition_buckets: dict[str, list[bool]] = {}
    subtype_buckets: dict[str, list[bool]] = {}
    for row in rows:
        prediction = row.get(prediction_key, "")
        target = row.get(target_key, row.get("target", ""))
        boxed = extract_single_boxed_answer(str(prediction))
        extracted = boxed.answer if boxed.is_valid else None
        exact = exact_match(str(prediction), str(target))
        numeric = competition_numeric_match(
            str(prediction),
            str(target),
            rel_tol=numeric_rel_tolerance,
            abs_tol=numeric_abs_tolerance,
        )
        correct = exact or numeric
        official_family = row.get("official_family") or row.get("family")
        subtype = row.get("subtype")
        records.append(
            EvaluationRecord(
                example_id=str(row.get(id_key, len(records))),
                prediction=str(prediction),
                target=str(target),
                boxed_valid=boxed.is_valid,
                exact=exact,
                numeric=numeric,
                competition_correct=correct,
                extracted_answer=extracted,
                official_family=None if official_family is None else str(official_family),
                subtype=None if subtype is None else str(subtype),
            )
        )
        if official_family:
            family_exact_buckets.setdefault(str(official_family), []).append(exact)
            family_competition_buckets.setdefault(str(official_family), []).append(correct)
        if official_family and subtype:
            subtype_buckets.setdefault(f"{official_family}:{subtype}", []).append(correct)

    total = len(records) or 1
    boxed_rate = sum(record.boxed_valid for record in records) / total
    exact_rate = sum(record.exact for record in records) / total
    numeric_rate = sum(record.numeric for record in records) / total
    competition_correct_rate = sum(record.competition_correct for record in records) / total
    invalid_rate = 1.0 - boxed_rate
    avg_output_tokens = sum(len(record.prediction.split()) for record in records) / total

    return {
        "num_examples": len(records),
        "boxed_rate": boxed_rate,
        "exact_match_rate": exact_rate,
        "numeric_match_rate": numeric_rate,
        "competition_correct_rate": competition_correct_rate,
        "invalid_output_rate": invalid_rate,
        "avg_output_tokens": avg_output_tokens,
        "family_wise_accuracy_exact": {
            family: sum(values) / max(1, len(values))
            for family, values in sorted(family_exact_buckets.items())
        },
        "family_wise_accuracy_competition": {
            family: sum(values) / max(1, len(values))
            for family, values in sorted(family_competition_buckets.items())
        },
        "family_wise_competition_correct": {
            family: sum(values) / max(1, len(values))
            for family, values in sorted(family_competition_buckets.items())
        },
        "subtype_wise_accuracy_competition": {
            subtype: sum(values) / max(1, len(values))
            for subtype, values in sorted(subtype_buckets.items())
        },
        "subtype_wise_competition_correct": {
            subtype: sum(values) / max(1, len(values))
            for subtype, values in sorted(subtype_buckets.items())
        },
        "records": [
            {
                "id": record.example_id,
                "prediction": record.prediction,
                "target": record.target,
                "boxed_valid": record.boxed_valid,
                "exact": record.exact,
                "numeric": record.numeric,
                "competition_correct": record.competition_correct,
                "extracted_answer": record.extracted_answer,
                "official_family": record.official_family,
                "subtype": record.subtype,
            }
            for record in records
        ],
    }
