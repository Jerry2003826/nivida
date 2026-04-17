from __future__ import annotations

from dataclasses import dataclass

from src.competition.answer_extract import extract_single_boxed_answer


@dataclass(slots=True)
class FormatValidation:
    is_valid: bool
    answer: str | None
    error: str | None


def wrap_boxed(answer: str) -> str:
    return f"\\boxed{{{answer.strip()}}}"


def validate_boxed_output(text: str) -> FormatValidation:
    result = extract_single_boxed_answer(text)
    return FormatValidation(is_valid=result.is_valid, answer=result.answer, error=result.error)


def ensure_boxed_output(text: str, *, fallback_answer: str | None = None) -> str:
    result = extract_single_boxed_answer(text)
    if result.is_valid and result.answer is not None:
        return wrap_boxed(result.answer)
    if fallback_answer is None:
        raise ValueError("Output does not contain exactly one boxed answer and no fallback was provided.")
    return wrap_boxed(fallback_answer)
