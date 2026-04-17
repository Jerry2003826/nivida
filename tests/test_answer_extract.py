from __future__ import annotations

from src.competition.answer_extract import extract_single_boxed_answer


def test_extract_single_boxed_answer_basic() -> None:
    result = extract_single_boxed_answer("reasoning ... \\boxed{42}")
    assert result.is_valid
    assert result.answer == "42"


def test_extract_multiple_boxed_answers_is_invalid() -> None:
    result = extract_single_boxed_answer("\\boxed{1} or \\boxed{2}")
    assert not result.is_valid
    assert result.error == "multiple_boxed_answers"


def test_extract_nested_braces() -> None:
    result = extract_single_boxed_answer("\\boxed{\\frac{1}{2}}")
    assert result.is_valid
    assert result.answer == "\\frac{1}{2}"


def test_extract_missing_boxed_answer() -> None:
    result = extract_single_boxed_answer("42")
    assert not result.is_valid
    assert result.error == "missing_boxed_answer"


def test_extract_boxed_answer_strips_outer_whitespace() -> None:
    result = extract_single_boxed_answer("\\boxed{  7  }")
    assert result.is_valid
    assert result.answer == "7"
