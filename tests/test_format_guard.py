from __future__ import annotations

import pytest

from src.student.format_guard import ensure_boxed_output, validate_boxed_output, wrap_boxed


def test_wrap_boxed() -> None:
    assert wrap_boxed("42") == "\\boxed{42}"


def test_validate_boxed_output() -> None:
    result = validate_boxed_output("Answer: \\boxed{99}")
    assert result.is_valid
    assert result.answer == "99"


def test_ensure_boxed_output_uses_fallback() -> None:
    assert ensure_boxed_output("plain answer", fallback_answer="7") == "\\boxed{7}"


def test_ensure_boxed_output_without_fallback_raises() -> None:
    with pytest.raises(ValueError):
        ensure_boxed_output("plain answer")
