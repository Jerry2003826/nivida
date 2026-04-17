from __future__ import annotations

from src.competition.metrics import evaluate_predictions, exact_match, numeric_match


def test_exact_match_ignores_trivial_whitespace() -> None:
    assert exact_match("\\boxed{ 42 }", "42")


def test_numeric_match_equivalent_strings() -> None:
    assert numeric_match("\\boxed{2.0}", "2")
    assert numeric_match("\\boxed{1/2}", "\\boxed{\\frac{1}{2}}")


def test_evaluate_predictions_summary() -> None:
    summary = evaluate_predictions(
        [
            {"id": "a", "prediction": "\\boxed{2}", "target_answer": "2"},
            {"id": "b", "prediction": "oops", "target_answer": "3"},
        ]
    )
    assert summary["num_examples"] == 2
    assert summary["boxed_rate"] == 0.5
    assert summary["exact_match_rate"] == 0.5
