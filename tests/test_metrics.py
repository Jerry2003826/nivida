from __future__ import annotations

from src.competition.metrics import competition_numeric_match, evaluate_predictions, exact_match


def test_exact_match_ignores_trivial_whitespace() -> None:
    assert exact_match("\\boxed{ 42 }", "42")


def test_competition_numeric_match_supports_relative_and_absolute_tolerance() -> None:
    assert competition_numeric_match("\\boxed{2.0}", "2")
    assert competition_numeric_match("\\boxed{1/2}", "\\boxed{\\frac{1}{2}}")
    assert competition_numeric_match("\\boxed{101.0}", "100.0", rel_tol=0.02, abs_tol=1e-5)
    assert competition_numeric_match("\\boxed{0.000009}", "0", rel_tol=1e-2, abs_tol=1e-5)


def test_evaluate_predictions_summary_uses_competition_correct() -> None:
    summary = evaluate_predictions(
        [
            {"id": "a", "prediction": "\\boxed{2.0}", "target_answer": "2", "official_family": "unit", "subtype": "scale"},
            {"id": "b", "prediction": "\\boxed{101}", "target_answer": "100", "official_family": "gravity", "subtype": "fit_constant"},
            {"id": "c", "prediction": "oops", "target_answer": "3", "official_family": "cipher", "subtype": "token_substitution"},
        ],
        numeric_rel_tolerance=0.02,
        numeric_abs_tolerance=1e-5,
    )
    assert summary["num_examples"] == 3
    assert summary["boxed_rate"] == 2 / 3
    assert summary["exact_match_rate"] == 0.0
    assert summary["numeric_match_rate"] == 2 / 3
    assert summary["competition_correct_rate"] == 2 / 3
    assert summary["family_wise_competition_correct"]["gravity"] == 1.0
    assert summary["subtype_wise_competition_correct"]["unit:scale"] == 1.0
