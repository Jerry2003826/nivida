from __future__ import annotations

from src.competition.metrics import competition_correct, competition_numeric_match, evaluate_predictions, exact_match


def test_exact_match_ignores_trivial_whitespace() -> None:
    assert exact_match("\\boxed{ 42 }", "42")


def test_competition_numeric_match_respects_relative_boundary() -> None:
    assert competition_numeric_match("\\boxed{2.01}", "2.0", rel_tol=1e-2, abs_tol=1e-5)
    assert not competition_numeric_match("\\boxed{2.03}", "2.0", rel_tol=1e-2, abs_tol=1e-5)


def test_competition_numeric_match_handles_relative_error_near_one_percent() -> None:
    assert competition_numeric_match("\\boxed{100}", "101", rel_tol=1e-2, abs_tol=1e-5)
    assert not competition_correct("\\boxed{100}", "102", rel_tol=1e-2, abs_tol=1e-5)


def test_evaluate_predictions_summary_uses_competition_correct() -> None:
    summary = evaluate_predictions(
        [
            {"id": "a", "prediction": "\\boxed{2.0}", "target_answer": "2", "official_family": "unit", "subtype": "unit_scale"},
            {"id": "b", "prediction": "\\boxed{101}", "target_answer": "100", "official_family": "gravity", "subtype": "gravity_inverse_square"},
            {"id": "c", "prediction": "oops", "target_answer": "3", "official_family": "cipher", "subtype": "cipher_vocab"},
        ],
        numeric_rel_tolerance=0.02,
        numeric_abs_tolerance=1e-5,
    )
    assert summary["num_examples"] == 3
    assert summary["boxed_rate"] == 2 / 3
    assert summary["exact_match_rate"] == 0.0
    assert summary["numeric_match_rate"] == 2 / 3
    assert summary["competition_correct_rate"] == 2 / 3
    assert summary["family_wise_accuracy_exact"]["gravity"] == 0.0
    assert summary["family_wise_accuracy_competition"]["gravity"] == 1.0
    assert summary["subtype_wise_accuracy_competition"]["unit:unit_scale"] == 1.0
    assert {"exact", "numeric", "competition_correct", "boxed_valid"} <= set(summary["records"][0])
