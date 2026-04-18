from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io import write_json
from scripts.decide_subtype_branch_promotion import (
    decide_branch_promotion,
    load_promotion_eval,
)


def _write_eval(
    path: Path,
    *,
    rate: float,
    num_examples: int,
    num_missing: int = 0,
    num_unexpected: int = 0,
    num_duplicate: int = 0,
) -> Path:
    write_json(
        path,
        {
            "competition_correct_rate": rate,
            "num_examples": num_examples,
            "coverage": {
                "num_missing": num_missing,
                "num_unexpected": num_unexpected,
                "num_duplicate": num_duplicate,
            },
        },
    )
    return path


def test_branch_promotion_accepts_exact_half_sample_boundary(tmp_path: Path) -> None:
    baseline_all = load_promotion_eval(
        _write_eval(tmp_path / "baseline_all.json", rate=0.8000, num_examples=100)
    )
    branch_all = load_promotion_eval(
        _write_eval(tmp_path / "branch_all.json", rate=0.7950, num_examples=100)
    )
    baseline_hard = load_promotion_eval(
        _write_eval(tmp_path / "baseline_hard.json", rate=0.5000, num_examples=50)
    )
    branch_hard = load_promotion_eval(
        _write_eval(tmp_path / "branch_hard.json", rate=0.5100, num_examples=50)
    )

    payload = decide_branch_promotion(
        baseline_all=baseline_all,
        baseline_hard=baseline_hard,
        branch_all=branch_all,
        branch_hard=branch_hard,
    )

    assert payload["promote"] is True
    assert payload["all_tol"] == pytest.approx(0.005)
    assert payload["hard_tol"] == pytest.approx(0.01)
    assert payload["allowed_all_drop_examples"] == pytest.approx(0.5)
    assert payload["required_hard_delta_examples"] == pytest.approx(0.5)
    assert payload["all_delta_examples"] == pytest.approx(-0.5)
    assert payload["hard_delta_examples"] == pytest.approx(0.5)


def test_branch_promotion_rejects_when_all_family_drops_too_far(tmp_path: Path) -> None:
    baseline_all = load_promotion_eval(
        _write_eval(tmp_path / "baseline_all.json", rate=0.8000, num_examples=100)
    )
    branch_all = load_promotion_eval(
        _write_eval(tmp_path / "branch_all.json", rate=0.7900, num_examples=100)
    )
    baseline_hard = load_promotion_eval(
        _write_eval(tmp_path / "baseline_hard.json", rate=0.5000, num_examples=50)
    )
    branch_hard = load_promotion_eval(
        _write_eval(tmp_path / "branch_hard.json", rate=0.5200, num_examples=50)
    )

    payload = decide_branch_promotion(
        baseline_all=baseline_all,
        baseline_hard=baseline_hard,
        branch_all=branch_all,
        branch_hard=branch_hard,
    )

    assert payload["promote"] is False


def test_branch_promotion_rejects_when_hard_triad_gain_is_below_half_sample(tmp_path: Path) -> None:
    baseline_all = load_promotion_eval(
        _write_eval(tmp_path / "baseline_all.json", rate=0.8000, num_examples=100)
    )
    branch_all = load_promotion_eval(
        _write_eval(tmp_path / "branch_all.json", rate=0.8000, num_examples=100)
    )
    baseline_hard = load_promotion_eval(
        _write_eval(tmp_path / "baseline_hard.json", rate=0.5000, num_examples=50)
    )
    branch_hard = load_promotion_eval(
        _write_eval(tmp_path / "branch_hard.json", rate=0.5090, num_examples=50)
    )

    payload = decide_branch_promotion(
        baseline_all=baseline_all,
        baseline_hard=baseline_hard,
        branch_all=branch_all,
        branch_hard=branch_hard,
    )

    assert payload["promote"] is False


def test_branch_promotion_hard_fails_on_incomplete_coverage(tmp_path: Path) -> None:
    path = _write_eval(
        tmp_path / "branch_all.json",
        rate=0.80,
        num_examples=100,
        num_missing=2,
    )

    with pytest.raises(SystemExit, match="coverage is incomplete"):
        load_promotion_eval(path)


def test_branch_promotion_rejects_num_examples_mismatch(tmp_path: Path) -> None:
    baseline_all = load_promotion_eval(
        _write_eval(tmp_path / "baseline_all.json", rate=0.8000, num_examples=100)
    )
    branch_all = load_promotion_eval(
        _write_eval(tmp_path / "branch_all.json", rate=0.8000, num_examples=99)
    )
    baseline_hard = load_promotion_eval(
        _write_eval(tmp_path / "baseline_hard.json", rate=0.5000, num_examples=50)
    )
    branch_hard = load_promotion_eval(
        _write_eval(tmp_path / "branch_hard.json", rate=0.5200, num_examples=50)
    )

    with pytest.raises(SystemExit, match="all-family proxy num_examples mismatch"):
        decide_branch_promotion(
            baseline_all=baseline_all,
            baseline_hard=baseline_hard,
            branch_all=branch_all,
            branch_hard=branch_hard,
        )
