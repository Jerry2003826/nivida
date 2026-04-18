from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io import write_json
from scripts.analyze_proxy_results import analyze_proxy_results


def _write_eval(
    path: Path,
    *,
    rate: float,
    num_examples: int = 100,
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


def test_analyze_proxy_results_returns_complete_summary(tmp_path: Path) -> None:
    stage2_hard = _write_eval(tmp_path / "stage2_hard.json", rate=0.51)
    stage2_all = _write_eval(tmp_path / "stage2_all.json", rate=0.81)
    stage3_hard = _write_eval(tmp_path / "stage3_hard.json", rate=0.53)
    stage3_all = _write_eval(tmp_path / "stage3_all.json", rate=0.82)
    branch_hard = _write_eval(tmp_path / "branch_hard.json", rate=0.54)
    branch_all = _write_eval(tmp_path / "branch_all.json", rate=0.80)
    stage2_selection = tmp_path / "stage2_selection.json"
    final_selection = tmp_path / "final_selection.json"
    write_json(stage2_selection, {"selected_candidate": "checkpoint-250"})
    write_json(final_selection, {"decision": {"selected_stage": "stage3"}})

    payload = analyze_proxy_results(
        stage2_hard_eval=stage2_hard,
        stage2_all_eval=stage2_all,
        stage3_hard_eval=stage3_hard,
        stage3_all_eval=stage3_all,
        branch_hard_eval=branch_hard,
        branch_all_eval=branch_all,
        stage2_selection_json=stage2_selection,
        final_selection_json=final_selection,
    )

    assert payload["status"] == "complete"
    assert payload["missing_groups"] == []
    assert payload["stage2"]["selection"]["selected_candidate"] == "checkpoint-250"
    assert payload["final_selection"]["decision"]["selected_stage"] == "stage3"
    assert payload["branch_promotion_preview"]["promote"] is False
    assert any("stage3; package adapter_final_selected" in rec for rec in payload["recommendations"])
    assert any("stage2 bestproxy preferred an intermediate checkpoint" in rec for rec in payload["recommendations"])
    assert any("branch stays stage2-only" in rec for rec in payload["recommendations"])


def test_analyze_proxy_results_strict_mode_fails_when_stage3_missing(tmp_path: Path) -> None:
    stage2_hard = _write_eval(tmp_path / "stage2_hard.json", rate=0.51)
    stage2_all = _write_eval(tmp_path / "stage2_all.json", rate=0.81)
    branch_hard = _write_eval(tmp_path / "branch_hard.json", rate=0.54)
    branch_all = _write_eval(tmp_path / "branch_all.json", rate=0.80)

    with pytest.raises(SystemExit, match="stage3: missing both hard/all proxy evals"):
        analyze_proxy_results(
            stage2_hard_eval=stage2_hard,
            stage2_all_eval=stage2_all,
            stage3_hard_eval=tmp_path / "missing_stage3_hard.json",
            stage3_all_eval=tmp_path / "missing_stage3_all.json",
            branch_hard_eval=branch_hard,
            branch_all_eval=branch_all,
        )


def test_analyze_proxy_results_allow_partial_accepts_missing_stage3_and_branch(tmp_path: Path) -> None:
    stage2_hard = _write_eval(tmp_path / "stage2_hard.json", rate=0.51)
    stage2_all = _write_eval(tmp_path / "stage2_all.json", rate=0.81)

    payload = analyze_proxy_results(
        stage2_hard_eval=stage2_hard,
        stage2_all_eval=stage2_all,
        stage3_hard_eval=tmp_path / "missing_stage3_hard.json",
        stage3_all_eval=tmp_path / "missing_stage3_all.json",
        branch_hard_eval=tmp_path / "missing_branch_hard.json",
        branch_all_eval=tmp_path / "missing_branch_all.json",
        allow_partial=True,
    )

    assert payload["status"] == "partial"
    assert set(payload["missing_groups"]) == {"stage3", "branch"}
    assert payload["branch_promotion_preview"] is None
    assert any("stage3 artifacts are still missing" in rec for rec in payload["recommendations"])
    assert any("branch artifacts are still missing" in rec for rec in payload["recommendations"])


def test_analyze_proxy_results_still_hard_fails_on_coverage_problems_in_partial_mode(
    tmp_path: Path,
) -> None:
    stage2_hard = _write_eval(tmp_path / "stage2_hard.json", rate=0.51, num_missing=1)
    stage2_all = _write_eval(tmp_path / "stage2_all.json", rate=0.81)

    with pytest.raises(SystemExit, match="num_missing"):
        analyze_proxy_results(
            stage2_hard_eval=stage2_hard,
            stage2_all_eval=stage2_all,
            stage3_hard_eval=tmp_path / "missing_stage3_hard.json",
            stage3_all_eval=tmp_path / "missing_stage3_all.json",
            branch_hard_eval=tmp_path / "missing_branch_hard.json",
            branch_all_eval=tmp_path / "missing_branch_all.json",
            allow_partial=True,
        )
