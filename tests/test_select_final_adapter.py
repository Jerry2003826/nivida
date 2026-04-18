"""Tests for ``scripts.select_final_adapter``."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.io import write_json
from scripts.select_final_adapter import (
    choose_adapter,
    copy_adapter,
    load_eval,
)


def _eval_payload(
    rate: float,
    *,
    num_examples: int = 400,
    num_missing: int = 0,
    num_unexpected: int = 0,
    num_duplicate: int = 0,
) -> dict[str, object]:
    return {
        "competition_correct_rate": rate,
        "num_examples": num_examples,
        "coverage": {
            "num_missing": num_missing,
            "num_unexpected": num_unexpected,
            "num_duplicate": num_duplicate,
        },
    }


def _make_adapter_dir(path: Path, *, marker: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(f'{{"r": 32, "marker": "{marker}"}}', encoding="utf-8")
    (path / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    return path


# --- choose_adapter -----------------------------------------------------------


def test_choose_adapter_picks_stage3_when_all_family_clearly_wins() -> None:
    stage2_all = _eval_payload(0.500, num_examples=400)
    stage3_all = _eval_payload(0.560, num_examples=400)
    stage2_hard = _eval_payload(0.400, num_examples=300)
    stage3_hard = _eval_payload(0.420, num_examples=300)

    decision = choose_adapter(stage2_all, stage3_all, stage2_hard, stage3_hard)
    assert decision["selected_stage"] == "stage3"
    assert decision["rule"] == "all_family_primary"
    assert decision["all_delta"] == pytest.approx(0.060)


def test_choose_adapter_picks_stage2_when_all_family_clearly_loses() -> None:
    stage2_all = _eval_payload(0.550, num_examples=400)
    stage3_all = _eval_payload(0.520, num_examples=400)
    stage2_hard = _eval_payload(0.400, num_examples=300)
    stage3_hard = _eval_payload(0.500, num_examples=300)

    decision = choose_adapter(stage2_all, stage3_all, stage2_hard, stage3_hard)
    assert decision["selected_stage"] == "stage2"
    assert decision["rule"] == "all_family_primary"


def test_choose_adapter_hard_triad_tiebreak_when_all_family_within_tolerance() -> None:
    # all_delta = 0.0005 (< 0.5/400 = 0.00125 tolerance)
    stage2_all = _eval_payload(0.5000, num_examples=400)
    stage3_all = _eval_payload(0.5005, num_examples=400)
    # hard_delta = 0.020 (> 0.5/300 = 0.00167 tolerance)
    stage2_hard = _eval_payload(0.400, num_examples=300)
    stage3_hard = _eval_payload(0.420, num_examples=300)

    decision = choose_adapter(stage2_all, stage3_all, stage2_hard, stage3_hard)
    assert decision["selected_stage"] == "stage3"
    assert decision["rule"] == "hard_triad_tiebreak"


def test_choose_adapter_prefers_stage2_on_complete_tie() -> None:
    stage2_all = _eval_payload(0.500, num_examples=400)
    stage3_all = _eval_payload(0.500, num_examples=400)
    stage2_hard = _eval_payload(0.400, num_examples=300)
    stage3_hard = _eval_payload(0.400, num_examples=300)

    decision = choose_adapter(stage2_all, stage3_all, stage2_hard, stage3_hard)
    assert decision["selected_stage"] == "stage2"
    assert decision["rule"] == "prefer_stage2_on_tie"


def test_choose_adapter_hard_triad_degradation_within_all_tol_still_picks_stage2() -> None:
    """Stage3 does not win the tiebreak unless its hard-triad proxy actually exceeds stage2."""
    stage2_all = _eval_payload(0.5000, num_examples=400)
    stage3_all = _eval_payload(0.5005, num_examples=400)  # within tol
    stage2_hard = _eval_payload(0.420, num_examples=300)
    stage3_hard = _eval_payload(0.400, num_examples=300)  # stage3 worse on hard triad

    decision = choose_adapter(stage2_all, stage3_all, stage2_hard, stage3_hard)
    assert decision["selected_stage"] == "stage2"
    assert decision["rule"] == "prefer_stage2_on_tie"


def test_choose_adapter_raises_on_num_examples_mismatch() -> None:
    stage2_all = _eval_payload(0.5, num_examples=400)
    stage3_all = _eval_payload(0.5, num_examples=399)
    stage2_hard = _eval_payload(0.4, num_examples=300)
    stage3_hard = _eval_payload(0.4, num_examples=300)

    with pytest.raises(SystemExit, match="all-family proxy num_examples mismatch"):
        choose_adapter(stage2_all, stage3_all, stage2_hard, stage3_hard)


# --- load_eval ----------------------------------------------------------------


def test_load_eval_rejects_missing_coverage(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    write_json(path, _eval_payload(0.5, num_missing=3))
    with pytest.raises(SystemExit, match="num_missing"):
        load_eval(str(path))


def test_load_eval_rejects_duplicate_predictions(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    write_json(path, _eval_payload(0.5, num_duplicate=2))
    with pytest.raises(SystemExit, match="num_duplicate"):
        load_eval(str(path))


def test_load_eval_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    write_json(path, {"coverage": {"num_missing": 0, "num_unexpected": 0, "num_duplicate": 0}})
    with pytest.raises(SystemExit, match="competition_correct_rate"):
        load_eval(str(path))


def test_load_eval_returns_clean_payload(tmp_path: Path) -> None:
    path = tmp_path / "ok.json"
    write_json(path, _eval_payload(0.75, num_examples=900))
    loaded = load_eval(str(path))
    assert loaded["competition_correct_rate"] == 0.75
    assert loaded["num_examples"] == 900


# --- copy_adapter -------------------------------------------------------------


def test_copy_adapter_replaces_existing_target(tmp_path: Path) -> None:
    src = _make_adapter_dir(tmp_path / "src", marker="from-stage3")
    dst = tmp_path / "dst"
    _make_adapter_dir(dst, marker="stale")

    copy_adapter(str(src), str(dst))
    config = json.loads((dst / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["marker"] == "from-stage3"


def test_copy_adapter_raises_when_source_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="adapter source not found"):
        copy_adapter(str(tmp_path / "nope"), str(tmp_path / "out"))
