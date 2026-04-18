"""Tests for ``src.student.proxy_selection`` helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.io import write_json
from src.student.proxy_selection import (
    compare_proxy_pairs,
    copy_adapter_dir,
    load_proxy_eval,
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


# --- load_proxy_eval ----------------------------------------------------------


def test_load_proxy_eval_happy_path(tmp_path: Path) -> None:
    path = tmp_path / "ok.json"
    write_json(path, _eval_payload(0.75, num_examples=900))
    loaded = load_proxy_eval(path)
    assert loaded["competition_correct_rate"] == 0.75
    assert loaded["num_examples"] == 900


def test_load_proxy_eval_rejects_coverage_missing(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    write_json(path, _eval_payload(0.5, num_missing=3))
    with pytest.raises(SystemExit, match="num_missing"):
        load_proxy_eval(path)


def test_load_proxy_eval_rejects_coverage_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    write_json(path, _eval_payload(0.5, num_duplicate=1))
    with pytest.raises(SystemExit, match="num_duplicate"):
        load_proxy_eval(path)


def test_load_proxy_eval_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    write_json(
        path,
        {"coverage": {"num_missing": 0, "num_unexpected": 0, "num_duplicate": 0}},
    )
    with pytest.raises(SystemExit, match="competition_correct_rate"):
        load_proxy_eval(path)


def test_load_proxy_eval_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(SystemExit, match="JSON object"):
        load_proxy_eval(path)


# --- compare_proxy_pairs ------------------------------------------------------


def test_compare_returns_right_when_all_family_clearly_wins() -> None:
    decision = compare_proxy_pairs(
        left_name="stage2",
        left_all=_eval_payload(0.50, num_examples=400),
        left_hard=_eval_payload(0.40, num_examples=300),
        right_name="stage3",
        right_all=_eval_payload(0.56, num_examples=400),
        right_hard=_eval_payload(0.42, num_examples=300),
        tiebreak_default="stage2",
    )
    assert decision["winner"] == "stage3"
    assert decision["rule"] == "all_family_primary"


def test_compare_returns_left_when_all_family_clearly_loses() -> None:
    decision = compare_proxy_pairs(
        left_name="stage2",
        left_all=_eval_payload(0.55, num_examples=400),
        left_hard=_eval_payload(0.40, num_examples=300),
        right_name="stage3",
        right_all=_eval_payload(0.52, num_examples=400),
        right_hard=_eval_payload(0.50, num_examples=300),
        tiebreak_default="stage2",
    )
    assert decision["winner"] == "stage2"
    assert decision["rule"] == "all_family_primary"


def test_compare_uses_hard_triad_tiebreak_when_all_family_within_tolerance() -> None:
    # all_delta = 0.0005 < 0.5/400 = 0.00125
    # hard_delta = 0.020 > 0.5/300 = 0.00167
    decision = compare_proxy_pairs(
        left_name="ckpt-1000",
        left_all=_eval_payload(0.5000, num_examples=400),
        left_hard=_eval_payload(0.400, num_examples=300),
        right_name="final",
        right_all=_eval_payload(0.5005, num_examples=400),
        right_hard=_eval_payload(0.420, num_examples=300),
        tiebreak_default="final",
    )
    assert decision["winner"] == "final"
    assert decision["rule"] == "hard_triad_tiebreak"


def test_compare_defaults_on_complete_tie_to_tiebreak_default() -> None:
    # Both deltas within tolerance.
    decision = compare_proxy_pairs(
        left_name="ckpt-500",
        left_all=_eval_payload(0.5000, num_examples=400),
        left_hard=_eval_payload(0.400, num_examples=300),
        right_name="final",
        right_all=_eval_payload(0.5000, num_examples=400),
        right_hard=_eval_payload(0.400, num_examples=300),
        tiebreak_default="final",
    )
    assert decision["winner"] == "final"
    assert decision["rule"] == "prefer_default_on_tie"


def test_compare_tiebreak_default_respects_stage2() -> None:
    decision = compare_proxy_pairs(
        left_name="stage2",
        left_all=_eval_payload(0.50, num_examples=400),
        left_hard=_eval_payload(0.40, num_examples=300),
        right_name="stage3",
        right_all=_eval_payload(0.50, num_examples=400),
        right_hard=_eval_payload(0.40, num_examples=300),
        tiebreak_default="stage2",
    )
    assert decision["winner"] == "stage2"
    assert decision["rule"] == "prefer_default_on_tie"


def test_compare_raises_on_num_examples_mismatch() -> None:
    with pytest.raises(SystemExit, match="all-family proxy num_examples"):
        compare_proxy_pairs(
            left_name="a",
            left_all=_eval_payload(0.5, num_examples=400),
            left_hard=_eval_payload(0.4, num_examples=300),
            right_name="b",
            right_all=_eval_payload(0.5, num_examples=399),
            right_hard=_eval_payload(0.4, num_examples=300),
            tiebreak_default="a",
        )


def test_compare_raises_on_invalid_tiebreak_default() -> None:
    with pytest.raises(ValueError, match="tiebreak_default"):
        compare_proxy_pairs(
            left_name="a",
            left_all=_eval_payload(0.5, num_examples=400),
            left_hard=_eval_payload(0.4, num_examples=300),
            right_name="b",
            right_all=_eval_payload(0.5, num_examples=400),
            right_hard=_eval_payload(0.4, num_examples=300),
            tiebreak_default="c",
        )


# --- copy_adapter_dir ---------------------------------------------------------


def test_copy_adapter_dir_replaces_existing(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "marker.txt").write_text("fresh", encoding="utf-8")
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "stale.txt").write_text("stale", encoding="utf-8")

    copy_adapter_dir(src, dst)
    assert (dst / "marker.txt").read_text(encoding="utf-8") == "fresh"
    assert not (dst / "stale.txt").exists()


def test_copy_adapter_dir_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="adapter source not found"):
        copy_adapter_dir(tmp_path / "nope", tmp_path / "out")
