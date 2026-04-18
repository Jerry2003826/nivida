"""Tests for ``scripts.decide_stage3_gate``.

The stage3 repair shell branches on the JSON produced by this helper. The
contract tested here:

- ``skip_stage3`` reflects whether hard-triad train failures is zero
- ``disable_eval_dataset`` reflects whether hard-triad valid failures is zero
- legacy ``rows`` payload is still readable
- non-dict payloads fail fast with a clear SystemExit
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.io import write_json
from scripts.decide_stage3_gate import decide_stage3_gate


def _write_failures(path: Path, count: int, *, key: str = "records") -> Path:
    records = [
        {"id": f"fail-{idx}", "competition_correct": False}
        for idx in range(count)
    ]
    write_json(path, {key: records})
    return path


def test_full_pipeline_when_both_have_failures(tmp_path: Path) -> None:
    train = _write_failures(tmp_path / "train.json", count=3)
    valid = _write_failures(tmp_path / "valid.json", count=2)
    decision = decide_stage3_gate(train_failures_path=train, valid_failures_path=valid)
    assert decision == {
        "train_failure_count": 3,
        "valid_failure_count": 2,
        "skip_stage3": False,
        "disable_eval_dataset": False,
    }


def test_skip_stage3_when_train_is_empty(tmp_path: Path) -> None:
    train = _write_failures(tmp_path / "train.json", count=0)
    valid = _write_failures(tmp_path / "valid.json", count=2)
    decision = decide_stage3_gate(train_failures_path=train, valid_failures_path=valid)
    assert decision["skip_stage3"] is True
    assert decision["disable_eval_dataset"] is False
    assert decision["train_failure_count"] == 0


def test_disable_eval_when_valid_is_empty(tmp_path: Path) -> None:
    train = _write_failures(tmp_path / "train.json", count=5)
    valid = _write_failures(tmp_path / "valid.json", count=0)
    decision = decide_stage3_gate(train_failures_path=train, valid_failures_path=valid)
    assert decision["skip_stage3"] is False
    assert decision["disable_eval_dataset"] is True


def test_both_empty_skips_and_disables(tmp_path: Path) -> None:
    train = _write_failures(tmp_path / "train.json", count=0)
    valid = _write_failures(tmp_path / "valid.json", count=0)
    decision = decide_stage3_gate(train_failures_path=train, valid_failures_path=valid)
    assert decision["skip_stage3"] is True
    assert decision["disable_eval_dataset"] is True


def test_legacy_rows_payload_is_accepted(tmp_path: Path) -> None:
    train = _write_failures(tmp_path / "train.json", count=4, key="rows")
    valid = _write_failures(tmp_path / "valid.json", count=1, key="rows")
    decision = decide_stage3_gate(train_failures_path=train, valid_failures_path=valid)
    assert decision["train_failure_count"] == 4
    assert decision["valid_failure_count"] == 1


def test_non_dict_payload_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(SystemExit):
        decide_stage3_gate(train_failures_path=path, valid_failures_path=path)


def test_non_list_records_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    write_json(path, {"records": "not-a-list"})
    with pytest.raises(SystemExit):
        decide_stage3_gate(train_failures_path=path, valid_failures_path=path)
