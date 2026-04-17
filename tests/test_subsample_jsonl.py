from __future__ import annotations

from pathlib import Path

from src.common.io import load_jsonl, write_jsonl
from scripts.subsample_jsonl import subsample_jsonl


def _write_rows(path: Path, count: int) -> Path:
    write_jsonl(
        path,
        [{"id": f"row-{idx}", "value": idx} for idx in range(count)],
    )
    return path


def test_subsample_jsonl_truncates_to_limit(tmp_path: Path) -> None:
    input_path = _write_rows(tmp_path / "input.jsonl", 5)
    output_path = tmp_path / "output.jsonl"

    subsample_jsonl(input_path, output_path, limit=2)

    assert [row["id"] for row in load_jsonl(output_path)] == ["row-0", "row-1"]


def test_subsample_jsonl_is_noop_when_limit_exceeds_row_count(tmp_path: Path) -> None:
    input_path = _write_rows(tmp_path / "input.jsonl", 3)
    output_path = tmp_path / "output.jsonl"

    subsample_jsonl(input_path, output_path, limit=10)

    assert load_jsonl(output_path) == load_jsonl(input_path)


def test_subsample_jsonl_preserves_order_and_structure(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    rows = [
        {"id": "first", "payload": {"x": 1}},
        {"id": "second", "payload": {"x": 2}},
        {"id": "third", "payload": {"x": 3}},
    ]
    write_jsonl(input_path, rows)
    output_path = tmp_path / "output.jsonl"

    subsample_jsonl(input_path, output_path, limit=2)

    assert load_jsonl(output_path) == rows[:2]
