from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io import read_json, write_json, write_jsonl
from scripts.split_eval_artifact import split_eval_artifact


def _write_eval_payload(path: Path, records: list[dict[str, object]]) -> Path:
    write_json(path, {"records": records, "headline_metric": "competition_correct_rate"})
    return path


def test_splits_failures_and_successes_by_competition_correct(tmp_path: Path) -> None:
    artifact = _write_eval_payload(
        tmp_path / "eval.json",
        [
            {"id": "a", "competition_correct": True},
            {"id": "b", "competition_correct": False},
            {"id": "c", "competition_correct": True},
        ],
    )
    failures, successes = split_eval_artifact(
        input_path=artifact,
        output_failures=tmp_path / "fail.json",
        output_successes=tmp_path / "ok.json",
    )
    assert [row["id"] for row in failures] == ["b"]
    assert [row["id"] for row in successes] == ["a", "c"]

    fail_payload = read_json(tmp_path / "fail.json")
    assert fail_payload["headline_metric"] == "competition_correct_rate"
    assert [row["id"] for row in fail_payload["records"]] == ["b"]
    ok_payload = read_json(tmp_path / "ok.json")
    assert [row["id"] for row in ok_payload["records"]] == ["a", "c"]


def test_restrict_ids_from_jsonl_filters_records(tmp_path: Path) -> None:
    artifact = _write_eval_payload(
        tmp_path / "eval.json",
        [
            {"id": "a", "competition_correct": True},
            {"id": "b", "competition_correct": False},
            {"id": "c", "competition_correct": False},
        ],
    )
    restrict_jsonl = tmp_path / "restrict.jsonl"
    write_jsonl(restrict_jsonl, [{"id": "b"}, {"id": "c"}])

    failures, successes = split_eval_artifact(
        input_path=artifact,
        restrict_ids_path=restrict_jsonl,
        output_failures=tmp_path / "fail.json",
    )
    assert [row["id"] for row in failures] == ["b", "c"]
    assert successes == []


def test_restrict_ids_from_json_records_shape(tmp_path: Path) -> None:
    artifact = _write_eval_payload(
        tmp_path / "eval.json",
        [
            {"id": "x", "competition_correct": True},
            {"id": "y", "competition_correct": True},
        ],
    )
    restrict_json = tmp_path / "restrict.json"
    write_json(restrict_json, {"records": [{"id": "x"}]})

    _, successes = split_eval_artifact(
        input_path=artifact,
        restrict_ids_path=restrict_json,
        output_successes=tmp_path / "ok.json",
    )
    assert [row["id"] for row in successes] == ["x"]


def test_restrict_ids_from_plain_list_shape(tmp_path: Path) -> None:
    artifact = _write_eval_payload(
        tmp_path / "eval.json",
        [
            {"id": "x", "competition_correct": True},
            {"id": "y", "competition_correct": False},
            {"id": "z", "competition_correct": False},
        ],
    )
    restrict_json = tmp_path / "restrict.json"
    write_json(restrict_json, ["y", "z"])

    failures, successes = split_eval_artifact(
        input_path=artifact,
        restrict_ids_path=restrict_json,
        output_failures=tmp_path / "fail.json",
    )
    assert {row["id"] for row in failures} == {"y", "z"}
    assert successes == []


def test_missing_output_paths_skip_writes(tmp_path: Path) -> None:
    artifact = _write_eval_payload(
        tmp_path / "eval.json",
        [{"id": "only", "competition_correct": False}],
    )
    failures, successes = split_eval_artifact(input_path=artifact)
    assert [row["id"] for row in failures] == ["only"]
    assert successes == []
    assert not (tmp_path / "fail.json").exists()
    assert not (tmp_path / "ok.json").exists()


def test_non_object_payload_raises(tmp_path: Path) -> None:
    artifact = tmp_path / "eval.json"
    write_json(artifact, ["not", "a", "dict"])
    with pytest.raises(ValueError, match="JSON object"):
        split_eval_artifact(input_path=artifact)
