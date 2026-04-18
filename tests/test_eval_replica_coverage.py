"""Coverage-gate tests for ``evaluate_replica``."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io import write_jsonl
from src.experiments.eval_competition_replica import evaluate_replica


def _write_labels(path: Path, ids: list[str]) -> Path:
    rows = [
        {
            "id": example_id,
            "raw_prompt": "raw",
            "official_instruction": "",
            "parsed_examples": [{"input": "i", "output": "o"}],
            "query": "q",
            "target_answer": "a",
            "metadata": {
                "official_family": "bit",
                "subtype": "bit_xor_mask",
                "source": "official",
                "split": "train",
            },
        }
        for example_id in ids
    ]
    write_jsonl(path, rows)
    return path


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> Path:
    write_jsonl(path, rows)
    return path


def test_coverage_reports_missing_ids_without_strict(tmp_path: Path) -> None:
    labels = _write_labels(tmp_path / "labels.jsonl", ["a", "b"])
    predictions = _write_predictions(
        tmp_path / "pred.jsonl",
        [{"id": "a", "prediction": "\\boxed{a}"}],
    )
    payload = evaluate_replica(prediction_path=predictions, label_path=labels)
    coverage = payload["coverage"]
    assert coverage["num_missing"] == 1
    assert coverage["missing_prediction_ids"] == ["b"]
    assert coverage["num_unexpected"] == 0
    assert coverage["num_duplicate"] == 0


def test_strict_coverage_raises_on_missing(tmp_path: Path) -> None:
    labels = _write_labels(tmp_path / "labels.jsonl", ["a", "b"])
    predictions = _write_predictions(
        tmp_path / "pred.jsonl",
        [{"id": "a", "prediction": "\\boxed{a}"}],
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        evaluate_replica(
            prediction_path=predictions,
            label_path=labels,
            require_complete_coverage=True,
        )


def test_strict_coverage_raises_on_duplicate(tmp_path: Path) -> None:
    labels = _write_labels(tmp_path / "labels.jsonl", ["a", "b"])
    predictions = _write_predictions(
        tmp_path / "pred.jsonl",
        [
            {"id": "a", "prediction": "\\boxed{a}"},
            {"id": "a", "prediction": "\\boxed{a}"},
            {"id": "b", "prediction": "\\boxed{b}"},
        ],
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        evaluate_replica(
            prediction_path=predictions,
            label_path=labels,
            require_complete_coverage=True,
        )


def test_strict_coverage_raises_on_unexpected(tmp_path: Path) -> None:
    labels = _write_labels(tmp_path / "labels.jsonl", ["a", "b"])
    predictions = _write_predictions(
        tmp_path / "pred.jsonl",
        [
            {"id": "a", "prediction": "\\boxed{a}"},
            {"id": "b", "prediction": "\\boxed{b}"},
            {"id": "c", "prediction": "\\boxed{c}"},
        ],
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        evaluate_replica(
            prediction_path=predictions,
            label_path=labels,
            require_complete_coverage=True,
        )


def test_strict_coverage_passes_when_exact(tmp_path: Path) -> None:
    labels = _write_labels(tmp_path / "labels.jsonl", ["a", "b"])
    predictions = _write_predictions(
        tmp_path / "pred.jsonl",
        [
            {"id": "a", "prediction": "\\boxed{a}"},
            {"id": "b", "prediction": "\\boxed{b}"},
        ],
    )
    payload = evaluate_replica(
        prediction_path=predictions,
        label_path=labels,
        require_complete_coverage=True,
    )
    coverage = payload["coverage"]
    assert coverage["num_missing"] == 0
    assert coverage["num_unexpected"] == 0
    assert coverage["num_duplicate"] == 0
    assert coverage["num_scored"] == 2


def test_coverage_lists_all_three_categories(tmp_path: Path) -> None:
    labels = _write_labels(tmp_path / "labels.jsonl", ["a", "b", "c"])
    predictions = _write_predictions(
        tmp_path / "pred.jsonl",
        [
            {"id": "a", "prediction": "\\boxed{a}"},
            {"id": "a", "prediction": "\\boxed{a}"},
            {"id": "x", "prediction": "\\boxed{x}"},
        ],
    )
    payload = evaluate_replica(prediction_path=predictions, label_path=labels)
    coverage = payload["coverage"]
    assert coverage["missing_prediction_ids"] == ["b", "c"]
    assert coverage["duplicate_prediction_ids"] == ["a"]
    assert coverage["unexpected_prediction_ids"] == ["x"]
