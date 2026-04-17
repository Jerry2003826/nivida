from __future__ import annotations

from pathlib import Path

from src.common.io import write_json, write_jsonl
from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair
from src.experiments.eval_competition_replica import evaluate_replica


def test_evaluate_replica_reports_checksum_and_split_metrics(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    splits_path = tmp_path / "splits.json"

    example = PuzzleExample(
        id="ex1",
        raw_prompt="raw",
        official_instruction="raw",
        parsed_examples=[PuzzlePair(input="1", output="2")],
        query="3",
        target_answer="4",
        metadata=PuzzleMetadata(official_family="equation", subtype="numeric", source="official"),
    )
    write_jsonl(labels_path, [example.to_dict()])
    write_jsonl(predictions_path, [{"id": "ex1", "prediction": "\\boxed{4}"}])
    write_json(splits_path, {"rule_novelty": {"train_ids": [], "valid_ids": ["ex1"]}})

    payload = evaluate_replica(
        prediction_path=predictions_path,
        label_path=labels_path,
        split_path=splits_path,
    )
    assert payload["competition_correct_rate"] == 1.0
    assert payload["checksum"]
    assert payload["split_metrics"]["rule_novelty"]["competition_correct_rate"] == 1.0
