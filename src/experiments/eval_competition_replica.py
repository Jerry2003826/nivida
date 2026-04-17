from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.common.io import load_jsonl, read_json, write_json
from src.competition.metrics import evaluate_predictions
from src.competition.schema import PuzzleExample


def _load_labels(path: str | Path) -> dict[str, PuzzleExample]:
    return {
        example.id: example
        for example in (PuzzleExample.from_dict(row) for row in load_jsonl(path))
    }


def _checksum(rows: list[dict[str, object]]) -> str:
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def evaluate_replica(
    *,
    prediction_path: str | Path,
    label_path: str | Path,
    split_path: str | Path | None = None,
) -> dict[str, object]:
    labels = _load_labels(label_path)
    predictions = load_jsonl(prediction_path)
    rows: list[dict[str, object]] = []
    for row in predictions:
        example = labels.get(str(row["id"]))
        if example is None:
            continue
        rows.append(
            {
                "id": example.id,
                "prediction": row.get("prediction", ""),
                "target_answer": example.target_answer or "",
                "official_family": example.metadata.official_family,
                "subtype": example.metadata.subtype,
            }
        )

    payload = evaluate_predictions(rows)
    payload["checksum"] = _checksum(
        [
            {
                "id": row["id"],
                "prediction": row["prediction"],
                "competition_correct": row["competition_correct"],
            }
            for row in payload["records"]
        ]
    )

    if split_path:
        split_payload = read_json(split_path)
        split_metrics: dict[str, object] = {}
        for split_name, split_ids in split_payload.items():
            valid_ids = set(split_ids.get("valid_ids", []))
            split_rows = [row for row in rows if row["id"] in valid_ids]
            split_metrics[split_name] = evaluate_predictions(split_rows)
        payload["split_metrics"] = split_metrics
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions against the local competition proxy.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--splits")
    parser.add_argument("--output", default="data/processed/competition_replica_eval.json")
    args = parser.parse_args()

    payload = evaluate_replica(
        prediction_path=args.predictions,
        label_path=args.labels,
        split_path=args.splits,
    )
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
