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
    require_complete_coverage: bool = False,
) -> dict[str, object]:
    labels = _load_labels(label_path)
    label_ids = set(labels)

    predictions = load_jsonl(prediction_path)

    seen_prediction_ids: set[str] = set()
    duplicate_prediction_ids: set[str] = set()
    unexpected_prediction_ids: set[str] = set()
    rows: list[dict[str, object]] = []

    for row in predictions:
        row_id = str(row.get("id", ""))
        if row_id in seen_prediction_ids:
            duplicate_prediction_ids.add(row_id)
        else:
            seen_prediction_ids.add(row_id)

        example = labels.get(row_id)
        if example is None:
            unexpected_prediction_ids.add(row_id)
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

    missing_prediction_ids = sorted(label_ids - seen_prediction_ids)

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
    payload["coverage"] = {
        "num_labels": len(label_ids),
        "num_predictions": len(predictions),
        "num_scored": len(rows),
        "num_missing": len(missing_prediction_ids),
        "num_unexpected": len(unexpected_prediction_ids),
        "num_duplicate": len(duplicate_prediction_ids),
        "missing_prediction_ids": missing_prediction_ids,
        "unexpected_prediction_ids": sorted(unexpected_prediction_ids),
        "duplicate_prediction_ids": sorted(duplicate_prediction_ids),
    }

    if require_complete_coverage and (
        missing_prediction_ids or unexpected_prediction_ids or duplicate_prediction_ids
    ):
        raise ValueError(
            "Prediction/label coverage mismatch: "
            f"missing={len(missing_prediction_ids)}, "
            f"unexpected={len(unexpected_prediction_ids)}, "
            f"duplicate={len(duplicate_prediction_ids)}"
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
    parser.add_argument(
        "--require-complete-coverage",
        action="store_true",
        help=(
            "Fail with non-zero exit when the prediction set does not exactly "
            "cover the label set (missing / unexpected / duplicate ids)."
        ),
    )
    args = parser.parse_args()

    payload = evaluate_replica(
        prediction_path=args.predictions,
        label_path=args.labels,
        split_path=args.splits,
        require_complete_coverage=args.require_complete_coverage,
    )
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
