"""Partition a baseline / replica eval artifact into failure and success records.

Used by ``scripts/train_stage3_repair.sh`` to derive stage2 student failure and
success buckets for stage3 repair + replay construction. Supports an optional
``--restrict-ids`` filter so a single eval artifact can be split per-split
(e.g., failures restricted to the hard-triad train subset while successes
cover the full official train pool).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import load_jsonl, read_json, write_json


def _load_restrict_ids(path: str | None) -> set[str]:
    """Load the set of example ids that should be retained before splitting.

    Accepts three shapes:

    - ``*.jsonl``: one record per line, each must contain an ``id`` field
    - JSON object with ``records`` / ``rows`` arrays of id-bearing dicts, or an
      ``ids`` list of primitives
    - JSON array of dicts (each with ``id``) or primitives used directly as ids
    """
    if not path:
        return set()

    source = Path(path)
    if source.suffix == ".jsonl":
        return {
            str(row["id"])
            for row in load_jsonl(source)
            if isinstance(row, dict) and "id" in row
        }

    payload = read_json(source)
    if isinstance(payload, dict):
        if "records" in payload and isinstance(payload["records"], list):
            return {
                str(row["id"])
                for row in payload["records"]
                if isinstance(row, dict) and "id" in row
            }
        if "rows" in payload and isinstance(payload["rows"], list):
            return {
                str(row["id"])
                for row in payload["rows"]
                if isinstance(row, dict) and "id" in row
            }
        if "ids" in payload and isinstance(payload["ids"], list):
            return {str(value) for value in payload["ids"]}

    if isinstance(payload, list):
        out: set[str] = set()
        for item in payload:
            if isinstance(item, dict) and "id" in item:
                out.add(str(item["id"]))
            else:
                out.add(str(item))
        return out

    raise ValueError(f"Unsupported restrict-id payload: {source}")


def _read_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("records", payload.get("rows", []))
    if not isinstance(records, list):
        raise ValueError("Eval artifact must contain a list under 'records' or 'rows'.")
    out: list[dict[str, Any]] = []
    for row in records:
        if isinstance(row, dict):
            out.append(dict(row))
    return out


def _write_partition(
    *,
    payload: dict[str, Any],
    records: list[dict[str, Any]],
    output_path: str | None,
) -> None:
    if not output_path:
        return
    result = dict(payload)
    result["records"] = records
    result.pop("rows", None)
    write_json(output_path, result)


def split_eval_artifact(
    *,
    input_path: str | Path,
    restrict_ids_path: str | Path | None = None,
    output_failures: str | Path | None = None,
    output_successes: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = read_json(input_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Eval artifact must be a JSON object: {input_path}")

    records = _read_records(payload)
    allowed_ids = _load_restrict_ids(str(restrict_ids_path) if restrict_ids_path else None)
    if allowed_ids:
        records = [row for row in records if str(row.get("id")) in allowed_ids]

    failures = [row for row in records if not bool(row.get("competition_correct", False))]
    successes = [row for row in records if bool(row.get("competition_correct", False))]

    _write_partition(
        payload=payload,
        records=failures,
        output_path=str(output_failures) if output_failures else None,
    )
    _write_partition(
        payload=payload,
        records=successes,
        output_path=str(output_successes) if output_successes else None,
    )
    return failures, successes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split an eval artifact into failure/success partitions.",
    )
    parser.add_argument("--input", required=True, help="Path to eval artifact JSON")
    parser.add_argument(
        "--restrict-ids",
        help="Optional JSON/JSONL file whose ids constrain which eval records are kept.",
    )
    parser.add_argument(
        "--output-failures",
        help="Where to write competition_correct=false records.",
    )
    parser.add_argument(
        "--output-successes",
        help="Where to write competition_correct=true records.",
    )
    args = parser.parse_args()

    split_eval_artifact(
        input_path=args.input,
        restrict_ids_path=args.restrict_ids,
        output_failures=args.output_failures,
        output_successes=args.output_successes,
    )


if __name__ == "__main__":
    main()
