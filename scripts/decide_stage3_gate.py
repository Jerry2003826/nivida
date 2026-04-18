"""Stage3 repair gate: decide whether to skip stage3 entirely or disable its eval set.

Stage3 repair consumes stage2 model failure artifacts. If stage2 is already
strong enough that a hard-triad train / valid partition contains zero
failures, the downstream ``sft_dataset_builder`` call would raise a
``RepairArtifactSchemaError`` because empty ``records`` is rejected by the
baseline schema guard. This helper inspects the two failure artifacts and
records a boolean decision that the shell script branches on:

- ``skip_stage3``           : True when there are no hard-triad train failures
                              to repair.
- ``disable_eval_dataset``  : True when there are no hard-triad valid failures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _count_records(path: str | Path) -> int:
    payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a JSON object with 'records' or 'rows'.")
    records = payload.get("records", payload.get("rows", []))
    if not isinstance(records, list):
        raise SystemExit(f"{path}: 'records'/'rows' must be a list, got {type(records).__name__}.")
    return len(records)


def decide_stage3_gate(
    *,
    train_failures_path: str | Path,
    valid_failures_path: str | Path,
) -> dict[str, Any]:
    train_failure_count = _count_records(train_failures_path)
    valid_failure_count = _count_records(valid_failures_path)
    return {
        "train_failure_count": train_failure_count,
        "valid_failure_count": valid_failure_count,
        "skip_stage3": train_failure_count == 0,
        "disable_eval_dataset": valid_failure_count == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decide whether stage3 repair should be skipped or run with a "
            "trainer-only eval_dataset=None setup, based on the stage2 "
            "adapter's failure counts."
        )
    )
    parser.add_argument("--train-failures", required=True)
    parser.add_argument("--valid-failures", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    decision = decide_stage3_gate(
        train_failures_path=args.train_failures,
        valid_failures_path=args.valid_failures,
    )
    Path(args.output).write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
