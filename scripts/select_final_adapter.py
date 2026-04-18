"""Pick the final adapter (stage2 or stage3) for submission.

Thin wrapper over :mod:`src.student.proxy_selection`. The selection logic
(all-family primary / hard-triad tiebreak / prefer stage2 on complete tie)
lives in that module so the stage-internal checkpoint selector can reuse it
without forking.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.student.proxy_selection import (  # noqa: E402
    compare_proxy_pairs,
    copy_adapter_dir,
    load_proxy_eval,
)


# Backward-compat aliases. External callers (and existing tests) import these
# names directly from the script module.
def load_eval(path: str) -> dict[str, Any]:
    return load_proxy_eval(path)


def copy_adapter(src: str, dst: str) -> None:
    copy_adapter_dir(src, dst)


def choose_adapter(
    stage2_all: dict[str, Any],
    stage3_all: dict[str, Any],
    stage2_hard: dict[str, Any],
    stage3_hard: dict[str, Any],
) -> dict[str, Any]:
    """Stage2 vs stage3 decision, tiebreak prefers stage2 (safer)."""
    decision = compare_proxy_pairs(
        left_name="stage2",
        left_all=stage2_all,
        left_hard=stage2_hard,
        right_name="stage3",
        right_all=stage3_all,
        right_hard=stage3_hard,
        tiebreak_default="stage2",
    )
    # Map to the legacy key/value names the rest of this script and its tests
    # rely on.
    decision["selected_stage"] = decision.pop("winner")
    if decision["rule"] == "prefer_default_on_tie":
        decision["rule"] = "prefer_stage2_on_tie"
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select and materialise the final submission adapter.",
    )
    parser.add_argument("--stage2-hard-eval", required=True)
    parser.add_argument("--stage2-all-eval", required=True)
    parser.add_argument("--stage2-adapter-dir", required=True)
    parser.add_argument("--stage3-hard-eval", required=True)
    parser.add_argument("--stage3-all-eval", required=True)
    parser.add_argument("--stage3-adapter-dir", required=True)
    parser.add_argument("--output-adapter-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    stage2_hard = load_eval(args.stage2_hard_eval)
    stage2_all = load_eval(args.stage2_all_eval)
    stage3_hard = load_eval(args.stage3_hard_eval)
    stage3_all = load_eval(args.stage3_all_eval)

    decision = choose_adapter(stage2_all, stage3_all, stage2_hard, stage3_hard)

    selected_source_dir = (
        args.stage3_adapter_dir
        if decision["selected_stage"] == "stage3"
        else args.stage2_adapter_dir
    )
    copy_adapter(selected_source_dir, args.output_adapter_dir)

    payload = {
        "decision": decision,
        "selected_adapter_dir": args.output_adapter_dir,
        "selected_source_dir": selected_source_dir,
        "stage2": {
            "all_family_proxy": stage2_all,
            "hard_triad_proxy": stage2_hard,
        },
        "stage3": {
            "all_family_proxy": stage3_all,
            "hard_triad_proxy": stage3_hard,
        },
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
