from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import read_json, write_json  # noqa: E402
from src.student.proxy_selection import load_proxy_eval  # noqa: E402


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    return payload


def _load_eval_group(
    *,
    name: str,
    hard_path: Path,
    all_path: Path,
    allow_missing_group: bool,
) -> dict[str, Any] | None:
    hard_exists = hard_path.is_file()
    all_exists = all_path.is_file()
    if not hard_exists and not all_exists:
        if allow_missing_group:
            return None
        raise SystemExit(f"{name}: missing both hard/all proxy evals")
    if hard_exists != all_exists:
        raise SystemExit(f"{name}: hard/all proxy evals must appear together")
    return {
        "hard": load_proxy_eval(hard_path),
        "all": load_proxy_eval(all_path),
    }


def _maybe_load_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return _load_json_object(path)


def analyze_proxy_results(
    *,
    stage2_hard_eval: str | Path,
    stage2_all_eval: str | Path,
    stage3_hard_eval: str | Path,
    stage3_all_eval: str | Path,
    branch_hard_eval: str | Path,
    branch_all_eval: str | Path,
    stage2_selection_json: str | Path | None = None,
    stage3_selection_json: str | Path | None = None,
    branch_selection_json: str | Path | None = None,
    final_selection_json: str | Path | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    stage2 = _load_eval_group(
        name="stage2",
        hard_path=Path(stage2_hard_eval),
        all_path=Path(stage2_all_eval),
        allow_missing_group=False,
    )
    assert stage2 is not None
    stage3 = _load_eval_group(
        name="stage3",
        hard_path=Path(stage3_hard_eval),
        all_path=Path(stage3_all_eval),
        allow_missing_group=allow_partial,
    )
    branch = _load_eval_group(
        name="branch",
        hard_path=Path(branch_hard_eval),
        all_path=Path(branch_all_eval),
        allow_missing_group=allow_partial,
    )

    missing_groups = [
        name
        for name, group in (("stage3", stage3), ("branch", branch))
        if group is None
    ]
    status = "partial" if missing_groups else "complete"

    return {
        "status": status,
        "missing_groups": missing_groups,
        "stage2": {
            **stage2,
            "selection": _maybe_load_optional(
                None if stage2_selection_json is None else Path(stage2_selection_json)
            ),
        },
        "stage3": None
        if stage3 is None
        else {
            **stage3,
            "selection": _maybe_load_optional(
                None if stage3_selection_json is None else Path(stage3_selection_json)
            ),
        },
        "branch": None
        if branch is None
        else {
            **branch,
            "selection": _maybe_load_optional(
                None if branch_selection_json is None else Path(branch_selection_json)
            ),
        },
        "final_selection": _maybe_load_optional(
            None if final_selection_json is None else Path(final_selection_json)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise proxy eval artifacts across canonical and branch runs."
    )
    parser.add_argument(
        "--stage2-hard-eval",
        default="data/processed/stage2_bestproxy_hard_eval.json",
    )
    parser.add_argument(
        "--stage2-all-eval",
        default="data/processed/stage2_bestproxy_all_eval.json",
    )
    parser.add_argument(
        "--stage3-hard-eval",
        default="data/processed/stage3_bestproxy_hard_eval.json",
    )
    parser.add_argument(
        "--stage3-all-eval",
        default="data/processed/stage3_bestproxy_all_eval.json",
    )
    parser.add_argument(
        "--branch-hard-eval",
        default="data/processed/stage2_subtype_rescue_bestproxy_hard_eval.json",
    )
    parser.add_argument(
        "--branch-all-eval",
        default="data/processed/stage2_subtype_rescue_bestproxy_all_eval.json",
    )
    parser.add_argument(
        "--stage2-selection-json",
        default="data/processed/stage2_best_checkpoint_selection.json",
    )
    parser.add_argument(
        "--stage3-selection-json",
        default="data/processed/stage3_best_checkpoint_selection.json",
    )
    parser.add_argument(
        "--branch-selection-json",
        default="data/processed/stage2_subtype_rescue_best_checkpoint_selection.json",
    )
    parser.add_argument(
        "--final-selection-json",
        default="data/processed/final_adapter_selection.json",
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    payload = analyze_proxy_results(
        stage2_hard_eval=args.stage2_hard_eval,
        stage2_all_eval=args.stage2_all_eval,
        stage3_hard_eval=args.stage3_hard_eval,
        stage3_all_eval=args.stage3_all_eval,
        branch_hard_eval=args.branch_hard_eval,
        branch_all_eval=args.branch_all_eval,
        stage2_selection_json=args.stage2_selection_json,
        stage3_selection_json=args.stage3_selection_json,
        branch_selection_json=args.branch_selection_json,
        final_selection_json=args.final_selection_json,
        allow_partial=args.allow_partial,
    )

    if args.output:
        write_json(args.output, payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
