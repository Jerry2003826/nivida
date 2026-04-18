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
from scripts.decide_subtype_branch_promotion import decide_branch_promotion  # noqa: E402
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


def _load_skipped_artifact(path: Path | None) -> dict[str, Any] | None:
    payload = _maybe_load_optional(path)
    if payload is None:
        return None
    if payload.get("skipped") is not True:
        raise SystemExit(f"{path}: expected skipped=true")
    return payload


def _promotion_ready_eval(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "coverage_summary": {
            "num_missing": int(payload.get("coverage", {}).get("num_missing", 0)),
            "num_unexpected": int(payload.get("coverage", {}).get("num_unexpected", 0)),
            "num_duplicate": int(payload.get("coverage", {}).get("num_duplicate", 0)),
            "completeness": 1.0,
        },
    }


def _selected_stage(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    decision = payload.get("decision")
    if isinstance(decision, dict) and isinstance(decision.get("selected_stage"), str):
        return decision["selected_stage"]
    if isinstance(payload.get("selected_stage"), str):
        return str(payload["selected_stage"])
    adapter_dir = payload.get("selected_adapter_dir") or payload.get("output_adapter_dir")
    if isinstance(adapter_dir, str):
        if "stage3" in adapter_dir:
            return "stage3"
        if "stage2" in adapter_dir:
            return "stage2"
    return None


def _build_recommendations(
    *,
    stage2: dict[str, Any],
    stage3: dict[str, Any] | None,
    branch: dict[str, Any] | None,
    branch_skipped: dict[str, Any] | None,
    final_selection: dict[str, Any] | None,
    missing_groups: list[str],
) -> tuple[list[str], dict[str, Any] | None]:
    recommendations: list[str] = []

    stage2_selection = stage2.get("selection")
    if isinstance(stage2_selection, dict) and stage2_selection.get("selected_candidate") not in (
        None,
        "final",
    ):
        recommendations.append(
            "stage2 bestproxy preferred an intermediate checkpoint; review whether the "
            "stage2 schedule overtrains after the best proxy step."
        )

    if stage3 is not None:
        stage3_selection = stage3.get("selection")
        if isinstance(stage3_selection, dict) and stage3_selection.get("selected_candidate") not in (
            None,
            "final",
        ):
            recommendations.append(
                "stage3 bestproxy preferred an intermediate checkpoint; compare the winning "
                "checkpoint against the final adapter before tuning the stage3 schedule."
            )

    selected_stage = _selected_stage(final_selection)
    if selected_stage == "stage2":
        recommendations.append(
            "final selector kept stage2 over stage3; package adapter_final_selected and "
            "treat stage3 as non-winning on the current proxy pair."
        )
    elif selected_stage == "stage3":
        recommendations.append(
            "final selector advanced stage3; package adapter_final_selected after validation."
        )

    branch_promotion: dict[str, Any] | None = None
    if branch is not None:
        branch_promotion = decide_branch_promotion(
            baseline_all=_promotion_ready_eval(stage2["all"]),
            baseline_hard=_promotion_ready_eval(stage2["hard"]),
            branch_all=_promotion_ready_eval(branch["all"]),
            branch_hard=_promotion_ready_eval(branch["hard"]),
        )
        if branch_promotion["promote"]:
            recommendations.append(
                "subtype-rescue branch clears the stage3 promotion gate."
            )
        else:
            recommendations.append(
                "subtype-rescue branch stays stage2-only under the current proxy deltas."
            )
    elif branch_skipped is not None:
        reason = branch_skipped.get("reason", "unknown reason")
        recommendations.append(
            "subtype-rescue branch was skipped before GPU training because the hint path "
            f"had zero treatment effect ({reason}). Do not run stage3_subtype_rescue."
        )

    if "stage3" in missing_groups:
        recommendations.append(
            "stage3 artifacts are still missing; rerun this analyzer after canonical stage3 bestproxy completes."
        )
    if "branch" in missing_groups:
        recommendations.append(
            "branch artifacts are still missing; rerun this analyzer after subtype-rescue stage2 bestproxy completes."
        )

    return recommendations, branch_promotion


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
    branch_skipped_json: str | Path | None = None,
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
    branch_skipped = _load_skipped_artifact(
        None if branch_skipped_json is None else Path(branch_skipped_json)
    )
    branch = _load_eval_group(
        name="branch",
        hard_path=Path(branch_hard_eval),
        all_path=Path(branch_all_eval),
        allow_missing_group=allow_partial or branch_skipped is not None,
    )
    if branch is not None and branch_skipped is not None:
        raise SystemExit(
            "branch: bestproxy evals should not coexist with a skipped artifact"
        )

    missing_groups = [
        name
        for name, group in (("stage3", stage3), ("branch", branch))
        if group is None
        and not (name == "branch" and branch_skipped is not None)
    ]
    status = "partial" if missing_groups else "complete"
    final_selection = _maybe_load_optional(
        None if final_selection_json is None else Path(final_selection_json)
    )
    recommendations, branch_promotion = _build_recommendations(
        stage2={
            **stage2,
            "selection": _maybe_load_optional(
                None if stage2_selection_json is None else Path(stage2_selection_json)
            ),
        },
        stage3=None
        if stage3 is None
        else {
            **stage3,
            "selection": _maybe_load_optional(
                None if stage3_selection_json is None else Path(stage3_selection_json)
            ),
        },
        branch=None
        if branch is None
        else {
            **branch,
            "selection": _maybe_load_optional(
                None if branch_selection_json is None else Path(branch_selection_json)
            ),
        },
        branch_skipped=branch_skipped,
        final_selection=final_selection,
        missing_groups=missing_groups,
    )

    stage2_payload = {
        **stage2,
        "selection": _maybe_load_optional(
            None if stage2_selection_json is None else Path(stage2_selection_json)
        ),
    }
    stage3_payload = None
    if stage3 is not None:
        stage3_payload = {
            **stage3,
            "selection": _maybe_load_optional(
                None if stage3_selection_json is None else Path(stage3_selection_json)
            ),
        }
    branch_payload = None
    if branch is not None:
        branch_payload = {
            **branch,
            "selection": _maybe_load_optional(
                None if branch_selection_json is None else Path(branch_selection_json)
            ),
        }

    return {
        "status": status,
        "missing_groups": missing_groups,
        "stage2": stage2_payload,
        "stage3": stage3_payload,
        "branch": branch_payload,
        "branch_skipped": branch_skipped,
        "branch_promotion_preview": branch_promotion,
        "final_selection": final_selection,
        "recommendations": recommendations,
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
        "--branch-skipped-json",
        default="data/processed/stage2_subtype_rescue_skipped.json",
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
        branch_skipped_json=args.branch_skipped_json,
        final_selection_json=args.final_selection_json,
        allow_partial=args.allow_partial,
    )

    if args.output:
        write_json(args.output, payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
