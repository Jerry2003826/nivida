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


def _coverage_summary(payload: dict[str, Any]) -> dict[str, Any]:
    coverage = payload.get("coverage", {})
    if not isinstance(coverage, dict):
        raise SystemExit("coverage must be a JSON object")
    num_examples = int(payload["num_examples"])
    num_missing = int(coverage.get("num_missing", 0))
    num_unexpected = int(coverage.get("num_unexpected", 0))
    num_duplicate = int(coverage.get("num_duplicate", 0))
    coverage_errors = num_missing + num_unexpected + num_duplicate
    completeness = max(0.0, 1.0 - (coverage_errors / max(1, num_examples)))
    return {
        "num_missing": num_missing,
        "num_unexpected": num_unexpected,
        "num_duplicate": num_duplicate,
        "completeness": completeness if coverage_errors else 1.0,
    }


def load_promotion_eval(path: str | Path) -> dict[str, Any]:
    try:
        normalized = load_proxy_eval(path)
    except SystemExit as exc:
        message = str(exc)
        if "coverage." in message:
            raise SystemExit(f"{path}: coverage is incomplete") from exc
        raise
    raw_payload = read_json(path)
    if not isinstance(raw_payload, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    coverage = _coverage_summary(raw_payload)
    if coverage["completeness"] != 1.0:
        raise SystemExit(
            f"{path}: proxy coverage is incomplete "
            f"(missing={coverage['num_missing']}, "
            f"unexpected={coverage['num_unexpected']}, "
            f"duplicate={coverage['num_duplicate']})"
        )
    return {
        **normalized,
        "coverage_summary": coverage,
    }


def decide_branch_promotion(
    *,
    baseline_all: dict[str, Any],
    baseline_hard: dict[str, Any],
    branch_all: dict[str, Any],
    branch_hard: dict[str, Any],
) -> dict[str, Any]:
    if baseline_all["num_examples"] != branch_all["num_examples"]:
        raise SystemExit(
            "all-family proxy num_examples mismatch: "
            f"baseline={baseline_all['num_examples']} vs "
            f"branch={branch_all['num_examples']}"
        )
    if baseline_hard["num_examples"] != branch_hard["num_examples"]:
        raise SystemExit(
            "hard-triad proxy num_examples mismatch: "
            f"baseline={baseline_hard['num_examples']} vs "
            f"branch={branch_hard['num_examples']}"
        )

    n_all = int(baseline_all["num_examples"])
    n_hard = int(baseline_hard["num_examples"])
    all_tol = 0.5 / max(1, n_all)
    hard_tol = 0.5 / max(1, n_hard)
    all_delta = (
        float(branch_all["competition_correct_rate"])
        - float(baseline_all["competition_correct_rate"])
    )
    hard_delta = (
        float(branch_hard["competition_correct_rate"])
        - float(baseline_hard["competition_correct_rate"])
    )

    eps = 1e-12
    promote = all_delta >= (-all_tol - eps) and hard_delta >= (hard_tol - eps)
    return {
        "promote": promote,
        "baseline_all_rate": baseline_all["competition_correct_rate"],
        "branch_all_rate": branch_all["competition_correct_rate"],
        "baseline_hard_rate": baseline_hard["competition_correct_rate"],
        "branch_hard_rate": branch_hard["competition_correct_rate"],
        "all_delta": all_delta,
        "hard_delta": hard_delta,
        "all_delta_examples": all_delta * n_all,
        "hard_delta_examples": hard_delta * n_hard,
        "all_tol": all_tol,
        "hard_tol": hard_tol,
        "allowed_all_drop_examples": 0.5,
        "required_hard_delta_examples": 0.5,
        "n_all": n_all,
        "n_hard": n_hard,
        "coverage": {
            "baseline_all": baseline_all["coverage_summary"],
            "baseline_hard": baseline_hard["coverage_summary"],
            "branch_all": branch_all["coverage_summary"],
            "branch_hard": branch_hard["coverage_summary"],
        },
        "rule": {
            "all_family_floor": "branch_all >= baseline_all - 0.5 / N_all",
            "hard_triad_gain": "branch_hard >= baseline_hard + 0.5 / N_hard",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decide whether the subtype-rescue branch should advance."
    )
    parser.add_argument(
        "--baseline-hard-eval",
        default="data/processed/stage2_bestproxy_hard_eval.json",
    )
    parser.add_argument(
        "--baseline-all-eval",
        default="data/processed/stage2_bestproxy_all_eval.json",
    )
    parser.add_argument(
        "--branch-hard-eval",
        default="data/processed/stage2_subtype_rescue_bestproxy_hard_eval.json",
    )
    parser.add_argument(
        "--branch-all-eval",
        default="data/processed/stage2_subtype_rescue_bestproxy_all_eval.json",
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--exit-nonzero-on-no-promotion",
        action="store_true",
        help="Exit with code 1 when the branch does not qualify for promotion.",
    )
    args = parser.parse_args()

    payload = decide_branch_promotion(
        baseline_all=load_promotion_eval(args.baseline_all_eval),
        baseline_hard=load_promotion_eval(args.baseline_hard_eval),
        branch_all=load_promotion_eval(args.branch_all_eval),
        branch_hard=load_promotion_eval(args.branch_hard_eval),
    )
    if args.output:
        write_json(args.output, payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.exit_nonzero_on_no_promotion and not payload["promote"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
