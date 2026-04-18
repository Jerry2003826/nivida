"""Pick the final adapter (stage2 or stage3) for submission.

Compares stage2 / stage3 proxy eval artifacts and copies the winning adapter
into ``--output-adapter-dir``. The selection is resolution-aware: small deltas
within half-sample tolerance are treated as ties and fall back to hard-triad
tie-break, otherwise prefer stage2.

Selection logic
---------------

- Primary metric: all-family proxy ``competition_correct_rate``.
  Tolerance ``0.5 / num_examples`` (half a sample at the proxy's
  resolution). If ``|all_delta|`` exceeds the tolerance, the winner is
  taken directly from the all-family proxy.
- Tie-break: hard-triad proxy ``competition_correct_rate`` with the same
  half-sample tolerance. Only applies when the all-family proxies are
  within tolerance.
- Default on complete ties: stage2. Stage2 is the safer choice because
  stage3 repair can overfit to hard-triad failure modes.

The coverage of both proxies is checked on load; any missing / unexpected /
duplicate id causes a SystemExit so a corrupted proxy cannot silently swing
the decision.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def load_eval(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a JSON object")

    coverage = payload.get("coverage", {})
    if not isinstance(coverage, dict):
        raise SystemExit(f"{path}: 'coverage' must be a dict")

    for field in ("num_missing", "num_unexpected", "num_duplicate"):
        value = coverage.get(field, 0)
        if value != 0:
            raise SystemExit(
                f"{path}: coverage.{field} = {value}, refusing to select from a "
                "proxy artifact whose prediction coverage is not complete."
            )

    if "competition_correct_rate" not in payload:
        raise SystemExit(f"{path}: missing 'competition_correct_rate'")
    if "num_examples" not in payload:
        raise SystemExit(f"{path}: missing 'num_examples'")

    return {
        "path": str(path),
        "competition_correct_rate": float(payload["competition_correct_rate"]),
        "num_examples": int(payload["num_examples"]),
        "coverage": coverage,
    }


def choose_adapter(
    stage2_all: dict[str, Any],
    stage3_all: dict[str, Any],
    stage2_hard: dict[str, Any],
    stage3_hard: dict[str, Any],
) -> dict[str, Any]:
    if stage2_all["num_examples"] != stage3_all["num_examples"]:
        raise SystemExit(
            "stage2 / stage3 all-family proxy num_examples mismatch: "
            f"stage2={stage2_all['num_examples']} vs stage3={stage3_all['num_examples']}"
        )
    if stage2_hard["num_examples"] != stage3_hard["num_examples"]:
        raise SystemExit(
            "stage2 / stage3 hard-triad proxy num_examples mismatch: "
            f"stage2={stage2_hard['num_examples']} vs stage3={stage3_hard['num_examples']}"
        )

    all_tol = 0.5 / max(1, stage2_all["num_examples"])
    hard_tol = 0.5 / max(1, stage2_hard["num_examples"])

    all_delta = (
        stage3_all["competition_correct_rate"] - stage2_all["competition_correct_rate"]
    )
    hard_delta = (
        stage3_hard["competition_correct_rate"] - stage2_hard["competition_correct_rate"]
    )

    base = {
        "all_delta": all_delta,
        "hard_delta": hard_delta,
        "all_tol": all_tol,
        "hard_tol": hard_tol,
    }

    if all_delta > all_tol:
        return {"selected_stage": "stage3", "rule": "all_family_primary", **base}
    if all_delta < -all_tol:
        return {"selected_stage": "stage2", "rule": "all_family_primary", **base}
    if hard_delta > hard_tol:
        return {"selected_stage": "stage3", "rule": "hard_triad_tiebreak", **base}
    return {"selected_stage": "stage2", "rule": "prefer_stage2_on_tie", **base}


def copy_adapter(src: str, dst: str) -> None:
    src_path = Path(src)
    if not src_path.is_dir():
        raise SystemExit(f"adapter source not found: {src}")
    dst_path = Path(dst)
    if dst_path.exists():
        shutil.rmtree(dst_path)
    shutil.copytree(src_path, dst_path)


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
