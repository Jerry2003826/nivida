from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_READINESS = Path("data/processed/no_gpu_readiness_gate.json")
DEFAULT_BATCH_GATE = Path("data/processed/eval/vllm_exact_eval_v3_batch1/batch1_gate_summary.json")
DEFAULT_SOLVER_SUMMARY = Path("data/processed/solver_breakout_v2_full/summary.json")
DEFAULT_LB_LOG = Path("data/processed/eval/lb_correlation_log.json")
DEFAULT_OUTPUT_JSON = Path("data/processed/research_next_step_plan.json")
DEFAULT_OUTPUT_MD = Path("docs/research_next_step_plan_latest.md")
PUBLIC_BASELINE = 0.57


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _rate_metric(summary: dict[str, Any] | None, family: str, key: str) -> float:
    if not summary:
        return 0.0
    try:
        return float(summary.get(family, {}).get("overall", {}).get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _count_metric(summary: dict[str, Any] | None, family: str, key: str) -> int:
    if not summary:
        return 0
    try:
        return int(summary.get(family, {}).get("overall", {}).get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _n(summary: dict[str, Any] | None, family: str) -> int:
    return _count_metric(summary, family, "n")


def _operator_gap_rate(summary: dict[str, Any] | None, family: str) -> float:
    total = _n(summary, family)
    return _count_metric(summary, family, "operator_gap_oracle_miss_count") / total if total else 0.0


def _public_stall_count(lb_log: dict[str, Any] | None) -> int:
    if not lb_log:
        return 0
    entries = lb_log.get("entries", [])
    if not isinstance(entries, list):
        return 0
    count = 0
    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue
        public_score = entry.get("public_score")
        exact = entry.get("exact_report", {})
        exact_acc = exact.get("official_verify_accuracy") if isinstance(exact, dict) else None
        try:
            public_numeric = None if public_score is None else float(public_score)
            exact_numeric = None if exact_acc is None else float(exact_acc)
        except (TypeError, ValueError):
            continue
        if public_numeric is not None and public_numeric <= PUBLIC_BASELINE and exact_numeric is not None and exact_numeric > PUBLIC_BASELINE:
            count += 1
            continue
        break
    return count


def _exploration_tracks(solver_summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    equation_oracle = _rate_metric(solver_summary, "equation_template", "oracle_at_k")
    equation_gap_rate = _operator_gap_rate(solver_summary, "equation_template")
    bit_top1 = _rate_metric(solver_summary, "bit_permutation", "top1_accuracy")
    bit_oracle = _rate_metric(solver_summary, "bit_permutation", "oracle_at_k")
    bit_safe = _rate_metric(solver_summary, "bit_permutation", "safe_override_possible_rate")
    bit_ranker_miss = _count_metric(solver_summary, "bit_permutation", "ranker_miss_oracle_hit_count")

    tracks: list[dict[str, Any]] = []
    if equation_oracle < 0.20 or equation_gap_rate >= 0.15:
        tracks.append(
            {
                "name": "equation_operator_dsl",
                "priority": 1,
                "gpu_required": False,
                "reason": (
                    f"equation oracle@k={equation_oracle:.3f}, "
                    f"operator_gap_rate={equation_gap_rate:.3f}; current search space is the bottleneck"
                ),
                "next_focus": [
                    "cluster target_not_expressible examples",
                    "add only low-risk operator families",
                    "keep high-risk rows answer-only",
                ],
            }
        )
    if (bit_oracle - bit_top1) >= 0.02 or bit_ranker_miss > 0:
        tracks.append(
            {
                "name": "bit_ranker_v2",
                "priority": 2,
                "gpu_required": False,
                "reason": (
                    f"bit top1={bit_top1:.3f}, oracle@k={bit_oracle:.3f}, "
                    f"ranker_miss={bit_ranker_miss}; rerank headroom remains"
                ),
                "next_focus": [
                    "leave-one-out stability weighting",
                    "complexity penalty calibration",
                    "oracle-distance feature ablation",
                ],
            }
        )
    if bit_safe >= 0.25:
        tracks.append(
            {
                "name": "bit_rescue_training_data",
                "priority": 3,
                "gpu_required": False,
                "reason": f"bit safe_override_possible_rate={bit_safe:.3f}; enough high-confidence rows for answer-only rescue data",
                "next_focus": [
                    "refresh bit_rescue_v2 data",
                    "train only after batch1 exact eval is scored",
                    "keep solver finalizer research-only",
                ],
            }
        )
    tracks.append(
        {
            "name": "eval_correlation_log",
            "priority": 4,
            "gpu_required": False,
            "reason": "public/local correlation remains the long-term stop rule",
            "next_focus": [
                "append every Kaggle result",
                "pause training after two local wins without public movement",
            ],
        }
    )
    return tracks


def build_research_plan(
    *,
    readiness_path: Path = DEFAULT_READINESS,
    batch_gate_path: Path = DEFAULT_BATCH_GATE,
    solver_summary_path: Path = DEFAULT_SOLVER_SUMMARY,
    lb_log_path: Path = DEFAULT_LB_LOG,
) -> dict[str, Any]:
    readiness = _load_json(readiness_path)
    batch_gate = _load_json(batch_gate_path)
    solver_summary = _load_json(solver_summary_path)
    lb_log = _load_json(lb_log_path)
    tracks = _exploration_tracks(solver_summary)
    public_stalls = _public_stall_count(lb_log)

    plan: dict[str, Any] = {
        "schema_version": 1,
        "inputs": {
            "readiness": str(readiness_path),
            "batch_gate": str(batch_gate_path),
            "solver_summary": str(solver_summary_path),
            "lb_log": str(lb_log_path),
        },
        "readiness_status": None if readiness is None else readiness.get("status"),
        "ready_for_gpu": False if readiness is None else bool(readiness.get("ready_for_gpu")),
        "batch_gate_available": batch_gate is not None,
        "public_stall_count": public_stalls,
        "exploration_tracks": tracks,
    }

    if not plan["ready_for_gpu"]:
        plan.update(
            {
                "primary_action": "run_no_gpu_readiness",
                "gpu_allowed": False,
                "reason": "local no-GPU readiness is missing or not passing",
                "next_commands": ["make no-gpu-readiness"],
            }
        )
        return plan

    if batch_gate is None:
        plan.update(
            {
                "primary_action": "run_gpu_eval_batch1",
                "gpu_allowed": True,
                "reason": "local gate is ready, but batch1 vLLM exact-eval has not been scored",
                "next_commands": [
                    "cd /workspace/nivida_h200_run",
                    "RUN_FULL=0 bash scripts/run_cloud_eval_batch1.sh",
                    "RUN_SMOKE=0 RUN_FULL=1 bash scripts/run_cloud_eval_batch1.sh",
                ],
            }
        )
        return plan

    if bool(batch_gate.get("ready_to_submit")):
        command = str(batch_gate.get("package_command") or "")
        plan.update(
            {
                "primary_action": "package_submit_candidate",
                "gpu_allowed": False,
                "reason": "batch1 gate found a submit-safe candidate above official_balanced",
                "submit_candidate": batch_gate.get("registry_candidate") or batch_gate.get("submit_candidate"),
                "next_commands": [command] if command else [],
            }
        )
        return plan

    if public_stalls >= 2:
        plan.update(
            {
                "primary_action": "recalibrate_local_eval",
                "gpu_allowed": False,
                "reason": "two or more recent local wins did not move public score",
                "next_commands": [
                    "python scripts/update_lb_correlation_log.py --help",
                    "python scripts/build_local_eval_manifests.py",
                    "python scripts/run_solver_breakout_v2.py --output-dir data/processed/solver_breakout_v2_full --output-md docs/solver_breakout_v2_full_latest.md",
                ],
            }
        )
        return plan

    plan.update(
        {
            "primary_action": "cpu_solver_breakout",
            "gpu_allowed": False,
            "reason": batch_gate.get("gate_reason", "batch1 gate did not find a submit-safe winner"),
            "next_commands": [
                "python scripts/run_solver_breakout_v2.py --output-dir data/processed/solver_breakout_v2_full --output-md docs/solver_breakout_v2_full_latest.md",
                "python scripts/build_research_rescue_data.py",
                "make no-gpu-readiness",
            ],
        }
    )
    return plan


def _write_markdown(path: Path, plan: dict[str, Any]) -> None:
    lines = [
        "# Research Next Step Plan",
        "",
        f"- primary_action: `{plan['primary_action']}`",
        f"- gpu_allowed: `{plan['gpu_allowed']}`",
        f"- reason: {plan['reason']}",
        f"- public_stall_count: `{plan.get('public_stall_count', 0)}`",
        "",
        "## Next Commands",
        "",
    ]
    for command in plan.get("next_commands", []):
        lines.append(f"- `{command}`")
    lines.extend(["", "## Exploration Tracks", ""])
    for track in plan.get("exploration_tracks", []):
        lines.append(f"### {track['priority']}. {track['name']}")
        lines.append(f"- gpu_required: `{track['gpu_required']}`")
        lines.append(f"- reason: {track['reason']}")
        for focus in track.get("next_focus", []):
            lines.append(f"- {focus}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan the next no-GPU/GPU research step from local gate artifacts.")
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--batch-gate", type=Path, default=DEFAULT_BATCH_GATE)
    parser.add_argument("--solver-summary", type=Path, default=DEFAULT_SOLVER_SUMMARY)
    parser.add_argument("--lb-log", type=Path, default=DEFAULT_LB_LOG)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args(argv)

    plan = build_research_plan(
        readiness_path=args.readiness,
        batch_gate_path=args.batch_gate,
        solver_summary_path=args.solver_summary,
        lb_log_path=args.lb_log,
    )
    _write_json(args.output_json, plan)
    _write_markdown(args.output_md, plan)
    print(json.dumps({"output_json": str(args.output_json), "primary_action": plan["primary_action"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
