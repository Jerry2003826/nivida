from __future__ import annotations

import json
from pathlib import Path

from scripts.plan_research_next_steps import build_research_plan


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _solver_summary() -> dict:
    return {
        "equation_template": {
            "overall": {
                "n": 100,
                "top1_accuracy": 0.08,
                "oracle_at_k": 0.10,
                "operator_gap_oracle_miss_count": 25,
                "ranker_miss_oracle_hit_count": 1,
                "safe_override_possible_rate": 0.02,
            }
        },
        "bit_permutation": {
            "overall": {
                "n": 80,
                "top1_accuracy": 0.38,
                "oracle_at_k": 0.44,
                "operator_gap_oracle_miss_count": 20,
                "ranker_miss_oracle_hit_count": 5,
                "safe_override_possible_rate": 0.38,
            }
        },
    }


def test_planner_blocks_gpu_when_readiness_is_missing(tmp_path: Path) -> None:
    plan = build_research_plan(
        readiness_path=tmp_path / "missing_readiness.json",
        batch_gate_path=tmp_path / "missing_gate.json",
        solver_summary_path=_write_json(tmp_path / "solver.json", _solver_summary()),
        lb_log_path=tmp_path / "missing_lb.json",
    )

    assert plan["primary_action"] == "run_no_gpu_readiness"
    assert plan["gpu_allowed"] is False
    assert "make no-gpu-readiness" in plan["next_commands"]


def test_planner_recommends_batch1_eval_when_ready_but_unscored(tmp_path: Path) -> None:
    readiness = _write_json(tmp_path / "ready.json", {"status": "pass", "ready_for_gpu": True, "failures": []})

    plan = build_research_plan(
        readiness_path=readiness,
        batch_gate_path=tmp_path / "missing_gate.json",
        solver_summary_path=_write_json(tmp_path / "solver.json", _solver_summary()),
        lb_log_path=tmp_path / "missing_lb.json",
    )

    assert plan["primary_action"] == "run_gpu_eval_batch1"
    assert plan["gpu_allowed"] is True
    assert any("run_cloud_eval_batch1.sh" in command for command in plan["next_commands"])


def test_planner_surfaces_package_command_when_gate_passes(tmp_path: Path) -> None:
    readiness = _write_json(tmp_path / "ready.json", {"status": "pass", "ready_for_gpu": True, "failures": []})
    gate = _write_json(
        tmp_path / "gate.json",
        {
            "ready_to_submit": True,
            "submit_candidate": "answer_only_continuation",
            "registry_candidate": "answer_only_continuation",
            "package_command": "python scripts/validate_submission.py --adapter-dir artifacts/answer --package-output artifacts/submissions/batch1_answer.zip",
            "gate_reason": "pass",
        },
    )

    plan = build_research_plan(
        readiness_path=readiness,
        batch_gate_path=gate,
        solver_summary_path=_write_json(tmp_path / "solver.json", _solver_summary()),
        lb_log_path=tmp_path / "missing_lb.json",
    )

    assert plan["primary_action"] == "package_submit_candidate"
    assert plan["gpu_allowed"] is False
    assert plan["submit_candidate"] == "answer_only_continuation"
    assert plan["next_commands"][0].startswith("python scripts/validate_submission.py")


def test_planner_turns_failed_gate_into_cpu_solver_exploration(tmp_path: Path) -> None:
    readiness = _write_json(tmp_path / "ready.json", {"status": "pass", "ready_for_gpu": True, "failures": []})
    gate = _write_json(
        tmp_path / "gate.json",
        {
            "ready_to_submit": False,
            "submit_candidate": None,
            "gate_reason": "no submit candidate passed ranking gate",
        },
    )

    plan = build_research_plan(
        readiness_path=readiness,
        batch_gate_path=gate,
        solver_summary_path=_write_json(tmp_path / "solver.json", _solver_summary()),
        lb_log_path=tmp_path / "missing_lb.json",
    )

    assert plan["primary_action"] == "cpu_solver_breakout"
    assert plan["gpu_allowed"] is False
    track_names = [track["name"] for track in plan["exploration_tracks"]]
    assert track_names[:2] == ["equation_operator_dsl", "bit_ranker_v2"]
    assert any("run_solver_breakout_v2.py" in command for command in plan["next_commands"])


def test_planner_prioritizes_eval_recalibration_after_public_stalls(tmp_path: Path) -> None:
    readiness = _write_json(tmp_path / "ready.json", {"status": "pass", "ready_for_gpu": True, "failures": []})
    gate = _write_json(
        tmp_path / "gate.json",
        {"ready_to_submit": False, "gate_reason": "no submit candidate passed ranking gate"},
    )
    lb_log = _write_json(
        tmp_path / "lb.json",
        {
            "entries": [
                {"candidate": "a", "public_score": 0.57, "exact_report": {"official_verify_accuracy": 0.61}},
                {"candidate": "b", "public_score": 0.57, "exact_report": {"official_verify_accuracy": 0.62}},
            ]
        },
    )

    plan = build_research_plan(
        readiness_path=readiness,
        batch_gate_path=gate,
        solver_summary_path=_write_json(tmp_path / "solver.json", _solver_summary()),
        lb_log_path=lb_log,
    )

    assert plan["primary_action"] == "recalibrate_local_eval"
    assert plan["gpu_allowed"] is False
    assert plan["public_stall_count"] == 2
