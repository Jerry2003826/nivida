from __future__ import annotations

import json
from pathlib import Path

from scripts.run_no_gpu_readiness_gate import (
    CommandOutcome,
    planned_steps,
    run_gate,
)


def _successful_runner(
    command: list[str],
    *,
    cwd: Path,
) -> CommandOutcome:
    return CommandOutcome(command=command, returncode=0, stdout="", stderr="")


def test_full_mode_plans_required_command_order() -> None:
    steps = planned_steps("full")
    commands = [" ".join(step.command) for step in steps]

    assert commands == [
        "python scripts/build_local_eval_manifests.py",
        "python scripts/audit_solver_coverage.py",
        "python scripts/diagnose_equation_template.py",
        "python scripts/diagnose_bit_permutation.py",
        "python scripts/run_solver_breakout_v2.py --limit 64 --output-dir data/processed/solver_breakout_v2 --output-md docs/solver_breakout_v2_latest.md",
        "python scripts/build_research_candidate_registry.py --check --output configs/research_breakout_candidates.json",
        "python scripts/build_stage2_answer_focused_data.py --dry-run",
        "python scripts/build_research_rescue_data.py",
        "python scripts/recheck_chat_template_sha16.py --output data/processed/recheck_chat_template_sha16.json",
        "python scripts/check_cloud_eval_inputs.py --dry-run --eval-inputs smoke_head6 --candidate answer_final=artifacts/adapter_stage2_official_balanced_answer_only --candidate bit_rescue_v2_20260430_trained=artifacts/adapter_stage2_bit_rescue_v2 --output data/processed/cloud_eval_preflight_plan.json",
        "sh scripts/check_prompt_suffix_alignment.sh data/processed/stage2_answer_only_train.jsonl data/processed/stage2_answer_only_valid.jsonl data/processed/stage2_short_trace_train.jsonl data/processed/stage2_short_trace_valid.jsonl",
        "python scripts/audit_teacher_gate_extractor_parity.py --train-jsonl ../data/processed/stage2_official_train_no_hard_valid.jsonl",
        "python -m pytest tests/test_adapter_merge.py tests/test_research_breakout.py tests/test_equation_template_diagnostic.py tests/test_bit_permutation_diagnostic.py tests/test_solver_breakout_v2.py tests/test_cloud_eval_preflight.py tests/test_cloud_run_scripts.py tests/test_shell_syntax.py tests/test_stage2_annotation_provenance.py tests/test_stage2_answer_focused_builder.py -q",
        "python -m pytest -q",
        "git diff --check",
        "git diff --name-only -- docs/solver_coverage_audit_latest.md docs/equation_template_diagnostic_latest.md docs/bit_permutation_diagnostic_latest.md docs/solver_breakout_v2_latest.md",
    ]


def test_full_mode_can_plan_rerun_teacher_parity_with_timeout() -> None:
    cached_step = next(step for step in planned_steps("full") if step.name == "teacher_gate_extractor_parity")
    steps = planned_steps("full", teacher_parity_mode="rerun")
    teacher_step = next(step for step in steps if step.name == "teacher_gate_extractor_parity")

    assert "--allow-rerun-chain-search" not in cached_step.command
    assert cached_step.timeout_seconds == 600
    assert "--allow-rerun-chain-search" in teacher_step.command
    assert teacher_step.timeout_seconds == 600


def test_full_mode_refreshes_derived_data_only_when_requested() -> None:
    default_steps = planned_steps("full")
    refresh_steps = planned_steps("full", refresh_derived_data=True)

    default_command = next(step.command for step in default_steps if step.name == "build_stage2_answer_focused_data")
    refresh_command = next(step.command for step in refresh_steps if step.name == "build_stage2_answer_focused_data")

    assert "rebuild_stage2_teacher_inputs" not in {step.name for step in default_steps}
    assert "rebuild_stage2_teacher_inputs" in {step.name for step in refresh_steps}
    assert default_command == ["python", "scripts/build_stage2_answer_focused_data.py", "--dry-run"]
    assert refresh_command == ["python", "scripts/build_stage2_answer_focused_data.py"]


def test_step_timeout_is_reported_as_failure(tmp_path: Path) -> None:
    def runner(command: list[str], *, cwd: Path) -> CommandOutcome:
        if any(part.endswith("audit_teacher_gate_extractor_parity.py") for part in command):
            return CommandOutcome(command=command, returncode=124, stdout="", stderr="timed out after 600s")
        return CommandOutcome(command=command, returncode=0, stdout="", stderr="")

    report = run_gate(
        mode="full",
        repo_root=tmp_path,
        output_path=tmp_path / "gate.json",
        allow_dirty=True,
        teacher_parity_mode="rerun",
        runner=runner,
    )

    teacher_step = next(step for step in report["steps"] if step["name"] == "teacher_gate_extractor_parity")
    assert teacher_step["status"] == "timeout"
    assert teacher_step["timeout_seconds"] == 600
    assert report["failures"] == [
        {
            "step": "teacher_gate_extractor_parity",
            "reason": "command timed out",
            "timeout_seconds": 600,
        }
    ]


def test_teacher_parity_insufficient_evidence_fails_gate(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], *, cwd: Path) -> CommandOutcome:
        calls.append(command)
        if any(part.endswith("audit_teacher_gate_extractor_parity.py") for part in command):
            output = cwd / "data" / "processed" / "audit_teacher_gate_extractor_parity.json"
            output.parent.mkdir(parents=True)
            output.write_text(
                json.dumps(
                    {
                        "status": "insufficient_evidence",
                        "reason": "stage2 provenance missing or mismatched",
                    }
                ),
                encoding="utf-8",
            )
        return CommandOutcome(command=command, returncode=0, stdout="", stderr="")

    report = run_gate(
        mode="full",
        repo_root=tmp_path,
        output_path=tmp_path / "gate.json",
        allow_dirty=True,
        runner=runner,
    )

    assert calls
    assert report["status"] == "fail"
    assert report["ready_for_gpu"] is False
    assert report["known_blockers"] == []
    assert report["failures"] == [
        {
            "step": "teacher_gate_extractor_parity",
            "reason": "stage2 provenance missing or mismatched",
        }
    ]


def test_cached_teacher_parity_missing_support_pairs_has_actionable_failure(tmp_path: Path) -> None:
    def runner(command: list[str], *, cwd: Path) -> CommandOutcome:
        if any(part.endswith("audit_teacher_gate_extractor_parity.py") for part in command):
            return CommandOutcome(
                command=command,
                returncode=1,
                stdout="",
                stderr=(
                    "ValueError: Missing support_pairs/query_prediction in metadata.extras. "
                    "Re-run with --allow-rerun-chain-search to rebuild annotations."
                ),
            )
        return CommandOutcome(command=command, returncode=0, stdout="", stderr="")

    report = run_gate(
        mode="full",
        repo_root=tmp_path,
        output_path=tmp_path / "gate.json",
        allow_dirty=True,
        runner=runner,
    )

    assert report["status"] == "fail"
    assert report["failures"] == [
        {
            "step": "teacher_gate_extractor_parity",
            "reason": "cached support pairs missing",
            "next_step": "rerun with --teacher-parity-mode rerun",
        }
    ]


def test_tracked_report_drift_fails_gate(tmp_path: Path) -> None:
    def runner(command: list[str], *, cwd: Path) -> CommandOutcome:
        if command[:3] == ["git", "diff", "--name-only"]:
            return CommandOutcome(
                command=command,
                returncode=0,
                stdout="docs/solver_coverage_audit_latest.md\n",
                stderr="",
            )
        return CommandOutcome(command=command, returncode=0, stdout="", stderr="")

    report = run_gate(
        mode="full",
        repo_root=tmp_path,
        output_path=tmp_path / "gate.json",
        allow_dirty=True,
        runner=runner,
    )

    assert report["status"] == "fail"
    assert report["ready_for_gpu"] is False
    assert report["failures"] == [
        {
            "step": "tracked_report_drift",
            "reason": "tracked report drift",
            "details": ["docs/solver_coverage_audit_latest.md"],
        }
    ]


def test_dry_run_outputs_planned_steps_without_running_commands(tmp_path: Path) -> None:
    def runner(command: list[str], *, cwd: Path) -> CommandOutcome:
        raise AssertionError(f"dry-run should not execute {command}")

    report = run_gate(
        mode="full",
        repo_root=tmp_path,
        output_path=tmp_path / "gate.json",
        dry_run=True,
        runner=runner,
    )

    assert report["status"] == "dry_run"
    assert report["ready_for_gpu"] is False
    assert len(report["steps"]) == len(planned_steps("full"))
    assert all(step["status"] == "planned" for step in report["steps"])


def test_makefile_has_no_gpu_readiness_target() -> None:
    makefile = Path(__file__).resolve().parents[1] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "no-gpu-readiness:" in text
    assert "$(PYTHON) scripts/run_no_gpu_readiness_gate.py --mode full" in text
    assert "research-registry:" in text
    assert "research-rescue-data:" in text
