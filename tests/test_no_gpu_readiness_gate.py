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
        "python scripts/build_stage2_answer_focused_data.py",
        "python scripts/recheck_chat_template_sha16.py --output data/processed/recheck_chat_template_sha16.json",
        "sh scripts/check_prompt_suffix_alignment.sh data/processed/stage2_answer_only_train.jsonl data/processed/stage2_answer_only_valid.jsonl data/processed/stage2_short_trace_train.jsonl data/processed/stage2_short_trace_valid.jsonl",
        "python scripts/audit_teacher_gate_extractor_parity.py --train-jsonl ../data/processed/stage2_official_train_no_hard_valid.jsonl --allow-rerun-chain-search",
        "python -m pytest tests/test_equation_template_diagnostic.py tests/test_bit_permutation_diagnostic.py tests/test_cloud_run_scripts.py tests/test_shell_syntax.py tests/test_stage2_answer_focused_builder.py -q",
        "python -m pytest -q",
        "git diff --check",
        "git diff --name-only -- docs/solver_coverage_audit_latest.md docs/equation_template_diagnostic_latest.md docs/bit_permutation_diagnostic_latest.md",
    ]


def test_teacher_parity_insufficient_evidence_is_known_blocker(tmp_path: Path) -> None:
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
    assert report["status"] == "pass_with_known_blockers"
    assert report["ready_for_gpu"] is True
    assert report["known_blockers"] == [
        {
            "step": "teacher_gate_extractor_parity",
            "status": "insufficient_evidence",
            "reason": "stage2 provenance missing or mismatched",
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
