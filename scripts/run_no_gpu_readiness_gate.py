from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Callable, Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("data/processed/no_gpu_readiness_gate.json")
TEACHER_PARITY_OUTPUT = Path("data/processed/audit_teacher_gate_extractor_parity.json")
TEACHER_PARITY_TRAIN_JSONL = Path("../data/processed/stage2_official_train_no_hard_valid.jsonl")
REBUILD_STAGE2_TEACHER_INPUTS_SUMMARY = Path(
    "data/processed/rebuild_stage2_teacher_inputs_summary.json"
)
ANSWER_FOCUSED_OUTPUTS = (
    Path("data/processed/stage2_answer_only_train.jsonl"),
    Path("data/processed/stage2_answer_only_valid.jsonl"),
    Path("data/processed/stage2_short_trace_train.jsonl"),
    Path("data/processed/stage2_short_trace_valid.jsonl"),
)
TRACKED_REPORT_PATHS = (
    Path("docs/solver_coverage_audit_latest.md"),
    Path("docs/equation_template_diagnostic_latest.md"),
    Path("docs/bit_permutation_diagnostic_latest.md"),
)
CLOUD_EVAL_PREFLIGHT_PLAN = Path("data/processed/cloud_eval_preflight_plan.json")
RESEARCH_REGISTRY = Path("configs/research_breakout_candidates.json")
RESEARCH_BREAKOUT_DIR = Path("data/processed/research_breakout")
GENERATED_FILES = (
    REBUILD_STAGE2_TEACHER_INPUTS_SUMMARY,
    Path("../data/processed/official_train_tagged.jsonl"),
    Path("../data/processed/official_train_tagged.jsonl.provenance.json"),
    Path("../data/splits/official/splits.json"),
    TEACHER_PARITY_TRAIN_JSONL,
    Path("../data/processed/stage2_official_train_no_hard_valid.jsonl.provenance.json"),
    Path("../data/processed/stage2_official_valid_hard_triad.jsonl"),
    Path("../data/processed/stage2_official_valid_hard_triad.jsonl.provenance.json"),
    Path("data/processed/local_eval_manifests/manifest_summary.json"),
    Path("data/processed/solver_coverage_records.csv"),
    Path("data/processed/equation_template_diagnostic.json"),
    Path("data/processed/equation_template_diagnostic.csv"),
    Path("data/processed/bit_permutation_diagnostic.json"),
    Path("data/processed/bit_permutation_diagnostic.csv"),
    Path("data/processed/recheck_chat_template_sha16.json"),
    CLOUD_EVAL_PREFLIGHT_PLAN,
    TEACHER_PARITY_OUTPUT,
    *ANSWER_FOCUSED_OUTPUTS,
    RESEARCH_BREAKOUT_DIR / "mixed_answer_short_train.jsonl",
    RESEARCH_BREAKOUT_DIR / "mixed_answer_short_valid.jsonl",
    RESEARCH_BREAKOUT_DIR / "equation_rescue_train.jsonl",
    RESEARCH_BREAKOUT_DIR / "equation_rescue_valid.jsonl",
    RESEARCH_BREAKOUT_DIR / "bit_rescue_train.jsonl",
    RESEARCH_BREAKOUT_DIR / "bit_rescue_valid.jsonl",
    RESEARCH_BREAKOUT_DIR / "eq_bit_rescue_train.jsonl",
    RESEARCH_BREAKOUT_DIR / "eq_bit_rescue_valid.jsonl",
    RESEARCH_BREAKOUT_DIR / "research_rescue_data_summary.json",
    *TRACKED_REPORT_PATHS,
)


@dataclass(frozen=True)
class GateStep:
    name: str
    command: list[str]


@dataclass(frozen=True)
class CommandOutcome:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[list[str]], CommandOutcome]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rel(path: str | Path, *, repo_root: Path) -> str:
    target = Path(path)
    if target.is_absolute():
        try:
            return target.relative_to(repo_root).as_posix()
        except ValueError:
            return str(target)
    return target.as_posix()


def _truncate(value: str, *, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json_if_exists(path: str | Path) -> Any | None:
    target = Path(path)
    if not target.is_file():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def _default_runner(command: list[str], *, cwd: Path) -> CommandOutcome:
    actual = [sys.executable, *command[1:]] if command and command[0] == "python" else command
    completed = subprocess.run(
        actual,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return CommandOutcome(
        command=command,
        returncode=int(completed.returncode),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _wrap_runner(
    runner: Callable[..., CommandOutcome] | None,
    *,
    repo_root: Path,
) -> Callable[[list[str]], CommandOutcome]:
    if runner is None:
        return lambda command: _default_runner(command, cwd=repo_root)

    def _wrapped(command: list[str]) -> CommandOutcome:
        return runner(command, cwd=repo_root)

    return _wrapped


def _python(*args: str) -> list[str]:
    return ["python", *args]


def _posix(path: str | Path) -> str:
    return Path(path).as_posix()


def planned_steps(mode: str = "full") -> list[GateStep]:
    if mode not in {"full", "fast"}:
        raise ValueError(f"Unsupported mode: {mode}")

    fast_steps = [
        GateStep(
            "research_registry_check",
            _python(
                "scripts/build_research_candidate_registry.py",
                "--check",
                "--output",
                _posix(RESEARCH_REGISTRY),
            ),
        ),
        GateStep(
            "cloud_eval_preflight_plan",
            _python(
                "scripts/check_cloud_eval_inputs.py",
                "--dry-run",
                "--eval-inputs",
                "smoke_head6",
                "--candidate",
                "answer_final=artifacts/adapter_stage2_official_balanced_answer_only",
                "--output",
                _posix(CLOUD_EVAL_PREFLIGHT_PLAN),
            ),
        ),
        GateStep(
            "answer_focused_data_dry_run",
            _python("scripts/build_stage2_answer_focused_data.py", "--dry-run"),
        ),
        GateStep(
            "targeted_pytest",
            _python(
                "-m",
                "pytest",
                "tests/test_research_breakout.py",
                "tests/test_equation_template_diagnostic.py",
                "tests/test_bit_permutation_diagnostic.py",
                "tests/test_cloud_eval_preflight.py",
                "tests/test_cloud_run_scripts.py",
                "tests/test_shell_syntax.py",
                "tests/test_stage2_annotation_provenance.py",
                "tests/test_stage2_answer_focused_builder.py",
                "-q",
            ),
        ),
        GateStep("git_diff_check", ["git", "diff", "--check"]),
        GateStep(
            "tracked_report_drift",
            [
                "git",
                "diff",
                "--name-only",
                "--",
                *(_posix(path) for path in TRACKED_REPORT_PATHS),
            ],
        ),
    ]
    if mode == "fast":
        return fast_steps

    return [
        GateStep("build_local_eval_manifests", _python("scripts/build_local_eval_manifests.py")),
        GateStep("audit_solver_coverage", _python("scripts/audit_solver_coverage.py")),
        GateStep("diagnose_equation_template", _python("scripts/diagnose_equation_template.py")),
        GateStep("diagnose_bit_permutation", _python("scripts/diagnose_bit_permutation.py")),
        GateStep(
            "research_registry_check",
            _python(
                "scripts/build_research_candidate_registry.py",
                "--check",
                "--output",
                _posix(RESEARCH_REGISTRY),
            ),
        ),
        GateStep("rebuild_stage2_teacher_inputs", _python("scripts/rebuild_stage2_teacher_inputs.py")),
        GateStep("build_stage2_answer_focused_data", _python("scripts/build_stage2_answer_focused_data.py")),
        GateStep("build_research_rescue_data", _python("scripts/build_research_rescue_data.py")),
        GateStep(
            "recheck_chat_template_sha16",
            _python(
                "scripts/recheck_chat_template_sha16.py",
                "--output",
                "data/processed/recheck_chat_template_sha16.json",
            ),
        ),
        GateStep(
            "cloud_eval_preflight_plan",
            _python(
                "scripts/check_cloud_eval_inputs.py",
                "--dry-run",
                "--eval-inputs",
                "smoke_head6",
                "--candidate",
                "answer_final=artifacts/adapter_stage2_official_balanced_answer_only",
                "--output",
                _posix(CLOUD_EVAL_PREFLIGHT_PLAN),
            ),
        ),
        GateStep(
            "check_prompt_suffix_alignment",
            [
                "sh",
                "scripts/check_prompt_suffix_alignment.sh",
                *(_posix(path) for path in ANSWER_FOCUSED_OUTPUTS),
            ],
        ),
        GateStep(
            "teacher_gate_extractor_parity",
            _python(
                "scripts/audit_teacher_gate_extractor_parity.py",
                "--train-jsonl",
                _posix(TEACHER_PARITY_TRAIN_JSONL),
                "--allow-rerun-chain-search",
            ),
        ),
        GateStep(
            "targeted_pytest",
            _python(
                "-m",
                "pytest",
                "tests/test_research_breakout.py",
                "tests/test_equation_template_diagnostic.py",
                "tests/test_bit_permutation_diagnostic.py",
                "tests/test_cloud_eval_preflight.py",
                "tests/test_cloud_run_scripts.py",
                "tests/test_shell_syntax.py",
                "tests/test_stage2_annotation_provenance.py",
                "tests/test_stage2_answer_focused_builder.py",
                "-q",
            ),
        ),
        GateStep("full_pytest", _python("-m", "pytest", "-q")),
        GateStep("git_diff_check", ["git", "diff", "--check"]),
        GateStep(
            "tracked_report_drift",
            [
                "git",
                "diff",
                "--name-only",
                "--",
                *(_posix(path) for path in TRACKED_REPORT_PATHS),
            ],
        ),
    ]


def _step_payload(
    step: GateStep,
    *,
    status: str,
    duration_seconds: float | None = None,
    outcome: CommandOutcome | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": step.name,
        "command": step.command,
        "status": status,
    }
    if duration_seconds is not None:
        payload["duration_seconds"] = round(duration_seconds, 3)
    if outcome is not None:
        payload.update(
            {
                "returncode": outcome.returncode,
                "stdout_tail": _truncate(outcome.stdout),
                "stderr_tail": _truncate(outcome.stderr),
            }
        )
    if message:
        payload["message"] = message
    return payload


def _teacher_parity_failure(repo_root: Path) -> dict[str, str] | None:
    payload = _read_json_if_exists(repo_root / TEACHER_PARITY_OUTPUT)
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("status") == "insufficient_evidence"
        and payload.get("reason") == "stage2 provenance missing or mismatched"
    ):
        return {
            "step": "teacher_gate_extractor_parity",
            "reason": "stage2 provenance missing or mismatched",
        }
    if payload.get("status") in {"pending_stage2_teacher_annotation", "insufficient_evidence"}:
        return {
            "step": "teacher_gate_extractor_parity",
            "reason": str(payload.get("reason", payload.get("status"))),
        }
    return None


def _clean_tracked_worktree(
    run: Callable[[list[str]], CommandOutcome],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for command, label in [
        (["git", "diff", "--quiet"], "unstaged tracked changes"),
        (["git", "diff", "--cached", "--quiet"], "staged tracked changes"),
    ]:
        outcome = run(command)
        if outcome.returncode != 0:
            failures.append(
                {
                    "step": "initial_worktree_clean_check",
                    "reason": label,
                    "command": command,
                }
            )
    return failures


def run_gate(
    *,
    mode: str = "full",
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    allow_dirty: bool = False,
    dry_run: bool = False,
    runner: Callable[..., CommandOutcome] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    steps = planned_steps(mode)
    run = _wrap_runner(runner, repo_root=root)

    step_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    known_blockers: list[dict[str, str]] = []

    if dry_run:
        report = {
            "status": "dry_run",
            "ready_for_gpu": False,
            "mode": mode,
            "dry_run": True,
            "timestamp_utc": _utc_now(),
            "known_blockers": [],
            "failures": [],
            "generated_files": [_rel(path, repo_root=root) for path in (*GENERATED_FILES, output)],
            "steps": [_step_payload(step, status="planned") for step in steps],
        }
        _write_json(output, report)
        return report

    if not allow_dirty:
        failures.extend(_clean_tracked_worktree(run))
        if failures:
            report = {
                "status": "fail",
                "ready_for_gpu": False,
                "mode": mode,
                "dry_run": False,
                "timestamp_utc": _utc_now(),
                "known_blockers": [],
                "failures": failures,
                "generated_files": [_rel(path, repo_root=root) for path in (*GENERATED_FILES, output)],
                "steps": [],
            }
            _write_json(output, report)
            return report

    for step in steps:
        started = perf_counter()
        outcome = run(step.command)
        duration = perf_counter() - started
        if outcome.returncode != 0:
            step_results.append(
                _step_payload(
                    step,
                    status="fail",
                    duration_seconds=duration,
                    outcome=outcome,
                )
            )
            failures.append(
                {
                    "step": step.name,
                    "reason": "command failed",
                    "returncode": outcome.returncode,
                }
            )
            break

        if step.name == "tracked_report_drift" and outcome.stdout.strip():
            drifted = [line.strip() for line in outcome.stdout.splitlines() if line.strip()]
            step_results.append(
                _step_payload(
                    step,
                    status="fail",
                    duration_seconds=duration,
                    outcome=outcome,
                    message="Tracked canonical reports changed; review and commit the drift before opening GPU.",
                )
            )
            failures.append(
                {
                    "step": step.name,
                    "reason": "tracked report drift",
                    "details": drifted,
                }
            )
            break

        status = "pass"
        message = None
        if step.name == "teacher_gate_extractor_parity":
            failure = _teacher_parity_failure(root)
            if failure is not None:
                status = "fail"
                message = failure["reason"]
                step_results.append(
                    _step_payload(
                        step,
                        status=status,
                        duration_seconds=duration,
                        outcome=outcome,
                        message=message,
                    )
                )
                failures.append(failure)
                break
        step_results.append(
            _step_payload(
                step,
                status=status,
                duration_seconds=duration,
                outcome=outcome,
                message=message,
            )
        )

    status = "fail" if failures else "pass_with_known_blockers" if known_blockers else "pass"
    report = {
        "status": status,
        "ready_for_gpu": not failures,
        "mode": mode,
        "dry_run": False,
        "timestamp_utc": _utc_now(),
        "known_blockers": known_blockers,
        "failures": failures,
        "generated_files": [_rel(path, repo_root=root) for path in (*GENERATED_FILES, output)],
        "steps": step_results,
    }
    _write_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local no-GPU readiness gate before opening GPU.")
    parser.add_argument("--mode", choices=["full", "fast"], default="full")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    report = run_gate(
        mode=args.mode,
        output_path=args.output,
        allow_dirty=bool(args.allow_dirty),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
