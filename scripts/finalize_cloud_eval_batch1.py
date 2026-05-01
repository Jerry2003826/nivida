from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.candidate_registry import candidate_by_name, load_registry  # noqa: E402

DEFAULT_PREDICTIONS_ROOT = Path("data/processed/vllm_exact_eval_v3_batch1")
DEFAULT_OUTPUT_ROOT = Path("data/processed/eval/vllm_exact_eval_v3_batch1")
DEFAULT_PREFERRED_EVAL = "combined_balanced_48pf"
DEFAULT_REGISTRY = Path("configs/research_breakout_candidates.json")
REQUIRED_BASELINE = "official_balanced"
LEGACY_CANDIDATE_ALIASES = {
    "answer_final": "answer_only_continuation",
    "short_trace_final": "short_trace_continuation",
}


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_scoring(predictions_root: Path, output_root: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/score_vllm_exact_eval_outputs.py",
            "--predictions-root",
            str(predictions_root),
            "--output-root",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def _choose_gate_eval(evals: dict[str, Any], preferred_eval: str) -> tuple[str | None, str]:
    if preferred_eval in evals:
        return preferred_eval, "preferred"
    for fallback in ("proxy_all_balanced_64pf", "hard_triad_full", "smoke_head6"):
        if fallback in evals:
            return fallback, f"fallback: {fallback}"
    if evals:
        name = sorted(evals)[0]
        return name, f"fallback: first available {name}"
    return None, "no evals found"


def _resolve_ranking_path(raw_path: str | None, output_root: Path, eval_name: str) -> Path | None:
    if raw_path:
        path = Path(raw_path)
        return path if path.is_absolute() else REPO_ROOT / path
    fallback = output_root / eval_name / "ranking.json"
    return fallback if fallback.exists() else None


def _find_submit_row(ranking: dict[str, Any], submit_candidate: str | None) -> dict[str, Any] | None:
    for row in ranking.get("rows", []):
        if submit_candidate is not None and row.get("model") == submit_candidate:
            return row
        if submit_candidate is None and row.get("submit_candidate"):
            return row
    return None


def _registry_candidate_name(model_name: str | None, registry: dict[str, Any] | None) -> str | None:
    if model_name is None:
        return None
    if registry is None:
        return model_name
    candidates = candidate_by_name(registry)
    if model_name in candidates:
        return model_name
    alias = LEGACY_CANDIDATE_ALIASES.get(model_name)
    if alias in candidates:
        return alias
    return model_name


def _candidate_metadata(
    model_name: str | None,
    registry: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any]]:
    registry_name = _registry_candidate_name(model_name, registry)
    if registry is None or registry_name is None:
        return registry_name, {}
    return registry_name, candidate_by_name(registry).get(registry_name, {})


def _package_command(
    *,
    candidate_name: str,
    metadata: dict[str, Any],
    output_root: Path,
) -> str | None:
    adapter_path = str(metadata.get("adapter_path") or "")
    if not adapter_path:
        return None
    config_path = str(metadata.get("config_path") or "configs/train_stage2_official_balanced_answer_only.yaml")
    validation_output = output_root / f"submission_validation_{candidate_name}.json"
    zip_output = Path("artifacts/submissions") / f"batch1_{candidate_name}.zip"
    parts = [
        sys.executable,
        "scripts/validate_submission.py",
        "--config",
        config_path,
        "--adapter-dir",
        adapter_path,
        "--output",
        str(validation_output),
        "--package-output",
        str(zip_output),
    ]
    return " ".join(shlex.quote(part) for part in parts)


def build_gate_summary(
    *,
    score_manifest: dict[str, Any],
    output_root: Path,
    preferred_eval: str = DEFAULT_PREFERRED_EVAL,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evals = dict(score_manifest.get("evals", {}))
    gate_eval, gate_eval_reason = _choose_gate_eval(evals, preferred_eval)
    if gate_eval is None:
        return {
            "status": "fail",
            "ready_to_submit": False,
            "gate_eval": None,
            "gate_eval_reason": gate_eval_reason,
            "submit_candidate": None,
            "gate_reason": "no scored evals found",
            "evals": evals,
        }

    gate_info = evals[gate_eval]
    ranking_path = _resolve_ranking_path(gate_info.get("ranking"), output_root, gate_eval)
    ranking = {} if ranking_path is None or not ranking_path.exists() else _load_json(ranking_path)
    baseline = str(gate_info.get("baseline") or ranking.get("baseline") or "")
    submit_candidate = gate_info.get("submit_candidate")
    submit_row = _find_submit_row(ranking, submit_candidate)
    if submit_candidate is None and submit_row is not None:
        submit_candidate = submit_row.get("model")
    registry_candidate, metadata = _candidate_metadata(submit_candidate, registry)

    ready = submit_candidate is not None and submit_row is not None
    gate_reason = "pass" if ready else "no submit candidate passed ranking gate"
    if baseline != REQUIRED_BASELINE:
        ready = False
        submit_candidate = None
        gate_reason = f"gate eval baseline must be {REQUIRED_BASELINE}, got {baseline or 'missing'}"
    elif ready and not metadata.get("adapter_path"):
        ready = False
        gate_reason = f"submit candidate lacks registry adapter_path: {submit_candidate}"
    elif ready and metadata.get("submission_safe") is False:
        ready = False
        gate_reason = f"submit candidate is marked submission_unsafe: {registry_candidate}"

    package_command = None
    if ready and registry_candidate is not None:
        package_command = _package_command(
            candidate_name=registry_candidate,
            metadata=metadata,
            output_root=output_root,
        )
    return {
        "status": "pass",
        "ready_to_submit": ready,
        "required_baseline": REQUIRED_BASELINE,
        "gate_eval": gate_eval,
        "gate_eval_reason": gate_eval_reason,
        "baseline": baseline,
        "submit_candidate": submit_candidate if ready else None,
        "registry_candidate": registry_candidate if ready else None,
        "submit_candidate_metadata": metadata if ready else None,
        "submit_row": submit_row if ready else None,
        "package_command": package_command,
        "gate_reason": gate_reason,
        "ranking": None if ranking_path is None else str(ranking_path),
        "evals": {
            name: {
                "baseline": info.get("baseline"),
                "submit_candidate": info.get("submit_candidate"),
                "ranking": info.get("ranking"),
            }
            for name, info in sorted(evals.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score and gate cloud batch1 vLLM exact-eval outputs locally.")
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preferred-eval", default=DEFAULT_PREFERRED_EVAL)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--skip-scoring", action="store_true", help="Use an existing score_manifest.json.")
    args = parser.parse_args(argv)

    predictions_root = _repo_path(args.predictions_root)
    output_root = _repo_path(args.output_root)
    if not args.skip_scoring:
        _run_scoring(predictions_root, output_root)

    manifest_path = output_root / "score_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing score manifest: {manifest_path}")
    summary = build_gate_summary(
        score_manifest=_load_json(manifest_path),
        output_root=output_root,
        preferred_eval=args.preferred_eval,
        registry=load_registry(_repo_path(args.registry)),
    )
    summary_path = _repo_path(args.summary_json) if args.summary_json else output_root / "batch1_gate_summary.json"
    _write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "ready_to_submit": summary["ready_to_submit"],
                "submit_candidate": summary["submit_candidate"],
                "registry_candidate": summary["registry_candidate"],
                "gate_reason": summary["gate_reason"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
