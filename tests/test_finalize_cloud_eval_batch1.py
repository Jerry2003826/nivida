from __future__ import annotations

import json
from pathlib import Path

from src.research.candidate_registry import build_default_registry
from scripts.finalize_cloud_eval_batch1 import build_gate_summary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_batch1_gate_summary_accepts_official_balanced_submit_candidate(tmp_path: Path) -> None:
    output_root = tmp_path / "eval"
    ranking_path = output_root / "combined_balanced_48pf" / "ranking.json"
    _write_json(
        ranking_path,
        {
            "baseline": "official_balanced",
            "rows": [
                {
                    "rank": 1,
                    "model": "answer_only_continuation",
                    "submit_candidate": True,
                    "pass_gate": True,
                    "official_verify_accuracy": 0.62,
                    "delta_vs_baseline": 0.05,
                    "boxed_valid_rate": 1.0,
                    "gate_reason": "pass",
                    "submission_safe": True,
                },
                {
                    "rank": 2,
                    "model": "official_balanced",
                    "submit_candidate": False,
                    "pass_gate": False,
                    "official_verify_accuracy": 0.57,
                    "delta_vs_baseline": 0.0,
                    "boxed_valid_rate": 1.0,
                    "gate_reason": "baseline",
                    "submission_safe": True,
                },
            ],
        },
    )
    score_manifest = {
        "evals": {
            "combined_balanced_48pf": {
                "baseline": "official_balanced",
                "ranking": str(ranking_path),
                "submit_candidate": "answer_only_continuation",
            }
        }
    }

    summary = build_gate_summary(
        score_manifest=score_manifest,
        output_root=output_root,
        preferred_eval="combined_balanced_48pf",
        registry=build_default_registry(),
    )

    assert summary["ready_to_submit"] is True
    assert summary["submit_candidate"] == "answer_only_continuation"
    assert summary["submit_candidate_metadata"]["adapter_path"] == "artifacts/adapter_stage2_official_balanced_answer_only"
    assert "scripts/validate_submission.py" in summary["package_command"]
    assert "--adapter-dir artifacts/adapter_stage2_official_balanced_answer_only" in summary["package_command"]
    assert summary["gate_eval"] == "combined_balanced_48pf"
    assert summary["submit_row"]["official_verify_accuracy"] == 0.62


def test_batch1_gate_summary_blocks_non_official_baseline(tmp_path: Path) -> None:
    output_root = tmp_path / "eval"
    ranking_path = output_root / "combined_balanced_48pf" / "ranking.json"
    _write_json(
        ranking_path,
        {
            "baseline": "b_thin",
            "rows": [
                {
                    "rank": 1,
                    "model": "answer_final",
                    "submit_candidate": True,
                    "pass_gate": True,
                    "official_verify_accuracy": 0.60,
                    "delta_vs_baseline": 0.06,
                    "boxed_valid_rate": 1.0,
                    "gate_reason": "pass",
                    "submission_safe": True,
                }
            ],
        },
    )
    score_manifest = {
        "evals": {
            "combined_balanced_48pf": {
                "baseline": "b_thin",
                "ranking": str(ranking_path),
                "submit_candidate": "answer_final",
            }
        }
    }

    summary = build_gate_summary(
        score_manifest=score_manifest,
        output_root=output_root,
        preferred_eval="combined_balanced_48pf",
        registry=build_default_registry(),
    )

    assert summary["ready_to_submit"] is False
    assert summary["submit_candidate"] is None
    assert "official_balanced" in summary["gate_reason"]


def test_batch1_gate_summary_maps_legacy_cloud_alias_to_registry_metadata(tmp_path: Path) -> None:
    output_root = tmp_path / "eval"
    ranking_path = output_root / "combined_balanced_48pf" / "ranking.json"
    _write_json(
        ranking_path,
        {
            "baseline": "official_balanced",
            "rows": [
                {
                    "rank": 1,
                    "model": "answer_final",
                    "submit_candidate": True,
                    "pass_gate": True,
                    "official_verify_accuracy": 0.61,
                    "delta_vs_baseline": 0.04,
                    "boxed_valid_rate": 1.0,
                    "gate_reason": "pass",
                    "submission_safe": True,
                }
            ],
        },
    )
    score_manifest = {
        "evals": {
            "combined_balanced_48pf": {
                "baseline": "official_balanced",
                "ranking": str(ranking_path),
                "submit_candidate": "answer_final",
            }
        }
    }

    summary = build_gate_summary(
        score_manifest=score_manifest,
        output_root=output_root,
        preferred_eval="combined_balanced_48pf",
        registry=build_default_registry(),
    )

    assert summary["ready_to_submit"] is True
    assert summary["submit_candidate"] == "answer_final"
    assert summary["registry_candidate"] == "answer_only_continuation"
    assert summary["submit_candidate_metadata"]["adapter_path"] == "artifacts/adapter_stage2_official_balanced_answer_only"
