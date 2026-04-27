from __future__ import annotations

import json
from pathlib import Path

from scripts.check_cloud_eval_inputs import run_checks


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_adapter(path: Path) -> None:
    path.mkdir(parents=True)
    path.joinpath("adapter_model.safetensors").write_bytes(b"weights")
    path.joinpath("adapter_config.json").write_text('{"r": 32}', encoding="utf-8")


def _write_required_repo_files(root: Path) -> None:
    for rel in [
        "scripts/check_cloud_vllm_env.sh",
        "scripts/eval_official_vllm_proxy.py",
        "scripts/run_cloud_vllm_exact_eval_v3.sh",
        "scripts/score_vllm_exact_eval_outputs.py",
        "scripts/write_cloud_artifact_manifest.py",
        "configs/train_stage2_official_balanced_answer_only.yaml",
    ]:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placeholder\n", encoding="utf-8")


def test_cloud_eval_preflight_passes_and_writes_hashes(tmp_path: Path) -> None:
    _write_required_repo_files(tmp_path)
    _write_jsonl(
        tmp_path / "data/processed/local_eval_manifests/smoke_6pf.jsonl",
        [
            {"id": "a", "target_answer": "1"},
            {"id": "b", "target_answer": "2"},
        ],
    )
    _write_adapter(tmp_path / "artifacts/answer")

    report = run_checks(
        repo_root=tmp_path,
        eval_inputs=["smoke_6pf"],
        candidates=["answer=artifacts/answer"],
        output_path=tmp_path / "preflight.json",
    )

    assert report["status"] == "pass"
    assert report["ready_for_vllm"] is True
    assert report["eval_inputs"][0]["rows"] == 2
    assert len(report["eval_inputs"][0]["sha256"]) == 64
    assert report["candidates"][0]["weights"][0]["size_bytes"] == len(b"weights")


def test_cloud_eval_preflight_fails_missing_adapter_weights(tmp_path: Path) -> None:
    _write_required_repo_files(tmp_path)
    _write_jsonl(tmp_path / "data/processed/local_eval_manifests/smoke_6pf.jsonl", [{"id": "a", "target_answer": "1"}])
    adapter = tmp_path / "artifacts/answer"
    adapter.mkdir(parents=True)
    adapter.joinpath("adapter_config.json").write_text("{}", encoding="utf-8")

    report = run_checks(
        repo_root=tmp_path,
        eval_inputs=["smoke_6pf"],
        candidates=["answer=artifacts/answer"],
        output_path=tmp_path / "preflight.json",
    )

    assert report["status"] == "fail"
    assert report["ready_for_vllm"] is False
    assert {
        "kind": "candidate",
        "name": "answer",
        "path": "artifacts/answer",
        "reason": "adapter weights missing",
    } in report["failures"]


def test_cloud_eval_preflight_fails_duplicate_eval_ids(tmp_path: Path) -> None:
    _write_required_repo_files(tmp_path)
    _write_jsonl(
        tmp_path / "data/processed/local_eval_manifests/smoke_6pf.jsonl",
        [
            {"id": "a", "target_answer": "1"},
            {"id": "a", "target_answer": "2"},
        ],
    )
    _write_adapter(tmp_path / "artifacts/answer")

    report = run_checks(
        repo_root=tmp_path,
        eval_inputs=["smoke_6pf"],
        candidates=["answer=artifacts/answer"],
        output_path=tmp_path / "preflight.json",
    )

    assert report["status"] == "fail"
    assert any("duplicate ids" in failure["reason"] for failure in report["failures"])


def test_cloud_eval_preflight_dry_run_does_not_require_files(tmp_path: Path) -> None:
    report = run_checks(
        repo_root=tmp_path,
        eval_inputs=["smoke_6pf"],
        candidates=["answer=artifacts/missing"],
        output_path=tmp_path / "preflight.json",
        dry_run=True,
    )

    assert report["status"] == "dry_run"
    assert report["ready_for_vllm"] is False
    assert report["planned_eval_inputs"] == ["smoke_6pf"]
    assert report["planned_candidates"] == ["answer=artifacts/missing"]
