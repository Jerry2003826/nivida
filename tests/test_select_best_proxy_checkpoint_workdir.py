from __future__ import annotations

from pathlib import Path

import scripts.select_best_proxy_checkpoint as sbpc


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_workdir_derives_from_stage_output_dir() -> None:
    stage_output_dir = Path("artifacts/adapter_stage2_selected_trace")
    workdir = sbpc.derive_default_workdir(stage_output_dir)
    assert workdir.as_posix().startswith("artifacts/_proxy_checkpoint_scratch/")
    assert "adapter_stage2_selected_trace" in workdir.as_posix()


def test_default_workdir_differs_for_distinct_stage_outputs() -> None:
    canonical = sbpc.derive_default_workdir(Path("artifacts/adapter_stage2_selected_trace"))
    branch = sbpc.derive_default_workdir(Path("artifacts/adapter_stage2_subtype_rescue"))
    assert canonical != branch


def test_canonical_and_branch_selector_calls_all_pass_explicit_workdir() -> None:
    required_callers = (
        "scripts/train_stage2_distill.sh",
        "scripts/train_stage2_subtype_rescue.sh",
        "scripts/train_stage3_repair.sh",
    )
    for rel in required_callers:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "select_best_proxy_checkpoint.py" in text
        assert "--workdir" in text, f"{rel}: selector call missing --workdir"
