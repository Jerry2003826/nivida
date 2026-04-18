from __future__ import annotations

import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_BRANCH_TOKENS = (
    "stage2_use_search_subtype_hint",
    "stage2_subtype_rescue",
    "adapter_stage2_subtype_rescue",
    "adapter_stage3_subtype_rescue",
    "stage3_subtype_rescue",
)
CANONICAL_FILES = (
    "README.md",
    "docs/stage1_acceptance.md",
    "docs/submission_runbook.md",
    "scripts/train_stage1_format_align.sh",
    "scripts/train_stage2_distill.sh",
    "scripts/train_stage3_repair.sh",
    "scripts/select_final_adapter.py",
    "scripts/select_best_proxy_checkpoint.py",
    "scripts/validate_submission.py",
    "configs/train_stage1_format.yaml",
    "configs/train_stage2_selected_trace.yaml",
    "configs/train_stage3_repair.yaml",
)


def _canonical_paths() -> list[Path]:
    paths = [REPO_ROOT / item for item in CANONICAL_FILES]
    missing = [str(path) for path in paths if not path.exists()]
    assert not missing, f"canonical manifest paths missing: {missing}"
    return paths


def _branch_config_paths() -> list[Path]:
    return sorted(REPO_ROOT.glob("configs/*_subtype_rescue*.yaml"))


def _yaml_training_triples(path: Path) -> set[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    training = payload.get("training", {}) or {}
    return {
        str(training[key])
        for key in ("output_dir", "dataset_path", "eval_path")
        if training.get(key)
    }


def test_canonical_files_do_not_reference_branch_only_tokens() -> None:
    canonical_paths = _canonical_paths()
    for path in canonical_paths:
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_BRANCH_TOKENS:
            assert token not in text, f"{path} unexpectedly references {token!r}"


def test_readme_never_points_canonical_submission_to_stage3_repair_adapter() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "artifacts/adapter_final_selected" in text
    assert not re.search(
        r"validate_submission\.py[\s\\]+.*?--adapter-dir\s+artifacts/adapter_stage3_repair\b",
        text,
        flags=re.DOTALL,
    )


def test_branch_and_canonical_output_paths_do_not_overlap() -> None:
    canonical_outputs: set[str] = set()
    branch_outputs: set[str] = set()

    for path in _canonical_paths():
        if path.suffix == ".yaml":
            canonical_outputs |= _yaml_training_triples(path)

    for path in _branch_config_paths():
        branch_outputs |= _yaml_training_triples(path)

    overlap = canonical_outputs & branch_outputs
    assert not overlap, f"canonical and branch outputs must stay disjoint: {sorted(overlap)}"


def test_stage3_subtype_rescue_script_does_not_reference_canonical_stage3_outputs() -> None:
    text = (REPO_ROOT / "scripts" / "train_stage3_subtype_rescue.sh").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "artifacts/adapter_stage3_repair",
        "artifacts/adapter_stage3_bestproxy",
        "data/processed/stage3_repair_train.jsonl",
        "data/processed/stage3_repair_valid.jsonl",
        "data/processed/stage3_bestproxy_hard_eval.json",
        "data/processed/stage3_bestproxy_all_eval.json",
    )
    for token in forbidden:
        assert token not in text, f"stage3 subtype-rescue scaffold leaked canonical path {token!r}"


def test_stage3_branch_scaffold_exports_dataset_overrides() -> None:
    canonical_text = (REPO_ROOT / "scripts" / "train_stage3_repair.sh").read_text(
        encoding="utf-8"
    )
    branch_text = (REPO_ROOT / "scripts" / "train_stage3_subtype_rescue.sh").read_text(
        encoding="utf-8"
    )

    assert 'REPAIR_STAGE3_TRAIN_DATASET="${REPAIR_STAGE3_TRAIN_DATASET:-data/processed/stage3_repair_train.jsonl}"' in canonical_text
    assert 'REPAIR_STAGE3_VALID_DATASET="${REPAIR_STAGE3_VALID_DATASET:-data/processed/stage3_repair_valid.jsonl}"' in canonical_text
    assert '--output "$REPAIR_STAGE3_TRAIN_DATASET"' in canonical_text
    assert '--output "$REPAIR_STAGE3_VALID_DATASET"' in canonical_text

    assert 'export REPAIR_STAGE3_TRAIN_DATASET="${REPAIR_STAGE3_TRAIN_DATASET:-data/processed/stage3_subtype_rescue_train.jsonl}"' in branch_text
    assert 'export REPAIR_STAGE3_VALID_DATASET="${REPAIR_STAGE3_VALID_DATASET:-data/processed/stage3_subtype_rescue_valid.jsonl}"' in branch_text


def test_stage3_repair_script_uses_env_for_stage2_inference_config() -> None:
    text = (REPO_ROOT / "scripts" / "train_stage3_repair.sh").read_text(
        encoding="utf-8"
    )
    default_decl = (
        'STAGE2_INFERENCE_CONFIG="${STAGE2_INFERENCE_CONFIG:-'
        'configs/train_stage2_selected_trace.yaml}"'
    )
    assert default_decl in text
    assert text.count('--config "$STAGE2_INFERENCE_CONFIG"') == 2
    assert "--config configs/train_stage2_selected_trace.yaml" not in text


def test_subtype_rescue_script_declares_refresh_and_atomic_write_pattern() -> None:
    text = (REPO_ROOT / "scripts" / "train_stage2_subtype_rescue.sh").read_text(
        encoding="utf-8"
    )
    assert 'REFRESH_SUBTYPE_RESCUE_INPUTS="${REFRESH_SUBTYPE_RESCUE_INPUTS:-0}"' in text
    assert "mktemp" in text
    assert "REFRESH_SUBTYPE_RESCUE_INPUTS" in text
