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


def test_readme_stage2_example_prefers_canonical_script_and_exported_subset() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "bash scripts/train_stage2_distill.sh" in text
    assert "Manual builder commands are intentionally omitted here." in text
    assert "--input data/processed/official_train_tagged.jsonl,data/synthetic/synth_hard_triads.jsonl" not in text


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


def test_stage3_subtype_runtime_config_uses_exported_paths() -> None:
    text = (REPO_ROOT / "scripts" / "train_stage3_subtype_rescue.sh").read_text(
        encoding="utf-8"
    )
    assert '"$FINAL_STAGE3_ADAPTER_DIR"' in text
    assert '"$REPAIR_STAGE3_TRAIN_DATASET"' in text
    assert '"$REPAIR_STAGE3_VALID_DATASET"' in text
    assert 'training["output_dir"] = "artifacts/adapter_stage3_subtype_rescue"' not in text
    assert 'training["dataset_path"] = "data/processed/stage3_subtype_rescue_train.jsonl"' not in text
    assert 'training["eval_path"] = "data/processed/stage3_subtype_rescue_valid.jsonl"' not in text


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
    assert 'ALLOW_SUBTYPE_RESCUE_REGENERATE_INPUTS="${ALLOW_SUBTYPE_RESCUE_REGENERATE_INPUTS:-0}"' in text
    assert 'FORCE_SUBTYPE_RESCUE_REGENERATE_INPUTS="${FORCE_SUBTYPE_RESCUE_REGENERATE_INPUTS:-0}"' in text
    assert "mktemp" in text
    assert "REFRESH_SUBTYPE_RESCUE_INPUTS" in text


def test_subtype_rescue_defaults_do_not_write_canonical_stage2_inputs() -> None:
    text = (REPO_ROOT / "scripts" / "train_stage2_subtype_rescue.sh").read_text(
        encoding="utf-8"
    )
    assert 'STAGE2_TRAIN_OFFICIAL_SUBSET="${STAGE2_TRAIN_OFFICIAL_SUBSET:-data/processed/stage2_subtype_rescue_official_train_no_hard_valid.jsonl}"' in text
    assert 'STAGE2_VALID_OFFICIAL_SUBSET="${STAGE2_VALID_OFFICIAL_SUBSET:-data/processed/stage2_subtype_rescue_official_valid_hard_triad.jsonl}"' in text
    assert 'ALL_FAMILY_PROXY_VALID_SUBSET="${ALL_FAMILY_PROXY_VALID_SUBSET:-data/processed/stage2_subtype_rescue_proxy_all_family_valid.jsonl}"' in text
    assert 'SYNTH_HARD_TRIADS_PATH="${SYNTH_HARD_TRIADS_PATH:-data/synthetic/synth_hard_triads_subtype_rescue.jsonl}"' in text
    assert 'SYNTH_HARD_TRIADS_SUMMARY_PATH="${SYNTH_HARD_TRIADS_SUMMARY_PATH:-data/synthetic/synth_hard_triads_subtype_rescue_summary.json}"' in text
    assert 'CANONICAL_STAGE2_TRAIN_OFFICIAL_SUBSET="${CANONICAL_STAGE2_TRAIN_OFFICIAL_SUBSET:-data/processed/stage2_official_train_no_hard_valid.jsonl}"' in text
    assert 'CANONICAL_SYNTH_HARD_TRIADS_PATH="${CANONICAL_SYNTH_HARD_TRIADS_PATH:-data/synthetic/synth_hard_triads.jsonl}"' in text
    assert 'copy_into_branch_path "$CANONICAL_STAGE2_TRAIN_OFFICIAL_SUBSET" "$STAGE2_TRAIN_OFFICIAL_SUBSET"' in text
    assert 'copy_into_branch_path "$CANONICAL_STAGE2_VALID_OFFICIAL_SUBSET" "$STAGE2_VALID_OFFICIAL_SUBSET"' in text
    assert 'copy_into_branch_path "$CANONICAL_ALL_FAMILY_PROXY_VALID_SUBSET" "$ALL_FAMILY_PROXY_VALID_SUBSET"' in text
    assert 'copy_into_branch_path "$CANONICAL_SYNTH_HARD_TRIADS_PATH" "$SYNTH_HARD_TRIADS_PATH"' in text


def test_subtype_rescue_refresh_prefers_canonical_copy_unless_force_regenerate() -> None:
    text = (REPO_ROOT / "scripts" / "train_stage2_subtype_rescue.sh").read_text(
        encoding="utf-8"
    )
    assert 'if [[ "$REFRESH_SUBTYPE_RESCUE_INPUTS" != "1" && "$FORCE_SUBTYPE_RESCUE_REGENERATE_INPUTS" != "1" && -f "$SYNTH_HARD_TRIADS_PATH" && -f "$SYNTH_HARD_TRIADS_SUMMARY_PATH" ]]; then' in text
    assert 'elif [[ "$REFRESH_SUBTYPE_RESCUE_INPUTS" != "1" && "$FORCE_SUBTYPE_RESCUE_REGENERATE_INPUTS" != "1" && ( -f "$SYNTH_HARD_TRIADS_PATH" || -f "$SYNTH_HARD_TRIADS_SUMMARY_PATH" ) ]]; then' in text
    assert "Partial branch-local synth artifacts found; set REFRESH_SUBTYPE_RESCUE_INPUTS=1 to recopy canonical inputs." in text
    assert 'elif [[ -f "$CANONICAL_SYNTH_HARD_TRIADS_PATH" && -f "$CANONICAL_SYNTH_HARD_TRIADS_SUMMARY_PATH" && "$FORCE_SUBTYPE_RESCUE_REGENERATE_INPUTS" != "1" ]]; then' in text
    assert 'if [[ "$REFRESH_SUBTYPE_RESCUE_INPUTS" != "1" && "$FORCE_SUBTYPE_RESCUE_REGENERATE_INPUTS" != "1" && -f "$STAGE2_TRAIN_OFFICIAL_SUBSET" ]]; then' in text
    assert 'if [[ "$REFRESH_SUBTYPE_RESCUE_INPUTS" != "1" && "$FORCE_SUBTYPE_RESCUE_REGENERATE_INPUTS" != "1" && -f "$STAGE2_VALID_OFFICIAL_SUBSET" ]]; then' in text
    assert 'if [[ "$REFRESH_SUBTYPE_RESCUE_INPUTS" != "1" && "$FORCE_SUBTYPE_RESCUE_REGENERATE_INPUTS" != "1" && -f "$ALL_FAMILY_PROXY_VALID_SUBSET" ]]; then' in text
    assert 'elif [[ -f "$CANONICAL_STAGE2_TRAIN_OFFICIAL_SUBSET" && "$FORCE_SUBTYPE_RESCUE_REGENERATE_INPUTS" != "1" ]]; then' in text
    assert 'elif [[ -f "$CANONICAL_STAGE2_VALID_OFFICIAL_SUBSET" && "$FORCE_SUBTYPE_RESCUE_REGENERATE_INPUTS" != "1" ]]; then' in text
    assert 'elif [[ -f "$CANONICAL_ALL_FAMILY_PROXY_VALID_SUBSET" && "$FORCE_SUBTYPE_RESCUE_REGENERATE_INPUTS" != "1" ]]; then' in text
    assert "regenerated_branch_local_forced" in text


def test_subtype_rescue_script_writes_input_manifest_and_skip_artifact() -> None:
    text = (REPO_ROOT / "scripts" / "train_stage2_subtype_rescue.sh").read_text(
        encoding="utf-8"
    )
    assert 'STAGE2_INPUT_MANIFEST="${STAGE2_INPUT_MANIFEST:-data/processed/stage2_subtype_rescue_input_manifest.json}"' in text
    assert 'STAGE2_SKIPPED_ARTIFACT="${STAGE2_SKIPPED_ARTIFACT:-data/processed/stage2_subtype_rescue_skipped.json}"' in text
    assert 'CLEAN_SUBTYPE_RESCUE_OUTPUTS="${CLEAN_SUBTYPE_RESCUE_OUTPUTS:-1}"' in text
    assert "write_input_manifest" in text
    assert "clean_stale_branch_outputs" in text
    assert '"source_type": source_type' in text
    assert '"sha256": sha256(path)' in text
    assert '"canonical_equivalent_path"' in text
    assert '"canonical_sha256"' in text
    assert '"matches_canonical"' in text
    assert '"git_head": git_head' in text
    assert '"dependencies": dependencies' in text
    assert 'rm -f "$STAGE2_SKIPPED_ARTIFACT"' in text


def test_subtype_rescue_script_requires_canonical_inputs_or_explicit_override() -> None:
    text = (REPO_ROOT / "scripts" / "train_stage2_subtype_rescue.sh").read_text(
        encoding="utf-8"
    )
    assert 'require_canonical_input_or_override "stage2 train subset" "$CANONICAL_STAGE2_TRAIN_OFFICIAL_SUBSET"' in text
    assert 'require_canonical_input_or_override "hard-triad synth input" "$CANONICAL_SYNTH_HARD_TRIADS_PATH"' in text
    assert "ALLOW_SUBTYPE_RESCUE_REGENERATE_INPUTS=1" in text
    assert "FORCE_SUBTYPE_RESCUE_REGENERATE_INPUTS=1" in text
    assert "standalone forced regeneration" in text


def test_subtype_rescue_script_cleans_stale_branch_outputs_by_default() -> None:
    text = (REPO_ROOT / "scripts" / "train_stage2_subtype_rescue.sh").read_text(
        encoding="utf-8"
    )
    assert 'CLEAN_SUBTYPE_RESCUE_OUTPUTS="${CLEAN_SUBTYPE_RESCUE_OUTPUTS:-1}"' in text
    assert 'CLEAN_SUBTYPE_RESCUE_ADAPTERS="${CLEAN_SUBTYPE_RESCUE_ADAPTERS:-1}"' in text
    assert 'CLEAN_SUBTYPE_RESCUE_SCRATCH="${CLEAN_SUBTYPE_RESCUE_SCRATCH:-1}"' in text
    assert 'STAGE2_PROMOTION_JSON="${STAGE2_PROMOTION_JSON:-data/processed/stage2_subtype_rescue_promotion.json}"' in text
    assert 'if [[ "$CLEAN_SUBTYPE_RESCUE_OUTPUTS" != "1" ]]; then' in text
    assert '"$STAGE2_PROMOTION_JSON"' in text
    assert '"$STAGE2_BESTPROXY_HARD_EVAL"' in text
    assert '"$STAGE2_BESTPROXY_ALL_EVAL"' in text
    assert '"$STAGE2_BESTPROXY_SELECTION_JSON"' in text
    assert "clean_stale_branch_outputs" in text
    assert "assert_subtype_path()" in text
    assert 'assert_subtype_path "$STAGE2_ADAPTER_DIR" "STAGE2_ADAPTER_DIR"' in text
    assert 'assert_subtype_path "$STAGE2_BESTPROXY_DIR" "STAGE2_BESTPROXY_DIR"' in text
    assert 'assert_subtype_path "$STAGE2_BESTPROXY_WORKDIR" "STAGE2_BESTPROXY_WORKDIR"' in text
    assert 'rm -rf "$STAGE2_ADAPTER_DIR" "$STAGE2_BESTPROXY_DIR"' in text
    assert 'rm -rf "$STAGE2_BESTPROXY_WORKDIR"' in text


def test_subtype_rescue_prepare_data_only_appears_in_regeneration_helper() -> None:
    text = (REPO_ROOT / "scripts" / "train_stage2_subtype_rescue.sh").read_text(
        encoding="utf-8"
    )
    assert "ensure_official_inputs_for_regeneration()" in text
    assert text.count("ensure_official_inputs_for_regeneration") >= 4
    assert 'if [[ ! -f "data/processed/official_train_tagged.jsonl" ]]; then\n  python scripts/prepare_data.py --config configs/data_official.yaml\nfi' not in text


def test_subtype_rescue_script_checks_branch_local_hash_against_canonical() -> None:
    text = (REPO_ROOT / "scripts" / "train_stage2_subtype_rescue.sh").read_text(
        encoding="utf-8"
    )
    assert "assert_branch_local_matches_canonical" in text
    assert '"branch-local train subset"' in text
    assert '"branch-local valid subset"' in text
    assert '"branch-local all-family proxy subset"' in text
    assert '"branch-local synth input"' in text


def test_stage2_subtype_runtime_config_uses_exported_paths() -> None:
    text = (REPO_ROOT / "scripts" / "train_stage2_subtype_rescue.sh").read_text(
        encoding="utf-8"
    )
    assert '"$STAGE2_ADAPTER_DIR"' in text
    assert '"$STAGE2_TRAIN_DATASET"' in text
    assert '"$STAGE2_VALID_DATASET"' in text
    assert 'training["output_dir"] = adapter_dir' in text
    assert 'training["dataset_path"] = train_dataset' in text
    assert 'training["eval_path"] = valid_dataset' in text


def test_stage3_subtype_manual_override_requires_second_confirmation() -> None:
    text = (REPO_ROOT / "scripts" / "train_stage3_subtype_rescue.sh").read_text(
        encoding="utf-8"
    )
    assert 'I_UNDERSTAND_SUBTYPE_STAGE3_WAS_NOT_PROMOTED="${I_UNDERSTAND_SUBTYPE_STAGE3_WAS_NOT_PROMOTED:-0}"' in text
    assert "Manual subtype stage3 override requires I_UNDERSTAND_SUBTYPE_STAGE3_WAS_NOT_PROMOTED=1" in text
