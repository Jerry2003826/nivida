from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_BRANCH_TOKENS = (
    "stage2_use_search_subtype_hint",
    "stage2_subtype_rescue",
    "adapter_stage2_subtype_rescue",
)


def _canonical_script_paths() -> list[Path]:
    return sorted(
        path
        for path in REPO_ROOT.glob("scripts/train_stage[123]_*.sh")
        if "_subtype_rescue.sh" not in path.name and "_smoke" not in path.name
    )


def _canonical_config_paths() -> list[Path]:
    return sorted(
        path
        for path in REPO_ROOT.glob("configs/train_stage[123]_*.yaml")
        if "_subtype_rescue" not in path.name and "smoke" not in path.parts
    )


def _branch_paths() -> list[Path]:
    return sorted(
        list(REPO_ROOT.glob("scripts/*_subtype_rescue*.sh"))
        + list(REPO_ROOT.glob("configs/*_subtype_rescue*.yaml"))
    )


def _yaml_training_triples(path: Path) -> set[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    training = payload.get("training", {}) or {}
    return {
        str(training[key])
        for key in ("output_dir", "dataset_path", "eval_path")
        if training.get(key)
    }


def test_canonical_files_do_not_reference_branch_only_tokens() -> None:
    canonical_paths = _canonical_script_paths() + _canonical_config_paths()
    assert canonical_paths, "expected canonical train scripts/configs to exist"

    for path in canonical_paths:
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_BRANCH_TOKENS:
            assert token not in text, f"{path} unexpectedly references {token!r}"


def test_branch_and_canonical_output_paths_do_not_overlap() -> None:
    canonical_paths = _canonical_config_paths()
    branch_paths = _branch_paths()
    assert branch_paths, "expected subtype-rescue branch files to exist"

    canonical_outputs: set[str] = set()
    branch_outputs: set[str] = set()

    for path in canonical_paths:
        canonical_outputs |= _yaml_training_triples(path)

    for path in branch_paths:
        if path.suffix == ".yaml":
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
