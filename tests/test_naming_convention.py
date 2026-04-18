from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "stage1_acceptance.md",
    REPO_ROOT / "docs" / "submission_runbook.md",
    REPO_ROOT / "docs" / "branch_compare_runbook.md",
    REPO_ROOT / "scripts" / "train_stage1_format_align.sh",
    REPO_ROOT / "scripts" / "train_stage2_distill.sh",
    REPO_ROOT / "scripts" / "train_stage3_repair.sh",
    REPO_ROOT / "scripts" / "train_stage3_subtype_rescue.sh",
]
BANNED_PATTERNS = (
    r"baseline_eval_latest",
    r"teacher_benchmark_latest",
    r"official_stage_c",
    r"stage_c_",
    r"train_lora_stage3_repair",
    r"train_lora_official_full_runpod",
    r"stage2_synth\.jsonl",
    r"synth_stage_b",
    r"(^|[\\/])selected_train\.jsonl([\"'\s]|$)",
    r"(^|[\\/])repair_train\.jsonl([\"'\s]|$)",
)


def test_docs_and_mainline_scripts_do_not_revive_legacy_names() -> None:
    matches: list[str] = []
    for path in SCAN_FILES:
        text = path.read_text(encoding="utf-8")
        for pattern in BANNED_PATTERNS:
            if re.search(pattern, text):
                matches.append(f"{path}: {pattern}")
    assert not matches, "legacy names revived in user-facing docs or scripts:\n" + "\n".join(matches)
