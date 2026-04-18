from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SHELL_SCRIPTS = (
    "scripts/train_stage1_format_align.sh",
    "scripts/train_stage2_distill.sh",
    "scripts/train_stage3_repair.sh",
    "scripts/train_stage2_subtype_rescue.sh",
    "scripts/train_stage3_subtype_rescue.sh",
)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
def test_shell_scripts_parse_with_bash_n() -> None:
    for rel in SHELL_SCRIPTS:
        subprocess.run(
            ["bash", "-n", str(REPO_ROOT / rel)],
            check=True,
            cwd=REPO_ROOT,
        )
