from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SHELL_SCRIPTS = (
    "scripts/build_stage2_answer_focused_data.sh",
    "scripts/check_prompt_suffix_alignment.sh",
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


@pytest.mark.skipif(shutil.which("sh") is None, reason="sh is not available")
def test_prompt_suffix_check_python_fallback_accepts_chat_template_prompt(tmp_path: Path) -> None:
    prompt = (
        "<|im_start|>user\nquestion\n"
        "Please put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    rows = [{"id": "ok", "prompt": prompt, "completion": "\\boxed{42}"}]
    data = tmp_path / "chat.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    env = {"PATH": str(Path(sys.executable).parent)}

    subprocess.run(
        [shutil.which("sh") or "sh", str(REPO_ROOT / "scripts/check_prompt_suffix_alignment.sh"), str(data)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
