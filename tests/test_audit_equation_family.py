from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from scripts.audit_equation_family import extract_shown_examples


REPO = Path(__file__).resolve().parents[1]


def _make_fixture_csv(tmpdir: Path) -> Path:
    rows = [
        {
            "id": "a",
            "family": "equation",
            "prompt": '!*[{ = \'"[\n}]`?( = ())\nNow, determine the result for: [[-!\'',
            "answer": '"[!',
        },
        {
            "id": "b",
            "family": "equation",
            "prompt": "34/44 = 1\n41/32 = 9\nNow, determine the result for: 69/52",
            "answer": "1",
        },
        {
            "id": "c",
            "family": "equation",
            "prompt": "abc = cba\nxyz = zyx\nNow, determine the result for: qwe",
            "answer": "ewq",
        },
        {
            "id": "d",
            "family": "bit",
            "prompt": "...",
            "answer": "01010101",
        },
    ]
    path = tmpdir / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_audit_runs_and_classifies(tmp_path: Path) -> None:
    csv = _make_fixture_csv(tmp_path)
    out = tmp_path / "audit.json"
    samp = tmp_path / "samp.md"
    subprocess.check_call(
        [
            "python",
            str(REPO / "scripts/audit_equation_family.py"),
            "--input",
            str(csv),
            "--output",
            str(out),
            "--sample-output",
            str(samp),
            "--samples-per-cell",
            "2",
        ]
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["total_examples"] == 3
    assert data["joint_counts"]["symbolic_char"]["answer_short_symbol_string"] >= 1
    assert data["joint_counts"]["operator_numeric_mixed"]["answer_integer"] >= 1
    assert data["joint_counts"]["position_transduction"]["answer_short_symbol_string"] >= 1
    assert data["recommended_branch"] in {
        "equation-symbolic-char-substitution",
        "equation-symbolic-sequence-transducer",
        "equation-symbolic-to-int",
        "equation-operator-induction",
        "equation-numeric-precision",
        "equation-position-transducer",
        "investigate-manual",
    }
    assert samp.exists() and samp.stat().st_size > 0


def test_inline_pair_at_prompt_end_is_counted() -> None:
    pairs = extract_shown_examples("A = 1, B = 2")
    assert ("A", "1") in pairs
    assert ("B", "2") in pairs
