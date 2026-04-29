from __future__ import annotations

import json
from pathlib import Path

from scripts.run_solver_breakout_v2 import main, summarize_rows


def test_solver_breakout_summary_counts_upper_bound_and_gaps() -> None:
    rows = [
        {
            "id": "hit",
            "path": "manifest.jsonl",
            "subtype": "bit_permutation",
            "top_query_correct": True,
            "oracle_rank": 1,
            "support_full_candidate_count": 1,
            "risk_class": "low_risk_top1",
            "top_support_full": True,
        },
        {
            "id": "miss",
            "path": "manifest.jsonl",
            "subtype": "bit_permutation",
            "top_query_correct": False,
            "oracle_rank": 3,
            "support_full_candidate_count": 4,
            "risk_class": "ranker_miss_oracle_hit",
            "top_support_full": True,
        },
        {
            "id": "gap",
            "path": "manifest.jsonl",
            "subtype": "bit_permutation",
            "top_query_correct": False,
            "oracle_rank": None,
            "support_full_candidate_count": 0,
            "risk_class": "operator_gap_oracle_miss",
            "top_support_full": True,
        },
    ]

    summary = summarize_rows("bit_permutation", rows)

    assert summary["overall"]["n"] == 3
    assert summary["overall"]["top1_correct_count"] == 1
    assert summary["overall"]["oracle_at_k_count"] == 2
    assert summary["overall"]["safe_override_possible_count"] == 1
    assert summary["overall"]["ranker_miss_oracle_hit_count"] == 1
    assert summary["overall"]["operator_gap_oracle_miss_count"] == 1
    assert summary["overall"]["theoretical_gain_ceiling_count"] == 1
    assert len(summary["ranker_miss_oracle_hit_examples"]) == 1


def test_solver_breakout_cli_writes_schema_smoke_outputs(tmp_path: Path) -> None:
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text("", encoding="utf-8")
    output_dir = tmp_path / "solver_breakout_v2"
    output_md = tmp_path / "solver_breakout_v2.md"

    assert (
        main(
            [
                "--input",
                str(manifest),
                "--limit",
                "0",
                "--output-dir",
                str(output_dir),
                "--output-md",
                str(output_md),
            ]
        )
        == 0
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == 1
    assert summary["equation_template"]["overall"]["n"] == 0
    assert summary["bit_permutation"]["overall"]["n"] == 0
    assert output_md.read_text(encoding="utf-8").startswith("# Solver Breakout v2")
