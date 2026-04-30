from __future__ import annotations

import json
from pathlib import Path

from scripts.run_solver_breakout_v2 import main, operator_gap_clusters, summarize_rows


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
    assert summary["operator_gap_clusters"]["top_operator_family"] == {"unknown": 1}


def test_solver_breakout_clusters_equation_operator_gaps() -> None:
    clusters = operator_gap_clusters(
        "equation_template",
        [
            {
                "risk_class": "operator_gap_oracle_miss",
                "ranker_support_key_coverage": 1.0,
                "provenance_unseen": 0,
                "target_expressible": False,
                "ranker_query_key_seen_any": True,
                "ranker_literal_reuse_risk": False,
            },
            {
                "risk_class": "operator_gap_oracle_miss",
                "ranker_support_key_coverage": 0.25,
                "provenance_unseen": 2,
                "target_expressible": True,
                "ranker_query_key_seen_any": False,
                "ranker_literal_reuse_risk": True,
            },
        ],
    )

    assert clusters["n"] == 2
    assert clusters["support_key_coverage"] == {"0.01-0.50": 1, "1.00": 1}
    assert clusters["target_literal_provenance"] == {
        "target_literals_seen": 1,
        "unseen_target_literal": 1,
    }
    assert clusters["query_key_seen"] == {"seen_query_key": 1, "unseen_query_key": 1}


def test_solver_breakout_clusters_bit_operator_gaps() -> None:
    clusters = operator_gap_clusters(
        "bit_permutation",
        [
            {
                "risk_class": "operator_gap_oracle_miss",
                "top_operator_family": "boolean_template",
                "oracle_operator_family": "unknown",
                "oracle_rank_bucket": "miss",
                "support_leave_one_out_stability": "ambiguous_support_fit",
                "top_hamming_to_target": 1,
                "top_oracle_hamming": "",
                "top_complexity_penalty": 0.4,
                "top_expression_complexity_total": 9,
            },
            {
                "risk_class": "operator_gap_oracle_miss",
                "top_operator_family": "affine_gf2",
                "oracle_operator_family": "unknown",
                "oracle_rank_bucket": "miss",
                "support_leave_one_out_stability": "unique_support_fit",
                "top_hamming_to_target": 4,
                "top_oracle_hamming": "",
                "top_complexity_penalty": 0.0,
                "top_expression_complexity_total": 0,
            },
        ],
    )

    assert clusters["n"] == 2
    assert clusters["top_operator_family"] == {"affine_gf2": 1, "boolean_template": 1}
    assert clusters["top_hamming_to_target"] == {"1": 1, "4+": 1}
    assert clusters["top_expression_complexity"] == {"0": 1, "9+": 1}


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
