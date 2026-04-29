from __future__ import annotations

from scripts.diagnose_bit_permutation import (
    bit_operator_family_from_steps,
    boolean_expression_features,
    classify_bit_candidate_gap,
    filter_diagnostics,
    hamming_distance,
    oracle_rank_bucket,
    support_leave_one_out_stability,
)


def test_hamming_distance_counts_differing_bits() -> None:
    assert hamming_distance("1010", "1001") == 2


def test_classify_bit_candidate_gap_top1_hit() -> None:
    assert (
        classify_bit_candidate_gap(
            top_correct=True,
            oracle_rank=1,
            top_support_full=True,
        )
        == "low_risk_top1"
    )


def test_classify_bit_candidate_gap_ranker_miss() -> None:
    assert (
        classify_bit_candidate_gap(
            top_correct=False,
            oracle_rank=3,
            top_support_full=True,
        )
        == "ranker_miss_oracle_hit"
    )


def test_classify_bit_candidate_gap_operator_gap() -> None:
    assert (
        classify_bit_candidate_gap(
            top_correct=False,
            oracle_rank=None,
            top_support_full=True,
        )
        == "operator_gap_oracle_miss"
    )


def test_boolean_expression_features_summarizes_ops_and_complexity() -> None:
    features = boolean_expression_features(
        [
            {"op": "copy", "complexity": 1},
            {"op": "xor", "complexity": 3},
            {"op": "choice", "complexity": 5},
        ]
    )

    assert features["expression_ops"] == "copy,xor,choice"
    assert features["expression_complexity_total"] == 9
    assert features["expression_complexity_avg"] == 3.0
    assert features["expression_complexity_max"] == 5


def test_bit_operator_family_classifies_low_risk_families() -> None:
    assert bit_operator_family_from_steps(steps="binary_affine_transform") == "affine_gf2"
    assert bit_operator_family_from_steps(steps="reverse_bits") == "reversal"
    assert bit_operator_family_from_steps(steps="swap_nibbles") == "nibble_byte_transform"
    assert bit_operator_family_from_steps(steps="binary_permutation") == "plain_permutation"
    assert (
        bit_operator_family_from_steps(
            steps="binary_boolean_expr",
            expression_ops="copy,xor",
        )
        == "boolean_template"
    )


def test_oracle_rank_bucket_and_support_stability() -> None:
    assert oracle_rank_bucket(None) == "miss"
    assert oracle_rank_bucket(1) == "top1"
    assert oracle_rank_bucket(3) == "rank2_3"
    assert oracle_rank_bucket(4) == "rank4_plus"
    assert (
        support_leave_one_out_stability(
            top_support_full=True,
            support_full_candidate_count=1,
            top_complexity_penalty=1.0,
            top_expression_complexity_total=99,
        )
        == "unique_support_fit"
    )
    assert (
        support_leave_one_out_stability(
            top_support_full=False,
            support_full_candidate_count=0,
            top_complexity_penalty=0.0,
            top_expression_complexity_total=0,
        )
        == "unstable_support"
    )


def test_filter_diagnostics_filters_risk_subtype_and_limit() -> None:
    rows = [
        {"id": "a", "risk_class": "operator_gap_oracle_miss", "subtype": "bit_permutation"},
        {"id": "b", "risk_class": "ranker_miss_oracle_hit", "subtype": "bit_permutation"},
        {"id": "c", "risk_class": "ranker_miss_oracle_hit", "subtype": "bit_affine"},
    ]

    filtered = filter_diagnostics(
        rows,
        failure_class="ranker_miss_oracle_hit",
        subtype="bit_permutation",
        limit=1,
    )

    assert filtered == [rows[1]]
