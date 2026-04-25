from __future__ import annotations

from scripts.diagnose_bit_permutation import (
    classify_bit_candidate_gap,
    hamming_distance,
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
