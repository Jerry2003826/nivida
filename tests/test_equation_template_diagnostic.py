from __future__ import annotations

from scripts.diagnose_equation_template import (
    classify_template_risk,
    target_char_provenance,
)


def test_target_char_provenance_counts_sources() -> None:
    row_text = {
        "support_inputs": ["ab*c", "de*f"],
        "support_outputs": ["aX", "dY"],
        "query": "gh*i",
        "target": "gYZ",
    }

    provenance = target_char_provenance(row_text)

    assert provenance == {
        "from_query": 1,
        "from_support_inputs": 0,
        "from_support_outputs": 1,
        "unseen": 1,
    }


def test_classify_template_risk_top1_hit() -> None:
    risk = classify_template_risk(
        oracle_rank=1,
        ambiguity_count=3,
        has_unseen_literal=False,
        support_full=True,
    )

    assert risk == "low_risk_support_stable"


def test_classify_template_risk_oracle_at_k_hit() -> None:
    risk = classify_template_risk(
        oracle_rank=2,
        ambiguity_count=5,
        has_unseen_literal=False,
        support_full=True,
    )

    assert risk == "ranker_miss_oracle_hit"


def test_classify_template_risk_unseen_literal() -> None:
    risk = classify_template_risk(
        oracle_rank=None,
        ambiguity_count=4,
        has_unseen_literal=True,
        support_full=True,
    )

    assert risk == "unseen_literal_high_risk"
