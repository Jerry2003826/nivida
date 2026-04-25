"""Tests for the stage2 silver-pool + rejection-diagnostics additions."""
from __future__ import annotations

import pytest

from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair
from src.student import sft_dataset_builder as builder
from src.student.sft_dataset_builder import (
    _select_official_stage2_silver,
    _select_official_stage2_strict,
    build_selected_sft_with_report,
)


def _example(
    example_id: str,
    *,
    family: str = "cipher",
    source: str = "official",
    confidence: float = 0.9,
    support_coverage: float = 1.0,
    solver_verifiable: bool = True,
    signature: str | None = "sig-default",
    target: str = "seven eight",
    query: str = "alpha beta",
) -> PuzzleExample:
    extras: dict = {
        "solver_verifiable": solver_verifiable,
        "support_coverage": support_coverage,
        "top1_top2_margin": 0.1,
    }
    return PuzzleExample(
        id=example_id,
        raw_prompt="In Alice's Wonderland, ...",
        official_instruction="",
        parsed_examples=[
            PuzzlePair(input="foo", output="one"),
            PuzzlePair(input="bar", output="two"),
            PuzzlePair(input="baz", output="three"),
        ],
        query=query,
        target_answer=target,
        metadata=PuzzleMetadata(
            official_family=family,
            subtype=f"{family}_sub",
            teacher_confidence=confidence,
            program_signature=signature,
            source=source,
            split="train",
            extras=extras,
        ),
    )


# --- gate-level tests ---------------------------------------------------------


def test_strict_gate_returns_reason_for_each_rejection() -> None:
    missing_query = _example("a", query="")
    ok, reason = _select_official_stage2_strict(missing_query)
    assert ok is False and reason == "missing_target_or_query"

    low_conf = _example("b", confidence=0.5)
    ok, reason = _select_official_stage2_strict(low_conf)
    assert ok is False and reason == "low_teacher_confidence"

    unverified = _example("c", solver_verifiable=False)
    ok, reason = _select_official_stage2_strict(unverified)
    assert ok is False and reason == "not_solver_verifiable"

    partial = _example("d", support_coverage=0.5)
    ok, reason = _select_official_stage2_strict(partial)
    assert ok is False and reason == "partial_support_coverage"

    no_signature = _example("e", signature=None)
    ok, reason = _select_official_stage2_strict(no_signature)
    assert ok is False and reason == "missing_program_signature"

    query_mismatch = _example("mismatch")
    query_mismatch.metadata.extras["query_solver_correct"] = False
    ok, reason = _select_official_stage2_strict(query_mismatch)
    assert ok is False and reason == "query_prediction_mismatch"

    happy = _example("f")
    ok, reason = _select_official_stage2_strict(happy)
    assert ok is True and reason is None


def test_strict_gate_rejects_high_risk_template_trace() -> None:
    example = _example("template_risk", family="equation", target="X", query="a*b")
    example.metadata.subtype = "equation_template"
    example.metadata.extras["template_risk_class"] = "unseen_literal_high_risk"

    ok, reason = _select_official_stage2_strict(example)

    assert ok is False
    assert reason == "high_risk_template_trace"


def test_silver_gate_only_accepts_hard_triad() -> None:
    easy_silver_candidate = _example(
        "easy", family="numeral", confidence=0.70, support_coverage=0.80, solver_verifiable=False
    )
    assert _select_official_stage2_silver(easy_silver_candidate) is False


def test_silver_gate_accepts_hard_triad_with_signature_but_no_solver() -> None:
    hard = _example(
        "hard",
        family="equation",
        confidence=0.70,  # strict would fail (< 0.80)
        support_coverage=0.80,  # strict would fail (< 1.0)
        solver_verifiable=False,
        signature="sig-silver",
    )
    assert _select_official_stage2_silver(hard) is True


def test_silver_gate_accepts_hard_triad_with_solver_but_no_signature() -> None:
    hard = _example(
        "hard",
        family="bit",
        confidence=0.70,
        support_coverage=0.70,
        solver_verifiable=True,
        signature=None,
    )
    assert _select_official_stage2_silver(hard) is True


def test_silver_gate_rejects_below_thresholds() -> None:
    too_low_conf = _example(
        "x", family="cipher", confidence=0.60, support_coverage=0.80
    )
    assert _select_official_stage2_silver(too_low_conf) is False

    too_low_support = _example(
        "y", family="cipher", confidence=0.80, support_coverage=0.50
    )
    assert _select_official_stage2_silver(too_low_support) is False


# --- build_selected_sft_with_report integration ------------------------------


def _patch_annotate_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the heavy chain-search annotation; passes examples through unchanged."""

    def _identity(examples, **_kwargs):
        return list(examples)

    monkeypatch.setattr(builder, "_annotate_examples", _identity)


def test_silver_samples_get_answer_only_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_annotate_identity(monkeypatch)
    strict = _example("strict", family="cipher")
    silver = _example(
        "silver",
        family="equation",
        confidence=0.70,
        support_coverage=0.80,
        solver_verifiable=False,
        signature="sig-silver",
    )
    bundle = build_selected_sft_with_report(
        [strict, silver],
        prompt_mode=builder.PROMPT_MODE_GENERIC,
        trace_style="token_trace",
        balance_by_family=False,
        hard_triad_repeat_factor=1,
        max_per_signature_bucket=0,
        enable_silver_official=True,
        silver_max_fraction=1.0,
        silver_max_absolute=100,
    )
    records = {record["id"]: record for record in bundle["records"]}
    assert records["strict"]["trace_style"] == "token_trace"
    assert records["silver"]["trace_style"] == "answer_only"


def test_silver_only_admits_hard_triad_families(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_annotate_identity(monkeypatch)
    # An easy-triad sample that would pass silver thresholds on numbers is
    # still excluded because silver gate rejects non-hard families.
    easy_candidate = _example(
        "numeral_silver",
        family="numeral",
        confidence=0.70,
        support_coverage=0.80,
        solver_verifiable=False,
        signature="sig",
    )
    strict = _example("strict_cipher", family="cipher")
    bundle = build_selected_sft_with_report(
        [strict, easy_candidate],
        prompt_mode=builder.PROMPT_MODE_GENERIC,
        balance_by_family=False,
        hard_triad_repeat_factor=1,
        max_per_signature_bucket=0,
        enable_silver_official=True,
        silver_max_fraction=1.0,
        silver_max_absolute=100,
    )
    ids = {record["id"] for record in bundle["records"]}
    assert "numeral_silver" not in ids
    assert bundle["selection_counts"]["official_silver"] == 0


def test_silver_cap_absolute_caps_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_annotate_identity(monkeypatch)
    examples = [_example("strict", family="cipher")]
    examples.extend(
        _example(
            f"silver_{idx}",
            family="equation",
            confidence=0.70,
            support_coverage=0.80,
            solver_verifiable=False,
            signature=f"sig-{idx}",
            query=f"q-{idx}",
            target=f"t-{idx}",
        )
        for idx in range(10)
    )
    bundle = build_selected_sft_with_report(
        examples,
        prompt_mode=builder.PROMPT_MODE_GENERIC,
        balance_by_family=False,
        hard_triad_repeat_factor=1,
        max_per_signature_bucket=0,
        enable_silver_official=True,
        # 10 candidates, only 1 strict -> fraction 10.0 * 1 = 10 cap; the tighter
        # absolute cap 3 should bind.
        silver_max_fraction=10.0,
        silver_max_absolute=3,
    )
    assert bundle["selection_counts"]["official_silver"] == 3


def test_silver_cap_fraction_caps_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_annotate_identity(monkeypatch)
    # 4 strict + 0 synth -> fraction 0.25 * 4 = 1 silver max
    strict = [
        _example(f"strict_{idx}", family="cipher", signature=f"s-{idx}")
        for idx in range(4)
    ]
    silvers = [
        _example(
            f"silver_{idx}",
            family="equation",
            confidence=0.70,
            support_coverage=0.80,
            solver_verifiable=False,
            signature=f"sig-{idx}",
            query=f"q-{idx}",
            target=f"t-{idx}",
        )
        for idx in range(5)
    ]
    bundle = build_selected_sft_with_report(
        strict + silvers,
        prompt_mode=builder.PROMPT_MODE_GENERIC,
        balance_by_family=False,
        hard_triad_repeat_factor=1,
        max_per_signature_bucket=0,
        enable_silver_official=True,
        silver_max_fraction=0.25,
        silver_max_absolute=1000,  # fraction cap wins
    )
    assert bundle["selection_counts"]["official_silver"] == 1


def test_silver_priority_is_equation_then_cipher_then_bit(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_annotate_identity(monkeypatch)
    bit = _example(
        "bit_silver",
        family="bit",
        confidence=0.70,
        support_coverage=0.80,
        solver_verifiable=False,
        signature="sig-bit",
        query="q-bit",
    )
    cipher = _example(
        "cipher_silver",
        family="cipher",
        confidence=0.70,
        support_coverage=0.80,
        solver_verifiable=False,
        signature="sig-cipher",
        query="q-cipher",
    )
    equation = _example(
        "eq_silver",
        family="equation",
        confidence=0.70,
        support_coverage=0.80,
        solver_verifiable=False,
        signature="sig-eq",
        query="q-eq",
    )
    strict = _example("strict", family="cipher", signature="strict-sig")

    # Feed them in a family order that is NOT the priority order so the test
    # is meaningful.
    bundle = build_selected_sft_with_report(
        [strict, bit, cipher, equation],
        prompt_mode=builder.PROMPT_MODE_GENERIC,
        balance_by_family=False,
        hard_triad_repeat_factor=1,
        max_per_signature_bucket=0,
        enable_silver_official=True,
        silver_max_fraction=1.0,
        silver_max_absolute=1,  # take exactly one silver
    )
    silver_ids = [
        record["id"]
        for record in bundle["records"]
        if record["id"].endswith("_silver")
    ]
    assert silver_ids == ["eq_silver"], "equation must come first in silver priority"


def test_rejection_diagnostics_reports_per_family_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_annotate_identity(monkeypatch)
    examples = [
        _example("strict", family="cipher"),
        _example("cipher_low_conf", family="cipher", confidence=0.50),
        _example("equation_no_sig", family="equation", signature=None),
        _example("bit_partial", family="bit", support_coverage=0.5),
    ]
    bundle = build_selected_sft_with_report(
        examples,
        prompt_mode=builder.PROMPT_MODE_GENERIC,
        balance_by_family=False,
        hard_triad_repeat_factor=1,
        max_per_signature_bucket=0,
    )
    diagnostics = bundle["official_rejection_diagnostics"]
    assert diagnostics["cipher"]["low_teacher_confidence"] == 1
    assert diagnostics["equation"]["missing_program_signature"] == 1
    assert diagnostics["bit"]["partial_support_coverage"] == 1


def test_selection_counts_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_annotate_identity(monkeypatch)
    bundle = build_selected_sft_with_report(
        [_example("strict", family="cipher")],
        prompt_mode=builder.PROMPT_MODE_GENERIC,
        balance_by_family=False,
        hard_triad_repeat_factor=1,
        max_per_signature_bucket=0,
    )
    counts = bundle["selection_counts"]
    assert set(counts.keys()) == {"official_strict", "official_silver", "synth"}
    assert counts["official_strict"] == 1
    assert counts["official_silver"] == 0
