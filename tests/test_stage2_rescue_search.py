"""Tests for the stage2 rescue search (second-pass chain search for rejected
official hard-triad samples).

Rescue only runs when ``rescue_hard_triad=True`` is passed in, never touches
examples that already pass the strict gate, only overwrites annotation when
the quality tuple strictly improves, and reports per-family attempt/promotion
counters so the stage2 report can show the selector's effect.
"""
from __future__ import annotations

import pytest

from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair
from src.student import sft_dataset_builder as builder
from src.student.sft_dataset_builder import (
    HARD_TRIAD_RESCUE_FAMILIES,
    _annotation_quality_tuple,
    _rescue_official_hard_triad_examples,
    build_selected_sft_with_report,
)


def _rejected_official(
    example_id: str,
    *,
    family: str = "equation",
    confidence: float = 0.5,
    support_coverage: float = 0.5,
    solver_verifiable: bool = False,
    signature: str | None = None,
    query: str = "q",
    target: str = "ans",
) -> PuzzleExample:
    """Build an official hard-triad sample that fails the strict gate."""
    return PuzzleExample(
        id=example_id,
        raw_prompt="In Alice's Wonderland, ...",
        official_instruction="",
        parsed_examples=[PuzzlePair(input="i", output="o")],
        query=query,
        target_answer=target,
        metadata=PuzzleMetadata(
            official_family=family,
            subtype=f"{family}_sub",
            teacher_confidence=confidence,
            program_signature=signature,
            source="official",
            split="train",
            extras={
                "solver_verifiable": solver_verifiable,
                "support_coverage": support_coverage,
                "top1_top2_margin": 0.0,
            },
        ),
    )


def _strict_official(example_id: str, *, family: str = "equation") -> PuzzleExample:
    """An official sample that already passes the strict gate."""
    return PuzzleExample(
        id=example_id,
        raw_prompt="...",
        official_instruction="",
        parsed_examples=[PuzzlePair(input="i", output="o")],
        query="q",
        target_answer="a",
        metadata=PuzzleMetadata(
            official_family=family,
            subtype=f"{family}_sub",
            teacher_confidence=0.95,
            program_signature="sig-strict",
            source="official",
            split="train",
            extras={
                "solver_verifiable": True,
                "support_coverage": 1.0,
                "top1_top2_margin": 0.2,
            },
        ),
    )


class _FakeCandidate:
    """Shape compatible with annotate_example_from_candidates()."""

    def __init__(
        self,
        *,
        score: float,
        confidence: float,
        query_prediction: str,
        predictions: list[str],
        steps: list = (),
    ) -> None:
        self.score = score
        self.confidence = confidence
        self.query_prediction = query_prediction
        self.predictions = list(predictions)
        self.steps = list(steps)

    def to_debug_dict(self) -> dict:
        return {}


def _patch_engine(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidates_by_id: dict[str, list[_FakeCandidate]],
) -> list[tuple[str, int, int, int]]:
    """Replace ChainSearchEngine with a stub that returns canned candidates.

    Returns a log of (example_id, beam_width, max_depth, top_k) for each
    solve_example call so the tests can assert rescue settings propagated.
    """
    calls: list[tuple[str, int, int, int]] = []

    class _StubEngine:
        def __init__(self, *, beam_width: int, max_depth: int) -> None:
            self._beam_width = beam_width
            self._max_depth = max_depth

        def solve_example(self, example: PuzzleExample, *, top_k: int):
            calls.append((example.id, self._beam_width, self._max_depth, top_k))
            return list(candidates_by_id.get(example.id, []))

    monkeypatch.setattr(builder, "ChainSearchEngine", _StubEngine)
    return calls


def _patch_first_pass_annotate_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Short-circuit the first _annotate_examples call; we want to exercise rescue only."""

    def _identity(examples, **_kwargs):
        return list(examples)

    monkeypatch.setattr(builder, "_annotate_examples", _identity)


# --- annotation_quality_tuple ordering ---------------------------------------


def test_quality_tuple_prefers_solver_verifiable() -> None:
    verified = _rejected_official("v", solver_verifiable=True)
    not_verified = _rejected_official("n", solver_verifiable=False)
    assert _annotation_quality_tuple(verified) > _annotation_quality_tuple(not_verified)


def test_quality_tuple_prefers_more_support() -> None:
    more = _rejected_official("more", support_coverage=0.9)
    less = _rejected_official("less", support_coverage=0.3)
    assert _annotation_quality_tuple(more) > _annotation_quality_tuple(less)


def test_quality_tuple_prefers_presence_of_signature() -> None:
    has = _rejected_official("h", signature="sig-present")
    lacks = _rejected_official("l", signature=None)
    assert _annotation_quality_tuple(has) > _annotation_quality_tuple(lacks)


# --- _rescue_official_hard_triad_examples -------------------------------------


def test_rescue_skips_samples_that_already_pass_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_engine(monkeypatch, candidates_by_id={})
    examples = [_strict_official("strict_eq", family="equation")]
    diagnostics = _rescue_official_hard_triad_examples(
        examples, families={"equation"}
    )
    # Strict example should never be fed to the rescue engine.
    assert [c[0] for c in calls] == []
    assert diagnostics["rescue_attempted"] == {}


def test_rescue_skips_non_hard_triad(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_engine(monkeypatch, candidates_by_id={})
    examples = [
        _rejected_official("easy_numeral", family="numeral"),
        _rejected_official("easy_unit", family="unit"),
    ]
    diagnostics = _rescue_official_hard_triad_examples(
        examples, families=HARD_TRIAD_RESCUE_FAMILIES
    )
    assert calls == []
    assert diagnostics["rescue_attempted"] == {}


def test_rescue_skips_non_official_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_engine(monkeypatch, candidates_by_id={})
    example = _rejected_official("synthetic_eq", family="equation")
    example.metadata.source = "synthetic"
    diagnostics = _rescue_official_hard_triad_examples(
        [example], families=HARD_TRIAD_RESCUE_FAMILIES
    )
    assert calls == []
    assert diagnostics["rescue_attempted"] == {}


def test_rescue_promotes_only_when_quality_strictly_improves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Candidate 1: rescue succeeds. Stub supplies a candidate whose chain-search
    # annotation would fill program_signature + raise confidence + make the
    # support match the pair, so annotate_example_from_candidates produces a
    # strictly better tuple than the rejected input.
    winning_candidate = _FakeCandidate(
        score=0.9,
        confidence=0.85,
        query_prediction="a",
        predictions=["o"],
        steps=[
            type("Step", (), {"op_name": "identity", "params": {}})(),
        ],
    )
    # Candidate 2: rescue should NOT promote — we'll hand it an empty candidate
    # list so annotate resets signature to None, which is strictly worse.
    losing_candidate_list: list[_FakeCandidate] = []

    calls = _patch_engine(
        monkeypatch,
        candidates_by_id={
            "good_eq": [winning_candidate],
            "bad_eq": losing_candidate_list,
        },
    )

    good = _rejected_official(
        "good_eq",
        family="equation",
        confidence=0.4,
        support_coverage=0.3,
        solver_verifiable=False,
        signature=None,
    )
    bad_before = _rejected_official(
        "bad_eq",
        family="equation",
        confidence=0.7,  # already reasonably high
        support_coverage=0.9,
        solver_verifiable=True,
        signature="sig-original",
    )

    diagnostics = _rescue_official_hard_triad_examples(
        [good, bad_before], families={"equation"}
    )

    # Both examples were attempted, only one was promoted.
    assert diagnostics["rescue_attempted"] == {"equation": 2}
    assert diagnostics["rescue_promoted"] == {"equation": 1}

    # The promoted example carries the rescue marker.
    assert good.metadata.extras.get("second_pass_rescue_applied") is True
    # ``use_search_subtype_hint`` defaults to False for canonical rescue;
    # the setting is recorded on every promoted sample for audit parity
    # with the experimental subtype-rescue branch.
    assert good.metadata.extras.get("second_pass_settings") == {
        "beam_width": 12,
        "max_depth": 4,
        "top_k": 3,
        "use_search_subtype_hint": False,
    }

    # The non-promoted example was rolled back to its original (better) state.
    assert bad_before.metadata.program_signature == "sig-original"
    assert bad_before.metadata.extras.get("solver_verifiable") is True
    assert "second_pass_rescue_applied" not in bad_before.metadata.extras

    # Both examples made it to the stub engine.
    called_ids = {c[0] for c in calls}
    assert called_ids == {"good_eq", "bad_eq"}


def test_rescue_settings_propagate_to_stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_engine(monkeypatch, candidates_by_id={})
    example = _rejected_official("eq", family="equation")
    _rescue_official_hard_triad_examples(
        [example],
        families={"equation"},
        beam_width=9,
        max_depth=5,
        top_k=7,
    )
    assert calls == [("eq", 9, 5, 7)]


def test_rescue_with_empty_family_set_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_engine(monkeypatch, candidates_by_id={})
    example = _rejected_official("eq", family="equation")
    diagnostics = _rescue_official_hard_triad_examples([example], families=set())
    assert calls == []
    assert diagnostics["rescue_attempted"] == {}
    assert diagnostics["rescue_families"] == []


# --- build_selected_sft_with_report integration -------------------------------


def test_build_selected_sft_reports_rescue_diagnostics_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_first_pass_annotate_noop(monkeypatch)
    example = _strict_official("strict_eq", family="equation")
    bundle = build_selected_sft_with_report(
        [example],
        prompt_mode=builder.PROMPT_MODE_GENERIC,
        balance_by_family=False,
        hard_triad_repeat_factor=1,
        max_per_signature_bucket=0,
    )
    diagnostics = bundle["rescue_diagnostics"]
    # When rescue is off, the diagnostics block still exists for a stable
    # report shape, but all counters are empty.
    assert diagnostics["rescue_attempted"] == {}
    assert diagnostics["rescue_promoted"] == {}
    assert diagnostics["rescue_families"] == []


def test_build_selected_sft_runs_rescue_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_first_pass_annotate_noop(monkeypatch)
    winning = _FakeCandidate(
        score=0.9,
        confidence=0.85,
        query_prediction="a",
        predictions=["o"],
        steps=[type("Step", (), {"op_name": "identity", "params": {}})()],
    )
    _patch_engine(monkeypatch, candidates_by_id={"recover_eq": [winning]})

    strict = _strict_official("strict_eq")
    rejected = _rejected_official(
        "recover_eq",
        family="equation",
        confidence=0.3,
        support_coverage=0.2,
        solver_verifiable=False,
        signature=None,
    )

    bundle = build_selected_sft_with_report(
        [strict, rejected],
        prompt_mode=builder.PROMPT_MODE_GENERIC,
        balance_by_family=False,
        hard_triad_repeat_factor=1,
        max_per_signature_bucket=0,
        rescue_hard_triad=True,
        rescue_families={"equation"},
    )
    diagnostics = bundle["rescue_diagnostics"]
    assert diagnostics["rescue_attempted"] == {"equation": 1}
    assert diagnostics["rescue_promoted"] == {"equation": 1}
    assert diagnostics["rescue_families"] == ["equation"]
    # ``use_search_subtype_hint`` defaults to False in canonical runs so
    # the experimental subtype hint cannot accidentally activate.
    assert diagnostics["rescue_settings"] == {
        "beam_width": 12,
        "max_depth": 4,
        "top_k": 3,
        "use_search_subtype_hint": False,
    }
