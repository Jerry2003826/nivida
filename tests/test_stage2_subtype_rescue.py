"""Tests for the Stage2 subtype-prior rescue (v1) branch.

These tests guard the three technical invariants behind the design:

1. ``_infer_subtype_hint_from_top_steps`` never invents a no-op
   ``cipher_vocab -> cipher_vocab`` transition (the generic fallback
   already routes through ``vocabulary_cipher``).
2. ``_rescue_official_hard_triad_examples`` temporarily overrides
   ``example.metadata.subtype`` — NOT just ``extras["subtype"]`` — before
   it calls ``ChainSearchEngine.solve_example``. Changing extras alone
   would be a silent no-op because the engine reads
   ``example.metadata.subtype`` directly.
3. When the rescue search is not promoted, every mutation performed by
   the hint path (subtype override, ``rescue_subtype_hint`` /
   ``rescue_original_subtype`` extras, annotation side effects) is
   rolled back so a failed rescue attempt cannot leak any state into
   downstream record rendering.

Tests use monkeypatch to stub ``ChainSearchEngine`` +
``annotate_example_from_candidates`` so behaviour is deterministic and
fast; no real chain search is ever invoked.
"""
from __future__ import annotations

import pytest

from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair
from src.student import sft_dataset_builder as builder
from src.student.sft_dataset_builder import (
    STAGE2_FALLBACK_SUBTYPES,
    _attach_search_subtype_hints,
    _infer_subtype_hint_from_top_steps,
    _rescue_official_hard_triad_examples,
    _step_names_from_example,
    build_selected_sft_with_report,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_example(
    example_id: str,
    *,
    family: str,
    subtype: str,
    top_steps: list[str] | None = None,
    confidence: float = 0.3,
    solver_verifiable: bool = False,
    support_coverage: float = 0.0,
    signature: str | None = None,
    source: str = "official",
    target: str = "one two",
    query: str = "alpha beta",
) -> PuzzleExample:
    extras: dict = {
        "solver_verifiable": solver_verifiable,
        "support_coverage": support_coverage,
        "top1_top2_margin": 0.0,
        "top_candidate_steps": list(top_steps or []),
    }
    return PuzzleExample(
        id=example_id,
        raw_prompt="Example prompt ...",
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
            subtype=subtype,
            teacher_confidence=confidence,
            program_signature=signature,
            source=source,
            split="train",
            extras=extras,
        ),
    )


def _patch_rescue_search(
    monkeypatch: pytest.MonkeyPatch,
    *,
    promote: bool,
    record_subtypes: list[str | None] | None = None,
) -> None:
    """Replace the rescue search path so it is deterministic.

    - ``record_subtypes`` collects ``example.metadata.subtype`` as seen by
      ``solve_example``. This is exactly what the real engine reads on
      line ``chain_search.py::solve_example`` to pick an op priority, so
      asserting on this list proves the override landed on the right
      attribute.
    - ``promote=True`` makes the faked annotation write a strictly higher
      quality tuple than the baseline so the rescue takes the promoted
      branch. ``promote=False`` mimics a degenerate second-pass result
      (no candidate) so the quality tuple regresses and the function
      takes the rollback branch.
    """
    sentinel = object()

    class _FakeEngine:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def solve_example(self, example, *, top_k=5):
            if record_subtypes is not None:
                record_subtypes.append(example.metadata.subtype)
            return [sentinel] if promote else []

    def _fake_annotate(example, candidates):
        if not candidates:
            example.metadata.program_signature = None
            example.metadata.teacher_confidence = None
            example.metadata.extras = {
                **dict(example.metadata.extras or {}),
                "support_coverage": 0.0,
                "top1_top2_margin": 0.0,
                "solver_verifiable": False,
                "top_candidate_score": None,
                "top_candidate_steps": [],
            }
            return example
        example.metadata.program_signature = "sig-rescued"
        example.metadata.teacher_confidence = 0.99
        example.metadata.composition_key = "rescued|rescued|rescued"
        example.metadata.extras = {
            **dict(example.metadata.extras or {}),
            "solver_verifiable": True,
            "support_coverage": 1.0,
            "top1_top2_margin": 1.0,
            "top_candidate_score": 0.9,
            "top_candidate_steps": ["rescued_op"],
        }
        return example

    monkeypatch.setattr(builder, "ChainSearchEngine", _FakeEngine)
    monkeypatch.setattr(builder, "annotate_example_from_candidates", _fake_annotate)


def _patch_annotate_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the first-pass annotation pass (used by build_selected_sft_with_report)."""

    def _identity(examples, **_kwargs):
        return list(examples)

    monkeypatch.setattr(builder, "_annotate_examples", _identity)


# ---------------------------------------------------------------------------
# _step_names_from_example
# ---------------------------------------------------------------------------


def test_step_names_from_example_handles_empty_and_malformed() -> None:
    empty = _make_example("a", family="equation", subtype="equation_symbolic")
    assert _step_names_from_example(empty) == set()

    bad = _make_example("b", family="equation", subtype="equation_symbolic")
    bad.metadata.extras["top_candidate_steps"] = "not-a-list"
    assert _step_names_from_example(bad) == set()


def test_step_names_from_example_returns_set_of_op_names() -> None:
    example = _make_example(
        "c",
        family="equation",
        subtype="equation_symbolic",
        top_steps=["operator_template", "position_transducer"],
    )
    assert _step_names_from_example(example) == {
        "operator_template",
        "position_transducer",
    }


# ---------------------------------------------------------------------------
# _infer_subtype_hint_from_top_steps — equation
# ---------------------------------------------------------------------------


def test_infer_equation_template_hint() -> None:
    example = _make_example(
        "eq-t",
        family="equation",
        subtype="equation_symbolic",
        top_steps=["operator_template"],
    )
    assert _infer_subtype_hint_from_top_steps(example) == "equation_template"


def test_infer_equation_position_hint() -> None:
    example = _make_example(
        "eq-p",
        family="equation",
        subtype="equation_symbolic",
        top_steps=["position_transducer"],
    )
    assert _infer_subtype_hint_from_top_steps(example) == "equation_position"


def test_infer_equation_delete_hint() -> None:
    example = _make_example(
        "eq-d",
        family="equation",
        subtype="equation_symbolic",
        top_steps=["delete_characters"],
    )
    assert _infer_subtype_hint_from_top_steps(example) == "equation_delete"


def test_equation_numeric_steps_do_not_emit_unsupported_subtype_hint() -> None:
    """``equation_numeric`` is not recognised by
    :meth:`ChainSearchEngine._prioritized_op_names` — numeric equation ops
    are routed via ``equation_mode == "numeric"``, not via the subtype
    string. Returning ``equation_numeric`` as a hint would silently fall
    through to the default symbolic op priority and pollute the
    ``rescue_promoted_with_hint`` diagnostic with deeper-search wins that
    actually have nothing to do with the hint. The inference function
    must therefore emit ``None`` whenever it sees numeric-family op
    evidence, until ``chain_search.py`` grows a dedicated branch.
    """
    for top_steps in (
        ["add_constant"],
        ["multiply_constant"],
        ["affine_transform"],
        ["evaluate_expression"],
        ["binary_equation_rule"],
        ["add_constant", "multiply_constant"],
    ):
        example = _make_example(
            "eq-n",
            family="equation",
            subtype="equation_symbolic",
            top_steps=top_steps,
        )
        assert _infer_subtype_hint_from_top_steps(example) is None, (
            f"equation_numeric hint must be disabled until chain_search supports it; "
            f"got a non-None hint for top_steps={top_steps!r}"
        )


def test_infer_skips_non_fallback_subtype() -> None:
    # Already routed to a specific subtype; hint must NOT overwrite it.
    example = _make_example(
        "eq-already",
        family="equation",
        subtype="equation_template",
        top_steps=["operator_template"],
    )
    assert _infer_subtype_hint_from_top_steps(example) is None


# ---------------------------------------------------------------------------
# _infer_subtype_hint_from_top_steps — cipher (push back #2)
# ---------------------------------------------------------------------------


def test_cipher_vocab_only_does_not_emit_noop_hint() -> None:
    """``cipher_vocab`` already routes to ``[vocabulary_cipher, fixed_substitution]``
    in the op priority. Inferring ``cipher_token_sub`` or re-emitting
    ``cipher_vocab`` from a lone ``vocabulary_cipher`` signal would be a
    silent no-op and would still dirty the rescue diagnostics.
    """
    example = _make_example(
        "c-vocab",
        family="cipher",
        subtype="cipher_vocab",
        top_steps=["vocabulary_cipher"],
    )
    assert _infer_subtype_hint_from_top_steps(example) is None


def test_cipher_perm_hint_requires_substitution_and_permutation_steps() -> None:
    example = _make_example(
        "c-perm",
        family="cipher",
        subtype="cipher_vocab",
        top_steps=["vocabulary_cipher", "reverse_tokens"],
    )
    assert _infer_subtype_hint_from_top_steps(example) == "cipher_perm"


def test_cipher_perm_hint_requires_both_signals() -> None:
    # Only permutation, no substitution → should not promote to cipher_perm.
    example = _make_example(
        "c-perm-only",
        family="cipher",
        subtype="cipher_vocab",
        top_steps=["reverse_tokens"],
    )
    assert _infer_subtype_hint_from_top_steps(example) is None


def test_cipher_caesar_emits_char_sub_hint() -> None:
    example = _make_example(
        "c-caesar",
        family="cipher",
        subtype="cipher_vocab",
        top_steps=["caesar_shift"],
    )
    assert _infer_subtype_hint_from_top_steps(example) == "cipher_char_sub"


# ---------------------------------------------------------------------------
# _infer_subtype_hint_from_top_steps — bit
# ---------------------------------------------------------------------------


def test_bit_rotate_and_mask_hints() -> None:
    rotate = _make_example(
        "b-rot",
        family="bit",
        subtype="bit_affine",
        top_steps=["binary_rotate_left"],
    )
    assert _infer_subtype_hint_from_top_steps(rotate) == "bit_rotate"

    mask = _make_example(
        "b-xor",
        family="bit",
        subtype="bit_affine",
        top_steps=["binary_xor_mask"],
    )
    assert _infer_subtype_hint_from_top_steps(mask) == "bit_xor_mask"


def test_bit_affine_no_op_returns_none() -> None:
    # ``binary_affine_transform`` already corresponds to ``bit_affine``;
    # the fallback subtype is ``bit_affine`` so this is a deliberate no-op.
    example = _make_example(
        "b-affine",
        family="bit",
        subtype="bit_affine",
        top_steps=["binary_affine_transform"],
    )
    assert _infer_subtype_hint_from_top_steps(example) is None


# ---------------------------------------------------------------------------
# _attach_search_subtype_hints
# ---------------------------------------------------------------------------


def test_attach_search_subtype_hints_writes_only_extras_and_records_transitions() -> None:
    target = _make_example(
        "eq",
        family="equation",
        subtype="equation_symbolic",
        top_steps=["operator_template"],
    )
    non_fallback = _make_example(
        "eq-already-routed",
        family="equation",
        subtype="equation_template",
        top_steps=["operator_template"],
    )
    non_official = _make_example(
        "eq-synth",
        family="equation",
        subtype="equation_symbolic",
        top_steps=["operator_template"],
        source="synthetic",
    )

    stats = _attach_search_subtype_hints([target, non_fallback, non_official])

    assert target.metadata.extras["search_subtype_hint"] == "equation_template"
    # subtype itself must NOT be touched by the attach pass — only the
    # rescue path may temporarily override it.
    assert target.metadata.subtype == "equation_symbolic"

    assert "search_subtype_hint" not in non_fallback.metadata.extras
    assert "search_subtype_hint" not in non_official.metadata.extras

    assert stats["hinted"] == {"equation": 1}
    assert stats["transitions"] == {"equation_symbolic->equation_template": 1}


# ---------------------------------------------------------------------------
# _rescue_official_hard_triad_examples — push back #3: override metadata.subtype
# ---------------------------------------------------------------------------


def test_rescue_second_pass_temporarily_overrides_metadata_subtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ChainSearchEngine.solve_example`` reads ``example.metadata.subtype``
    directly, not ``extras['subtype']``. The rescue path MUST override
    ``metadata.subtype`` before calling solve, otherwise the hint is a
    silent no-op. This test records the value ``solve_example`` received
    and asserts it is the hinted subtype, not the original fallback.
    """
    example = _make_example(
        "eq-hint",
        family="equation",
        subtype="equation_symbolic",
        top_steps=["operator_template"],
    )
    example.metadata.extras["search_subtype_hint"] = "equation_template"

    received: list[str | None] = []
    _patch_rescue_search(monkeypatch, promote=True, record_subtypes=received)

    diag = _rescue_official_hard_triad_examples(
        [example],
        families={"equation"},
        use_search_subtype_hint=True,
    )

    assert received == ["equation_template"], (
        "solve_example must see metadata.subtype=equation_template; "
        f"got {received!r}. The hint path is failing to override "
        "metadata.subtype — ChainSearchEngine reads metadata.subtype "
        "directly, so extras-only changes would be silent no-ops."
    )

    # Promoted → the override is kept alongside the new annotation,
    # and the audit markers identify the subtype change.
    assert diag["rescue_attempted"] == {"equation": 1}
    assert diag["rescue_promoted"] == {"equation": 1}
    assert diag["rescue_hint_attempted"] == {"equation": 1}
    assert diag["rescue_promoted_with_hint"] == {"equation": 1}
    assert diag["rescue_hint_transitions"] == {"equation_symbolic->equation_template": 1}
    assert example.metadata.subtype == "equation_template"
    assert example.metadata.extras["rescue_subtype_hint"] == "equation_template"
    assert example.metadata.extras["rescue_original_subtype"] == "equation_symbolic"
    # Rescue markers landed.
    assert example.metadata.extras["second_pass_rescue_applied"] is True
    assert (
        example.metadata.extras["second_pass_settings"]["use_search_subtype_hint"]
        is True
    )


# ---------------------------------------------------------------------------
# _rescue_official_hard_triad_examples — push back #3: rollback completeness
# ---------------------------------------------------------------------------


def test_rescue_not_promoted_rolls_back_subtype_and_hint_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the second-pass search does not strictly beat the first pass,
    EVERY mutation made along the hint path must roll back: the subtype
    override, the ``rescue_subtype_hint`` / ``rescue_original_subtype``
    extras, and the annotation fields (``program_signature``,
    ``teacher_confidence``, ``composition_key``, ``extras``). A partial
    rollback would leak experimental state into canonical record
    rendering downstream.
    """
    example = _make_example(
        "eq-rollback",
        family="equation",
        subtype="equation_symbolic",
        top_steps=["operator_template"],
        signature="sig-initial",
        confidence=0.3,
    )
    example.metadata.composition_key = "equation|equation_symbolic|initial"
    example.metadata.extras["search_subtype_hint"] = "equation_template"
    initial_extras = dict(example.metadata.extras)

    _patch_rescue_search(monkeypatch, promote=False)

    diag = _rescue_official_hard_triad_examples(
        [example],
        families={"equation"},
        use_search_subtype_hint=True,
    )

    assert diag["rescue_attempted"] == {"equation": 1}
    assert diag["rescue_promoted"] == {}
    # hint_attempted counts the override attempt regardless of promotion.
    assert diag["rescue_hint_attempted"] == {"equation": 1}
    assert diag["rescue_promoted_with_hint"] == {}

    # Full rollback of every field touched by the hint path.
    assert example.metadata.subtype == "equation_symbolic"
    assert example.metadata.program_signature == "sig-initial"
    assert example.metadata.teacher_confidence == pytest.approx(0.3)
    assert example.metadata.composition_key == "equation|equation_symbolic|initial"
    # rescue_subtype_hint / rescue_original_subtype must not leak into
    # the canonical extras dict after a rolled-back attempt.
    assert "rescue_subtype_hint" not in example.metadata.extras
    assert "rescue_original_subtype" not in example.metadata.extras
    # The pre-rescue ``search_subtype_hint`` marker stays — it was set
    # by the attach pass, not the rescue pass, and carries diagnostic
    # value for the next iteration.
    assert example.metadata.extras == initial_extras


def test_rescue_without_hint_flag_does_not_override_subtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``use_search_subtype_hint`` is False (canonical stage2), the
    rescue path must NEVER read ``search_subtype_hint`` and must NEVER
    override ``metadata.subtype``. This protects canonical runs from
    accidental behaviour drift if the extras carry a stale hint.
    """
    example = _make_example(
        "eq-no-hint",
        family="equation",
        subtype="equation_symbolic",
        top_steps=["operator_template"],
    )
    example.metadata.extras["search_subtype_hint"] = "equation_template"

    received: list[str | None] = []
    _patch_rescue_search(monkeypatch, promote=True, record_subtypes=received)

    _rescue_official_hard_triad_examples(
        [example],
        families={"equation"},
        use_search_subtype_hint=False,
    )

    assert received == ["equation_symbolic"]
    assert "rescue_subtype_hint" not in example.metadata.extras
    assert "rescue_original_subtype" not in example.metadata.extras


# ---------------------------------------------------------------------------
# End-to-end: build_selected_sft_with_report returns subtype_hint_diagnostics
# ---------------------------------------------------------------------------


def test_report_includes_subtype_hint_diagnostics_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_annotate_identity(monkeypatch)
    _patch_rescue_search(monkeypatch, promote=True)

    strict = _make_example(
        "strict",
        family="cipher",
        subtype="cipher_sub",
        confidence=0.9,
        solver_verifiable=True,
        support_coverage=1.0,
        signature="sig-strict",
    )
    equation_hintable = _make_example(
        "eq-hint",
        family="equation",
        subtype="equation_symbolic",
        top_steps=["operator_template"],
        confidence=0.3,
    )

    bundle = build_selected_sft_with_report(
        [strict, equation_hintable],
        prompt_mode=builder.PROMPT_MODE_GENERIC,
        balance_by_family=False,
        hard_triad_repeat_factor=1,
        max_per_signature_bucket=0,
        rescue_hard_triad=True,
        rescue_families={"equation"},
        stage2_use_search_subtype_hint=True,
    )

    diag = bundle["subtype_hint_diagnostics"]
    assert diag["enabled"] is True
    assert diag["hinted"] == {"equation": 1}
    assert diag["transitions"] == {"equation_symbolic->equation_template": 1}
    assert diag["rescue_hint_attempted"] == {"equation": 1}
    assert diag["rescue_promoted_with_hint"] == {"equation": 1}
    assert diag["rescue_hint_transitions"] == {
        "equation_symbolic->equation_template": 1
    }


def test_report_subtype_hint_diagnostics_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_annotate_identity(monkeypatch)

    strict = _make_example(
        "strict",
        family="cipher",
        subtype="cipher_sub",
        confidence=0.9,
        solver_verifiable=True,
        support_coverage=1.0,
        signature="sig-strict",
    )
    bundle = build_selected_sft_with_report(
        [strict],
        prompt_mode=builder.PROMPT_MODE_GENERIC,
        balance_by_family=False,
        hard_triad_repeat_factor=1,
        max_per_signature_bucket=0,
    )

    diag = bundle["subtype_hint_diagnostics"]
    assert diag["enabled"] is False
    assert diag["hinted"] == {}
    assert diag["transitions"] == {}
    assert diag["rescue_hint_attempted"] == {}
    assert diag["rescue_promoted_with_hint"] == {}
    assert diag["rescue_hint_transitions"] == {}


# ---------------------------------------------------------------------------
# Invariants: the fallback constant matches family_tagger.py
# ---------------------------------------------------------------------------


def test_fallback_subtype_map_covers_hard_triad_families() -> None:
    assert set(STAGE2_FALLBACK_SUBTYPES.keys()) == {"equation", "cipher", "bit"}
    assert STAGE2_FALLBACK_SUBTYPES["equation"] == frozenset({"equation_symbolic"})
    assert STAGE2_FALLBACK_SUBTYPES["cipher"] == frozenset({"cipher_vocab"})
    assert STAGE2_FALLBACK_SUBTYPES["bit"] == frozenset({"bit_affine"})
