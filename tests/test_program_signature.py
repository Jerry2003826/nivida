from __future__ import annotations

from src.teacher.program_signature import build_signature_bucket, canonicalize_candidate
from src.teacher.trace_compiler import render_short_trace, render_token_trace


def _candidate(mask: str = "11110000"):
    step = type("Step", (), {"op_name": "binary_xor_mask", "params": {"mask": mask}})()
    return type("Candidate", (), {"steps": [step]})()


def test_canonicalization_is_stable_for_same_candidate() -> None:
    signature_a = canonicalize_candidate(_candidate(), "bit", "bit_xor_mask")
    signature_b = canonicalize_candidate(_candidate(), "bit", "bit_xor_mask")
    assert signature_a.signature == signature_b.signature
    assert signature_a.signature_bucket == signature_b.signature_bucket


def test_different_params_change_signature_but_can_share_bucket() -> None:
    first = canonicalize_candidate(_candidate("11110000"), "bit", "bit_xor_mask")
    second = canonicalize_candidate(_candidate("00001111"), "bit", "bit_xor_mask")
    assert first.signature != second.signature
    assert first.signature_bucket == second.signature_bucket == build_signature_bucket(first.signature)


def test_trace_renderers_include_boxed_answer() -> None:
    signature = canonicalize_candidate(_candidate(), "bit", "bit_xor_mask")
    short_trace = render_short_trace(signature, "01101001")
    token_trace = render_token_trace(signature, "01101001")
    assert "\\boxed{01101001}" in short_trace
    assert "\\boxed{01101001}" in token_trace
    assert len(short_trace) < len("\\boxed{01101001}") + 120
