from __future__ import annotations

from src.competition.schema import PuzzleExample
from src.student.format_guard import wrap_boxed
from src.teacher.program_signature import ProgramSignature, build_signature_bucket


def render_answer_only(target_answer: str) -> str:
    return wrap_boxed(target_answer or "")


def render_short_trace_body(signature: ProgramSignature) -> str:
    """Return the trace metadata without the boxed answer.

    Used by :func:`wrap_as_thinking` when the answer is emitted separately
    after a ``</think>`` close. Keeps the trace body free of any ``\\boxed{}``
    so the evaluator's last-boxed-anchor rule stays unambiguous.
    """
    parts = [f"family={signature.official_family}"]
    if signature.subtype:
        parts.append(f"sub={signature.subtype}")
    parts.append(f"sig={signature.signature}")
    return "; ".join(parts)


def render_token_trace_body(signature: ProgramSignature) -> str:
    """Return the pipe-separated trace metadata without the boxed answer."""
    subtype = signature.subtype or "unknown"
    return (
        f"fam={signature.official_family}|sub={subtype}|sig={signature.signature}"
    )


def render_short_trace(signature: ProgramSignature, target_answer: str) -> str:
    answer = render_answer_only(target_answer)
    body = render_short_trace_body(signature)
    return f"{body}; answer={answer}"


def render_token_trace(signature: ProgramSignature, target_answer: str) -> str:
    answer = render_answer_only(target_answer)
    body = render_token_trace_body(signature)
    return f"{body}|{answer}"


def _signature_from_example(example: PuzzleExample) -> ProgramSignature:
    family = example.metadata.official_family or "unknown"
    subtype = example.metadata.subtype
    signature = example.metadata.program_signature or "identity"
    signature_bucket = (
        example.metadata.extras.get("program_signature_bucket")
        or build_signature_bucket(signature)
    )
    return ProgramSignature(
        official_family=family,
        subtype=subtype,
        steps=[],
        parameters=[],
        depth=0,
        signature=signature,
        signature_bucket=signature_bucket,
    )


def compile_completion(example: PuzzleExample, *, style: str) -> str:
    signature = _signature_from_example(example)
    if style == "answer_only":
        return render_answer_only(example.target_answer or "")
    if style == "short_trace":
        return render_short_trace(signature, example.target_answer or "")
    if style == "token_trace":
        return render_token_trace(signature, example.target_answer or "")
    raise ValueError(f"Unsupported completion style: {style}")


def compile_completion_body(example: PuzzleExample, *, style: str) -> str:
    """Return the body portion of a completion without the boxed answer.

    Consumed by thinking-mode completions where the body becomes the content
    of the ``<think>`` segment and the boxed answer is appended after
    ``</think>`` (see :func:`src.competition.harness_prompt.wrap_as_thinking`).

    For ``style="answer_only"`` this returns an empty string, producing an
    empty-thinking completion.
    """
    if style == "answer_only":
        return ""
    signature = _signature_from_example(example)
    if style == "short_trace":
        return render_short_trace_body(signature)
    if style == "token_trace":
        return render_token_trace_body(signature)
    raise ValueError(f"Unsupported completion style: {style}")
