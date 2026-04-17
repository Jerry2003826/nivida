from __future__ import annotations

from src.competition.schema import PuzzleExample
from src.student.format_guard import wrap_boxed


def compile_completion(example: PuzzleExample, *, style: str) -> str:
    boxed_answer = wrap_boxed(example.target_answer or "")
    if style == "answer_only":
        return boxed_answer

    family = example.metadata.official_family or "unknown"
    subtype = example.metadata.subtype or "unknown"
    program = example.metadata.program_signature or "identity"
    if style == "short_trace":
        return f"family={family}; subtype={subtype}; program={program}\n{boxed_answer}"
    if style == "token_trace":
        return f"fam={family}|sub={subtype}|prog={program}|{boxed_answer}"
    raise ValueError(f"Unsupported completion style: {style}")
