from __future__ import annotations

from src.competition.official_prompts import build_official_style_prompt, ensure_answer_contract
from src.competition.schema import PuzzleExample


def build_competition_prompt(example: PuzzleExample) -> str:
    """Use the official raw prompt whenever available and enforce a one-line answer contract."""
    if example.raw_prompt.strip():
        return ensure_answer_contract(example.raw_prompt)
    return build_official_style_prompt(example)
