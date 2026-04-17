from __future__ import annotations

import re

from src.competition.official_prompts import ANSWER_CONTRACT, build_official_style_prompt
from src.competition.schema import PuzzleExample


PROMPT_MODE_RAW_WITH_GUARD = "raw_with_guard"
PROMPT_MODE_GENERIC = "generic"
PROMPT_MODE_CHAT_THINKING = "chat_thinking"
"""Harness-equivalent prompt mode.

Requires a tokenizer implementing ``apply_chat_template``; the rendering path
lives in :mod:`src.competition.harness_prompt` because
:func:`build_competition_prompt` cannot depend on a live tokenizer. Callers
that select this mode must resolve the prompt themselves via
``build_chat_thinking_prompt``.
"""
VALID_PROMPT_MODES = {
    PROMPT_MODE_RAW_WITH_GUARD,
    PROMPT_MODE_GENERIC,
    PROMPT_MODE_CHAT_THINKING,
}
_CONTRACT_HINT_RE = re.compile(r"(\\boxed\{|boxed|final answer)", flags=re.IGNORECASE)


def _append_single_guard(prompt: str) -> str:
    stripped = prompt.rstrip()
    if not stripped:
        return ANSWER_CONTRACT
    if ANSWER_CONTRACT in stripped or _CONTRACT_HINT_RE.search(stripped):
        return stripped
    return f"{stripped}\n{ANSWER_CONTRACT}"


def build_raw_prompt_with_guard(example: PuzzleExample) -> str:
    if example.raw_prompt.strip():
        return _append_single_guard(example.raw_prompt)
    return build_generic_prompt(example)


def build_generic_prompt(example: PuzzleExample) -> str:
    return build_official_style_prompt(example)


def build_competition_prompt(example: PuzzleExample, mode: str = PROMPT_MODE_RAW_WITH_GUARD) -> str:
    if mode == PROMPT_MODE_RAW_WITH_GUARD:
        return build_raw_prompt_with_guard(example)
    if mode == PROMPT_MODE_GENERIC:
        return build_generic_prompt(example)
    if mode == PROMPT_MODE_CHAT_THINKING:
        raise ValueError(
            "chat_thinking prompt mode requires a tokenizer; call "
            "src.competition.harness_prompt.build_chat_thinking_prompt(example, tokenizer) "
            "directly instead of build_competition_prompt(example, mode=...)."
        )
    raise ValueError(f"Unsupported prompt mode: {mode}")
