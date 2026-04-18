"""Harness-aligned prompt and completion helpers.

Contract source: the official Kaggle evaluator at
``官方资料/kaggle_metric/nvidia-nemotron-metric.ipynb``. Observed behaviour for
Nemotron-3-Nano-30B-A3B-BF16 tokenizer, captured in
``artifacts/chat_template_probe.json`` (probe run: 2026-04-18):

- ``apply_chat_template(..., enable_thinking=True)`` ends its output with
  ``"<|im_start|>assistant\\n<think>\\n"``.
- ``enable_thinking=False`` still emits a ``<think></think>`` pair; there is no
  way to avoid a thinking segment.

Consequence: every training completion must close the thinking segment
(``</think>\\n``) before the final ``\\boxed{...}`` answer, otherwise the
evaluator's ``extract_final_answer`` anchor cannot find the boxed answer at the
expected position.

See ``docs/harness_alignment.md`` for the full design rationale.
"""

from __future__ import annotations

from typing import Protocol

from src.competition.schema import PuzzleExample
from src.student.format_guard import wrap_boxed


HARNESS_GUARD: str = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
"""Exact guard text appended by the evaluator to every test ``prompt``.

Literal copy from ``metric/nvidia-nemotron-metric.ipynb`` cell 0, function
``generate_predictions``.
"""

THINKING_CLOSE: str = "</think>"
"""Token sequence that closes the Nemotron thinking segment."""

EXPECTED_CHAT_TEMPLATE_SHA16: str = "ab7813c3abdd9cb6"
"""Probe-derived SHA16 of the expected Nemotron chat template."""


class _SupportsChatTemplate(Protocol):
    """Structural type for any tokenizer exposing ``apply_chat_template``.

    Matches both the real ``transformers.PreTrainedTokenizerFast`` and the
    ``FakeNemotronTokenizer`` used in unit tests.
    """

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str: ...


def build_user_content(raw_prompt: str) -> str:
    """Append the harness guard to a raw prompt.

    This is what the evaluator passes as the ``content`` of the single user
    message before running ``apply_chat_template``.
    """
    return raw_prompt + HARNESS_GUARD


def build_chat_thinking_prompt(
    example: PuzzleExample,
    tokenizer: _SupportsChatTemplate,
) -> str:
    """Render a training-time prompt that is byte-identical to the evaluator's input.

    The returned string covers everything up to (and including) the ``<think>\\n``
    seed. The training completion is whatever follows the seed — by convention
    ``{trace_body}\\n</think>\\n\\boxed{answer}`` (see ``wrap_as_thinking``).
    """
    user_content = build_user_content(example.raw_prompt)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def wrap_as_thinking(body: str, target_answer: str) -> str:
    """Produce a completion that closes ``<think>`` and emits the boxed answer.

    The chat-template seed already supplies an opening ``<think>\\n``, so the
    completion continues inside the thinking segment. This helper appends
    ``</think>\\n\\boxed{target_answer}`` after an optional body. If ``body`` is
    empty (or becomes empty after rstrip), the resulting completion represents
    an empty-thinking segment, which remains a valid training target.

    Invariants:

    - The returned string contains exactly one ``\\boxed{...}``, at the tail.
    - The returned string contains exactly one ``</think>``.
    - The boxed answer is passed through :func:`wrap_boxed`, preserving any
      repo-wide answer-formatting conventions.
    """
    body_stripped = (body or "").rstrip()
    boxed = wrap_boxed(target_answer or "")
    if body_stripped:
        return f"{body_stripped}\n{THINKING_CLOSE}\n{boxed}"
    return f"{THINKING_CLOSE}\n{boxed}"
