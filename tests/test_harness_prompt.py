"""Byte-level contract tests for the harness-aligned prompt/completion helpers.

The ``FakeNemotronTokenizer`` below replicates the chat-template output
captured by ``scripts/probe_chat_template.py`` on 2026-04-18:

    apply_chat_template(..., enable_thinking=True) →
        '<|im_start|>system\\n<|im_end|>\\n'
        '<|im_start|>user\\n{content}<|im_end|>\\n'
        '<|im_start|>assistant\\n<think>\\n'

If the real Nemotron tokenizer is updated and the chat_template_sha16 stored
in artifacts/chat_template_probe.json drifts, this fake must be updated too.
"""

from __future__ import annotations

from src.competition.harness_prompt import (
    EXPECTED_CHAT_TEMPLATE_SHA16,
    HARNESS_GUARD,
    THINKING_CLOSE,
    build_chat_thinking_prompt,
    build_user_content,
    wrap_as_thinking,
)
from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair


class FakeNemotronTokenizer:
    """Byte-level replica of the Nemotron chat template observed on 2026-04-18."""

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is False, "probe fixture only supports tokenize=False"
        out = ["<|im_start|>system\n<|im_end|>\n"]
        for msg in messages:
            out.append(
                f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            )
        if add_generation_prompt:
            if enable_thinking:
                out.append("<|im_start|>assistant\n<think>\n")
            else:
                out.append("<|im_start|>assistant\n<think></think>")
        return "".join(out)


def _make_example() -> PuzzleExample:
    return PuzzleExample(
        id="ex-1",
        raw_prompt="Solve this puzzle.",
        official_instruction="",
        parsed_examples=[PuzzlePair(input="a", output="b")],
        query="x",
        target_answer="42",
        metadata=PuzzleMetadata(
            official_family="equation",
            subtype="equation_numeric",
            source="official",
            split="train",
        ),
    )


def test_harness_guard_matches_metric_kernel_literal() -> None:
    """Guard text must be byte-identical to the metric kernel literal."""
    # Source: 官方资料/kaggle_metric/nvidia-nemotron-metric.ipynb cell 0,
    # generate_predictions().
    expected = (
        "\nPlease put your final answer inside `\\boxed{}`. "
        "For example: `\\boxed{your answer}`"
    )
    assert HARNESS_GUARD == expected


def test_thinking_close_is_canonical_tag() -> None:
    assert THINKING_CLOSE == "</think>"


def test_expected_chat_template_sha16_matches_probe_contract() -> None:
    assert EXPECTED_CHAT_TEMPLATE_SHA16 == "ab7813c3abdd9cb6"


def test_build_user_content_appends_guard_verbatim() -> None:
    assert build_user_content("Hello") == "Hello" + HARNESS_GUARD


def test_build_user_content_handles_empty_prompt() -> None:
    assert build_user_content("") == HARNESS_GUARD


def test_build_chat_thinking_prompt_ends_with_think_seed() -> None:
    prompt = build_chat_thinking_prompt(_make_example(), FakeNemotronTokenizer())
    assert prompt.endswith("<think>\n"), (
        "Assistant seed must end with '<think>\\n'; probe confirmed this is "
        "the Nemotron default with enable_thinking=True."
    )


def test_build_chat_thinking_prompt_embeds_role_markers() -> None:
    prompt = build_chat_thinking_prompt(_make_example(), FakeNemotronTokenizer())
    assert "<|im_start|>user" in prompt
    assert "<|im_end|>" in prompt
    assert HARNESS_GUARD in prompt


def test_build_chat_thinking_prompt_preserves_raw_prompt_content() -> None:
    prompt = build_chat_thinking_prompt(_make_example(), FakeNemotronTokenizer())
    assert "Solve this puzzle." in prompt


def test_wrap_as_thinking_closes_think_and_boxes_answer() -> None:
    assert (
        wrap_as_thinking("family=bit", "11001010")
        == "family=bit\n</think>\n\\boxed{11001010}"
    )


def test_wrap_as_thinking_empty_body_produces_empty_thinking() -> None:
    assert wrap_as_thinking("", "42") == "</think>\n\\boxed{42}"


def test_wrap_as_thinking_none_body_treated_as_empty() -> None:
    # Defensive: some callers may pass None through type-loose paths.
    assert wrap_as_thinking(None, "42") == "</think>\n\\boxed{42}"  # type: ignore[arg-type]


def test_wrap_as_thinking_strips_trailing_whitespace_from_body() -> None:
    assert (
        wrap_as_thinking("sig=xor:0x5a\n  ", "00")
        == "sig=xor:0x5a\n</think>\n\\boxed{00}"
    )


def test_wrap_as_thinking_has_exactly_one_boxed_tail() -> None:
    """extract_final_answer anchors on the last \\boxed{}; assert there is exactly one."""
    result = wrap_as_thinking("family=bit; sub=x; sig=y", "11")
    assert result.count("\\boxed{") == 1
    assert result.endswith("\\boxed{11}")


def test_wrap_as_thinking_has_exactly_one_think_close() -> None:
    result = wrap_as_thinking("some trace body", "ans")
    assert result.count("</think>") == 1


def test_fake_tokenizer_enable_thinking_false_matches_probe_output() -> None:
    """Sanity-check that the probe fixture matches both branches of apply_chat_template."""
    tok = FakeNemotronTokenizer()
    on = tok.apply_chat_template(
        [{"role": "user", "content": "HELLO"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    off = tok.apply_chat_template(
        [{"role": "user", "content": "HELLO"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert on.endswith("<think>\n")
    assert off.endswith("<think></think>")
    assert on != off


def test_probe_artifact_confirms_first_public_sample_budget() -> None:
    import json
    from pathlib import Path

    probe_path = Path("artifacts/chat_template_probe.json")
    assert probe_path.exists(), "probe artifact should be checked into the repo"

    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    assert payload["chat_template_sha16"] == EXPECTED_CHAT_TEMPLATE_SHA16
    assert payload["conclusions"]["first_public_sample_fits_budget"] is True
