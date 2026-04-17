from __future__ import annotations

from src.competition.prompt_templates import (
    PROMPT_MODE_GENERIC,
    build_competition_prompt,
    build_generic_prompt,
    build_raw_prompt_with_guard,
)
from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair


def test_raw_prompt_is_preserved_and_guard_is_appended_once() -> None:
    prompt = "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.\nNow, determine the output for: 00110100"
    example = PuzzleExample(
        id="raw",
        raw_prompt=prompt,
        official_instruction="In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.",
        parsed_examples=[],
        query="00110100",
        metadata=PuzzleMetadata(official_family="bit", subtype="bit_rotate"),
    )
    rendered = build_raw_prompt_with_guard(example)
    assert rendered.startswith(prompt)
    assert rendered.count(r"Return exactly one final answer as \boxed{...}.") == 1


def test_existing_output_contract_is_not_duplicated() -> None:
    prompt = "Solve it.\nReturn exactly one final answer as \\boxed{...}."
    example = PuzzleExample(
        id="guarded",
        raw_prompt=prompt,
        official_instruction="Solve it.",
        parsed_examples=[],
        query="x",
    )
    rendered = build_raw_prompt_with_guard(example)
    assert rendered == prompt


def test_generic_mode_still_works() -> None:
    example = PuzzleExample(
        id="synth",
        raw_prompt="",
        official_instruction="In Alice's Wonderland, a secret unit conversion is applied to measurements.",
        parsed_examples=[
            PuzzlePair(input="10.08 m", output="6.69"),
            PuzzlePair(input="17.83 m", output="11.83"),
        ],
        query="25.09 m",
        metadata=PuzzleMetadata(official_family="unit", subtype="unit_scale"),
    )
    rendered = build_generic_prompt(example)
    assert "secret unit conversion is applied to measurements" in rendered
    assert "10.08 m becomes 6.69" in rendered
    assert "Now, convert the following measurement: 25.09 m" in rendered


def test_raw_prompt_missing_falls_back_to_generic() -> None:
    example = PuzzleExample(
        id="fallback",
        raw_prompt="",
        official_instruction="In Alice's Wonderland, secret encryption rules are used on text.",
        parsed_examples=[PuzzlePair(input="a", output="b")],
        query="c",
        metadata=PuzzleMetadata(official_family="cipher", subtype="cipher_char_sub"),
    )
    rendered = build_competition_prompt(example, mode=PROMPT_MODE_GENERIC)
    assert "secret encryption rules are used on text" in rendered
    assert r"\boxed{...}" in rendered
