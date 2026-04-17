from __future__ import annotations

from src.competition.prompt_templates import build_competition_prompt
from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair


def test_build_competition_prompt_preserves_raw_prompt_and_appends_contract() -> None:
    prompt = "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.\nNow, determine the output for: 00110100"
    example = PuzzleExample(
        id="raw",
        raw_prompt=prompt,
        official_instruction="In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.",
        parsed_examples=[],
        query="00110100",
        metadata=PuzzleMetadata(official_family="bit", subtype="rotate"),
    )
    rendered = build_competition_prompt(example)
    assert prompt in rendered
    assert rendered.endswith(r"Return exactly one final answer as \boxed{...}.")


def test_build_competition_prompt_builds_synthetic_official_style_prompt() -> None:
    example = PuzzleExample(
        id="synth",
        raw_prompt="",
        official_instruction="In Alice's Wonderland, a secret unit conversion is applied to measurements.",
        parsed_examples=[
            PuzzlePair(input="10.08 m", output="6.69"),
            PuzzlePair(input="17.83 m", output="11.83"),
        ],
        query="25.09 m",
        metadata=PuzzleMetadata(official_family="unit", subtype="scale"),
    )
    rendered = build_competition_prompt(example)
    assert "secret unit conversion is applied to measurements" in rendered
    assert "10.08 m becomes 6.69" in rendered
    assert "Now, convert the following measurement: 25.09 m" in rendered
