from __future__ import annotations

from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair
from src.teacher.family_tagger import apply_family_tags


def _equation_example(pairs: list[tuple[str, str]], query: str = "?") -> PuzzleExample:
    return PuzzleExample(
        id="eq",
        raw_prompt="",
        official_instruction="",
        parsed_examples=[PuzzlePair(input=input_text, output=output_text) for input_text, output_text in pairs],
        query=query,
        target_answer=None,
        metadata=PuzzleMetadata(official_family="equation", source="test", split="train"),
    )


def test_equation_literal_outputs_are_template_not_position() -> None:
    example = _equation_example(
        [
            ("#/-\\@", "-@#"),
            ("'#+/#", "%\""),
        ],
        query="'/-%)",
    )

    [tagged] = apply_family_tags([example])

    assert tagged.metadata.subtype == "equation_template"


def test_equation_reordered_input_symbols_remain_position() -> None:
    example = _equation_example(
        [
            ("abcde", "eca"),
            ("fghij", "igf"),
        ],
        query="klmno",
    )

    [tagged] = apply_family_tags([example])

    assert tagged.metadata.subtype == "equation_position"


def test_equation_repeated_output_symbol_is_template() -> None:
    example = _equation_example(
        [
            ("a*cde", "aaa"),
            ("f+hij", "fff"),
        ],
        query="k-lmo",
    )

    [tagged] = apply_family_tags([example])

    assert tagged.metadata.subtype == "equation_template"
