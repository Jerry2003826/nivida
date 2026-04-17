from __future__ import annotations

from pathlib import Path

from src.competition.parser import parse_competition_file, parse_row


def test_parse_prompt_only_row() -> None:
    row = {
        "id": "demo",
        "prompt": "Input: abc\nOutput: cba\nInput: xyz\nOutput: zyx\nQuery: lamp",
        "answer": "pmal",
    }
    example = parse_row(row, source="kaggle", split="train", row_index=0)
    assert example.id == "demo"
    assert len(example.parsed_examples) == 2
    assert example.parsed_examples[0].input == "abc"
    assert example.parsed_examples[1].output == "zyx"
    assert example.query == "lamp"
    assert example.target_answer == "pmal"


def test_parse_fixture_csv() -> None:
    fixture_path = Path("tests/fixtures/sample_competition.csv")
    examples = parse_competition_file(fixture_path, source="kaggle", split="train")
    assert len(examples) == 5
    assert examples[0].query == "lamp"
    assert examples[-1].target_answer == "pon"


def test_parse_official_arrow_prompt_format() -> None:
    row = {
        "id": "official_like",
        "prompt": """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.

Here are some examples of input -> output:
01010001 -> 11011101
00001001 -> 01101101
11111111 -> 10000001

Now, determine the output for: 00110100""",
    }
    example = parse_row(row, source="kaggle", split="test", row_index=0)
    assert example.metadata.official_family == "bit"
    assert example.official_instruction == "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers."
    assert len(example.parsed_examples) == 3
    assert example.parsed_examples[0].input == "01010001"
    assert example.parsed_examples[0].output == "11011101"
    assert example.query == "00110100"


def test_parse_official_gravity_prompt_format() -> None:
    row = {
        "id": "gravity_like",
        "prompt": """In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:
For t = 1.37s, distance = 14.92 m
For t = 4.27s, distance = 144.96 m
For t = 3.28s, distance = 85.54 m
Now, determine the falling distance for t = 4.41s given d = 0.5*g*t^2.""",
        "answer": "154.62",
    }
    example = parse_row(row, source="kaggle", split="train", row_index=0)
    assert example.metadata.official_family == "gravity"
    assert len(example.parsed_examples) == 3
    assert example.parsed_examples[0].input == "1.37"
    assert example.parsed_examples[0].output == "14.92"
    assert example.query == "4.41"


def test_parse_official_unit_and_decrypt_queries() -> None:
    unit_row = {
        "id": "unit_like",
        "prompt": """In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:
10.08 m becomes 6.69
17.83 m becomes 11.83
Now, convert the following measurement: 25.09 m""",
    }
    unit_example = parse_row(unit_row, source="kaggle", split="test", row_index=0)
    assert len(unit_example.parsed_examples) == 2
    assert unit_example.metadata.official_family == "unit"
    assert unit_example.parsed_examples[0].input == "10.08 m"
    assert unit_example.parsed_examples[0].output == "6.69"
    assert unit_example.query == "25.09 m"

    decrypt_row = {
        "id": "cipher_like",
        "prompt": """In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley
pqrsfv pqorzg wvgwpo trgbjo -> dragon dreams inside castle
Now, decrypt the following text: trb wzrswvog hffk""",
    }
    decrypt_example = parse_row(decrypt_row, source="kaggle", split="test", row_index=1)
    assert decrypt_example.metadata.official_family == "cipher"
    assert len(decrypt_example.parsed_examples) == 2
    assert decrypt_example.query == "trb wzrswvog hffk"
