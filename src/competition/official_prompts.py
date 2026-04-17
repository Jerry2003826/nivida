from __future__ import annotations

import re

from src.competition.schema import PuzzleExample


ANSWER_CONTRACT = r"Return exactly one final answer as \boxed{...}."

OFFICIAL_FAMILY_INSTRUCTIONS = {
    "bit": "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.",
    "gravity": "In Alice's Wonderland, the gravitational constant has been secretly changed.",
    "unit": "In Alice's Wonderland, a secret unit conversion is applied to measurements.",
    "cipher": "In Alice's Wonderland, secret encryption rules are used on text.",
    "numeral": "In Alice's Wonderland, numbers are secretly converted into a different numeral system.",
    "equation": "In Alice's Wonderland, a secret set of transformation rules is applied to equations.",
}

_OFFICIAL_PROMPT_PATTERNS = {
    re.compile(r"bit manipulation rule transforms 8-bit binary numbers", flags=re.IGNORECASE): "bit",
    re.compile(r"gravitational constant has been secretly changed", flags=re.IGNORECASE): "gravity",
    re.compile(r"secret unit conversion is applied to measurements", flags=re.IGNORECASE): "unit",
    re.compile(r"secret encryption rules are used on text", flags=re.IGNORECASE): "cipher",
    re.compile(r"numbers are secretly converted into a different numeral system", flags=re.IGNORECASE): "numeral",
    re.compile(r"secret set of transformation rules is applied to equations", flags=re.IGNORECASE): "equation",
}


def detect_official_family(prompt: str) -> str | None:
    for pattern, family in _OFFICIAL_PROMPT_PATTERNS.items():
        if pattern.search(prompt):
            return family
    return None


def extract_official_instruction(prompt: str, *, family: str | None = None) -> str:
    stripped = prompt.strip()
    if stripped:
        first_block = stripped.split("\n\n", 1)[0].strip()
        if first_block:
            return first_block
        first_line = stripped.splitlines()[0].strip()
        if first_line:
            return first_line
    if family and family in OFFICIAL_FAMILY_INSTRUCTIONS:
        return OFFICIAL_FAMILY_INSTRUCTIONS[family]
    detected = detect_official_family(prompt)
    if detected:
        return OFFICIAL_FAMILY_INSTRUCTIONS[detected]
    return ""


def ensure_answer_contract(prompt: str) -> str:
    stripped = prompt.strip()
    if not stripped:
        return ANSWER_CONTRACT
    if ANSWER_CONTRACT in stripped:
        return stripped
    return f"{stripped}\n{ANSWER_CONTRACT}"


def build_official_style_prompt(example: PuzzleExample) -> str:
    family = example.metadata.official_family or detect_official_family(example.raw_prompt or "") or "equation"
    instruction = example.official_instruction or OFFICIAL_FAMILY_INSTRUCTIONS.get(family, "")

    blocks: list[str] = [instruction]
    if family == "gravity":
        blocks.append("Here are some example observations:")
        for pair in example.parsed_examples:
            blocks.append(f"For t = {pair.input}s, distance = {pair.output} m")
        blocks.append(f"Now, determine the falling distance for t = {example.query}s given d = 0.5*g*t^2.")
    elif family == "unit":
        blocks.append("For example:")
        for pair in example.parsed_examples:
            blocks.append(f"{pair.input} becomes {pair.output}")
        blocks.append(f"Now, convert the following measurement: {example.query}")
    elif family == "cipher":
        blocks.append("Here are some examples:")
        for pair in example.parsed_examples:
            blocks.append(f"{pair.input} -> {pair.output}")
        blocks.append(f"Now, decrypt the following text: {example.query}")
    elif family == "numeral":
        blocks.append("Here are some examples:")
        for pair in example.parsed_examples:
            blocks.append(f"{pair.input} -> {pair.output}")
        blocks.append(f"Now, write the number {example.query} in the Wonderland numeral system.")
    else:
        blocks.append("Here are some examples of input -> output:")
        for pair in example.parsed_examples:
            blocks.append(f"{pair.input} -> {pair.output}")
        blocks.append(f"Now, determine the output for: {example.query}")
    return ensure_answer_contract("\n".join(blocks))
