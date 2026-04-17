from __future__ import annotations

from src.competition.schema import PuzzleExample


def build_competition_prompt(example: PuzzleExample) -> str:
    """Build a short instruction prompt that enforces one final boxed answer."""
    blocks: list[str] = [
        "Solve the transformation-rule reasoning puzzle.",
        "Keep reasoning brief and end with exactly one final answer in the form \\boxed{...}.",
        "",
        "Training examples:",
    ]
    for idx, pair in enumerate(example.train_pairs, start=1):
        blocks.append(f"{idx}. Input: {pair.input}")
        blocks.append(f"   Output: {pair.output}")
    blocks.extend(["", f"Query: {example.query_input}"])
    return "\n".join(blocks).strip()
