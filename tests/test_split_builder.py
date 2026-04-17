from __future__ import annotations

from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair
from src.competition.split_builder import build_splits


def _example(example_id: str, family: str, subtype: str, signature: str, bucket: str) -> PuzzleExample:
    return PuzzleExample(
        id=example_id,
        raw_prompt="raw",
        official_instruction="raw",
        parsed_examples=[PuzzlePair(input="1", output="2")],
        query="3",
        target_answer="4",
        metadata=PuzzleMetadata(
            official_family=family,
            subtype=subtype,
            teacher_confidence=0.9,
            program_signature=signature,
            source="official",
            split="train",
            extras={"solver_verifiable": True, "program_signature_bucket": bucket},
        ),
    )


def test_rule_novelty_keeps_signature_bucket_out_of_train_when_held_out() -> None:
    examples = [
        _example("a1", "bit", "bit_xor_mask", "xor:11110000", "xor"),
        _example("a2", "bit", "bit_xor_mask", "xor:00001111", "xor"),
        _example("b1", "cipher", "cipher_token_sub", "vocab_sub", "vocab_sub"),
        _example("b2", "cipher", "cipher_token_sub", "vocab_sub", "vocab_sub"),
    ]
    payload = build_splits(examples, rule_novelty_valid_ratio=0.5, hard_triad_valid_ratio=0.5, seed=7)
    split = payload["rule_novelty_all"]
    assert set(split["train_ids"]).isdisjoint(set(split["valid_ids"]))


def test_hard_triad_split_only_targets_hard_families() -> None:
    examples = [
        _example("bit1", "bit", "bit_xor_mask", "xor:11110000", "xor"),
        _example("eq1", "equation", "equation_numeric", "eq_rule", "eq_rule"),
        _example("unit1", "unit", "unit_scale", "scale:1000", "scale"),
    ]
    payload = build_splits(examples, rule_novelty_valid_ratio=0.5, hard_triad_valid_ratio=0.5, seed=11)
    hard_ids = set(payload["hard_triad_rule_novelty"]["train_ids"]) | set(payload["hard_triad_rule_novelty"]["valid_ids"])
    assert "unit1" not in hard_ids


def test_legacy_splits_are_still_present() -> None:
    examples = [_example("x1", "gravity", "gravity_inverse_square", "inverse_square", "inverse_square")]
    payload = build_splits(examples, rule_novelty_valid_ratio=1.0, hard_triad_valid_ratio=1.0, seed=3)
    assert {"iid", "family_holdout_legacy", "composition_holdout_legacy"} <= set(payload)
