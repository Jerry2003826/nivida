"""Tests for ``filter_examples_by_split`` / ``export_split_subset`` exclusion.

The two canonical splits ``rule_novelty_all`` and ``hard_triad_rule_novelty``
are built independently in :mod:`src.competition.split_builder` (different
seeds, different item pools), so a hard-triad item can end up simultaneously
in ``rule_novelty_all.train`` and ``hard_triad_rule_novelty.valid``. Callers
that treat one split as nested in the other have to compose the pair via the
``exclude_split_*`` parameters to avoid leaking valid ids into train.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io import write_json
from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair
from src.student.sft_dataset_builder import (
    export_split_subset,
    filter_examples_by_split,
)


def _example(example_id: str, *, source: str = "official") -> PuzzleExample:
    return PuzzleExample(
        id=example_id,
        raw_prompt="raw",
        official_instruction="",
        parsed_examples=[PuzzlePair(input="i", output="o")],
        query="q",
        target_answer="a",
        metadata=PuzzleMetadata(
            official_family="bit",
            subtype="bit_xor_mask",
            source=source,
            split="train",
        ),
    )


def _write_splits(path: Path) -> Path:
    write_json(
        path,
        {
            "rule_novelty_all": {
                "train_ids": ["a", "b"],
                "valid_ids": ["c"],
            },
            "hard_triad_rule_novelty": {
                "train_ids": ["a"],
                "valid_ids": ["b", "d"],
            },
        },
    )
    return path


def test_exclude_split_removes_overlap(tmp_path: Path) -> None:
    splits = _write_splits(tmp_path / "splits.json")
    examples = [_example("a"), _example("b"), _example("c"), _example("d")]

    filtered = filter_examples_by_split(
        examples,
        split_file=splits,
        split_name="rule_novelty_all",
        split_role="train",
        exclude_split_file=splits,
        exclude_split_name="hard_triad_rule_novelty",
        exclude_split_role="valid",
    )
    assert [example.id for example in filtered] == ["a"]


def test_exclude_split_is_no_op_when_disjoint(tmp_path: Path) -> None:
    splits = _write_splits(tmp_path / "splits.json")
    examples = [_example("a"), _example("b"), _example("c"), _example("d")]

    filtered = filter_examples_by_split(
        examples,
        split_file=splits,
        split_name="rule_novelty_all",
        split_role="valid",
        exclude_split_file=splits,
        exclude_split_name="hard_triad_rule_novelty",
        exclude_split_role="train",
    )
    assert [example.id for example in filtered] == ["c"]


def test_exclude_split_requires_all_fields(tmp_path: Path) -> None:
    splits = _write_splits(tmp_path / "splits.json")

    with pytest.raises(ValueError, match="must be provided together"):
        filter_examples_by_split(
            [_example("a")],
            split_file=splits,
            split_name="rule_novelty_all",
            split_role="train",
            exclude_split_file=splits,
        )


def test_synth_kept_for_train_even_with_exclude(tmp_path: Path) -> None:
    splits = _write_splits(tmp_path / "splits.json")
    examples = [
        _example("a"),
        _example("b"),
        _example("synthetic_1", source="synthetic"),
    ]

    filtered = filter_examples_by_split(
        examples,
        split_file=splits,
        split_name="rule_novelty_all",
        split_role="train",
        exclude_split_file=splits,
        exclude_split_name="hard_triad_rule_novelty",
        exclude_split_role="valid",
    )
    # Synth-source examples are kept for train role regardless of split ids...
    assert "synthetic_1" in {example.id for example in filtered}
    # ... but official b is dropped because it leaks into hard-triad valid.
    assert "b" not in {example.id for example in filtered}


def test_exclude_split_drops_synth_id_if_it_leaks(tmp_path: Path) -> None:
    """Synth rows whose id happens to match an excluded split id must be dropped."""
    splits = _write_splits(tmp_path / "splits.json")
    examples = [_example("d", source="synthetic")]

    filtered = filter_examples_by_split(
        examples,
        split_file=splits,
        split_name="rule_novelty_all",
        split_role="train",
        exclude_split_file=splits,
        exclude_split_name="hard_triad_rule_novelty",
        exclude_split_role="valid",
    )
    assert filtered == []


def test_export_split_subset_passes_exclude_through(tmp_path: Path) -> None:
    splits = _write_splits(tmp_path / "splits.json")
    examples = [_example("a"), _example("b"), _example("c"), _example("d")]

    rows = export_split_subset(
        examples,
        split_file=splits,
        split_name="rule_novelty_all",
        split_role="train",
        exclude_split_file=splits,
        exclude_split_name="hard_triad_rule_novelty",
        exclude_split_role="valid",
    )
    assert [row["id"] for row in rows] == ["a"]


def test_all_family_proxy_construction_has_no_hard_triad_train_overlap(tmp_path: Path) -> None:
    """The canonical all-family proxy subset is rule_novelty_all/valid minus
    hard_triad_rule_novelty/train. The two splits are built independently in
    split_builder, so this exclude must be applied to keep the proxy clean
    against stage3 repair which trains on hard_triad_rule_novelty/train.
    """
    splits = _write_splits(tmp_path / "splits.json")
    # rule_novelty_all.valid = {c}
    # hard_triad_rule_novelty.train = {a}
    # Expected proxy (valid role): {c}, and it must not overlap with {a}.
    examples = [_example("a"), _example("b"), _example("c"), _example("d")]

    filtered = filter_examples_by_split(
        examples,
        split_file=splits,
        split_name="rule_novelty_all",
        split_role="valid",
        exclude_split_file=splits,
        exclude_split_name="hard_triad_rule_novelty",
        exclude_split_role="train",
    )

    proxy_ids = {example.id for example in filtered}
    hard_train_ids = {"a"}
    assert proxy_ids.isdisjoint(hard_train_ids), (
        "all-family proxy must not overlap hard_triad_rule_novelty/train"
    )
    assert proxy_ids == {"c"}
