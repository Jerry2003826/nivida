"""Verify ``build_repair_set`` restricts chain-search annotation to target ids.

The stage3 train path now feeds the builder the full ``official_train_tagged``
pool (~10k samples) so replay can span all families. Running the CPU-heavy
ChainSearchEngine over all of them would be wasted work because only the
failure and success ids actually become training records. This test pins down
that the annotation step receives exactly the ids referenced by the repair and
replay artifacts, not the whole input pool.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.io import write_json
from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair
from src.student import sft_dataset_builder


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
            teacher_confidence=1.0,
            program_signature="sig-1",
            source=source,
            split="train",
            extras={
                "solver_verifiable": True,
                "support_coverage": 1.0,
                "top1_top2_margin": 0.5,
            },
        ),
    )


def test_annotation_runs_only_on_failure_and_success_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    examples = [_example(f"id-{index}") for index in range(10)]

    repair_artifact = tmp_path / "failures.json"
    write_json(
        repair_artifact,
        {
            "records": [
                {"id": "id-0", "competition_correct": False},
                {"id": "id-1", "competition_correct": False},
            ]
        },
    )
    replay_artifact = tmp_path / "successes.json"
    write_json(
        replay_artifact,
        {
            "records": [
                {"id": "id-5", "competition_correct": True},
            ]
        },
    )

    seen_ids: list[set[str]] = []

    def _fake_annotate(
        examples: list[PuzzleExample],
        *,
        beam_width: int,
        max_depth: int,
        top_k: int,
    ) -> list[PuzzleExample]:
        seen_ids.append({example.id for example in examples})
        return examples

    monkeypatch.setattr(sft_dataset_builder, "_annotate_examples", _fake_annotate)

    records = sft_dataset_builder.build_repair_set(
        examples,
        repair_artifact=repair_artifact,
        replay_input=replay_artifact,
        replay_ratio=0.25,
    )

    assert seen_ids == [{"id-0", "id-1", "id-5"}]
    record_ids = {record["id"] for record in records}
    assert {"id-0", "id-1"}.issubset(record_ids)


def test_annotation_skips_replay_when_ratio_zero(tmp_path: Path, monkeypatch) -> None:
    examples = [_example(f"id-{index}") for index in range(5)]
    repair_artifact = tmp_path / "failures.json"
    write_json(
        repair_artifact,
        {
            "records": [
                {"id": "id-0", "competition_correct": False},
            ]
        },
    )
    replay_artifact = tmp_path / "successes.json"
    write_json(
        replay_artifact,
        {
            "records": [
                {"id": "id-2", "competition_correct": True},
            ]
        },
    )

    seen_ids: list[set[str]] = []

    def _fake_annotate(examples_in, **_: Any) -> list[PuzzleExample]:
        seen_ids.append({example.id for example in examples_in})
        return examples_in

    monkeypatch.setattr(sft_dataset_builder, "_annotate_examples", _fake_annotate)

    sft_dataset_builder.build_repair_set(
        examples,
        repair_artifact=repair_artifact,
        replay_input=replay_artifact,
        replay_ratio=0.0,  # disables replay
    )
    # With replay disabled the success id is never annotated.
    assert seen_ids == [{"id-0"}]
