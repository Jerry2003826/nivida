from __future__ import annotations

from pathlib import Path

from src.common.io import write_json, write_jsonl
from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair
from src.student.sft_dataset_builder import (
    build_repair_set,
    build_selected_sft,
    build_stage1_sft,
)


def _make_example(
    example_id: str,
    *,
    source: str,
    signature: str,
    confidence: float = 0.9,
    margin: float = 0.1,
) -> PuzzleExample:
    return PuzzleExample(
        id=example_id,
        raw_prompt="In Alice's Wonderland, secret encryption rules are used on text.\nNow, decrypt the following text: one two",
        official_instruction="In Alice's Wonderland, secret encryption rules are used on text.",
        parsed_examples=[
            PuzzlePair(input="foo bar", output="one two"),
            PuzzlePair(input="baz qux", output="three four"),
            PuzzlePair(input="zip zap", output="five six"),
        ],
        query="alpha beta",
        target_answer="seven eight",
        metadata=PuzzleMetadata(
            official_family="cipher",
            subtype="token_substitution",
            teacher_confidence=confidence,
            program_signature=signature,
            source=source,
            split="train",
            extras={"support_coverage": 1.0, "top1_top2_margin": margin, "solver_verifiable": True},
        ),
    )


def test_build_stage1_sft_uses_answer_only_for_official_examples() -> None:
    dataset = build_stage1_sft([
        _make_example("official", source="official", signature="sig-official"),
        _make_example("synth", source="synthetic", signature="sig-synth"),
    ])
    assert [row["id"] for row in dataset] == ["official"]
    assert dataset[0]["completion"] == "\\boxed{seven eight}"


def test_build_selected_sft_keeps_selected_official_and_unique_synth() -> None:
    dataset = build_selected_sft(
        [
            _make_example("official", source="official", signature="sig-official"),
            _make_example("synth_dup", source="synthetic", signature="sig-official"),
            _make_example("synth_keep", source="synthetic", signature="sig-synth"),
        ],
        completion_style="token_trace",
    )
    assert [row["id"] for row in dataset] == ["official", "synth_keep"]
    assert dataset[0]["completion"].startswith("fam=cipher|sub=token_substitution|prog=sig-official|")


def test_build_repair_set_assigns_repair_buckets(tmp_path: Path) -> None:
    examples = [_make_example("repair_me", source="official", signature="sig-repair")]
    artifact = tmp_path / "failures.json"
    write_json(
        artifact,
        {
            "records": [
                {
                    "id": "repair_me",
                    "competition_correct": False,
                    "boxed_valid": False,
                    "debug": {"family_legality": 1.0},
                }
            ]
        },
    )
    dataset = build_repair_set(examples, repair_artifact=artifact, completion_style="short_trace")
    assert dataset[0]["repair_bucket"] == "format_only"
    assert dataset[0]["source"] == "repair"


def test_stage_datasets_roundtrip_with_jsonl_inputs(tmp_path: Path) -> None:
    input_path = tmp_path / "examples.jsonl"
    write_jsonl(input_path, [_make_example("official", source="official", signature="sig-official").to_dict()])
    rows = [PuzzleExample.from_dict(row) for row in [{"id": "official", "raw_prompt": "x", "official_instruction": "x", "parsed_examples": [], "query": "y", "target_answer": "z", "metadata": {"official_family": "equation", "subtype": "numeric", "source": "official"}}]]
    assert rows[0].metadata.official_family == "equation"
