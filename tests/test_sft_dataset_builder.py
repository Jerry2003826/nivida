from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import src.student.sft_dataset_builder as builder_module
from src.common.io import load_jsonl, write_json, write_jsonl
from src.competition.prompt_templates import PROMPT_MODE_CHAT_THINKING
from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair
from src.student.sft_dataset_builder import (
    PROMPT_MODE_GENERIC,
    RepairArtifactSchemaError,
    build_repair_set,
    build_selected_sft,
    build_sft_record,
    build_stage1_sft,
    export_split_subset,
    summarise_repair_sft,
    summarise_selected_sft,
)
from tests.test_harness_prompt import FakeNemotronTokenizer


def _make_example(
    example_id: str,
    *,
    source: str,
    signature: str,
    official_family: str = "cipher",
    subtype: str = "cipher_token_sub",
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
            official_family=official_family,
            subtype=subtype,
            teacher_confidence=confidence,
            program_signature=signature,
            source=source,
            split="train",
            extras={"support_coverage": 1.0, "top1_top2_margin": margin, "solver_verifiable": True},
        ),
    )


def test_build_sft_record_supports_three_trace_styles() -> None:
    example = _make_example("base", source="official", signature="vocab_sub")
    assert build_sft_record(example, stage="stage1", trace_style="answer_only")["completion"] == "\\boxed{seven eight}"
    assert "family=cipher" in build_sft_record(example, stage="stage2", trace_style="short_trace")["completion"]
    assert "fam=cipher" in build_sft_record(example, stage="stage2", trace_style="token_trace")["completion"]


def test_build_sft_record_supports_prompt_modes_and_metadata_fallback() -> None:
    example = _make_example("prompt", source="official", signature="vocab_sub")
    example.metadata.extras = {"source_dataset": "official_selected"}
    generic = build_sft_record(example, stage="stage1", prompt_mode=PROMPT_MODE_GENERIC)
    assert generic["prompt_mode"] == "generic"
    assert generic["source_dataset"] == "official_selected"


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
        trace_style="token_trace",
    )
    assert [row["id"] for row in dataset] == ["official", "synth_keep"]
    assert dataset[0]["trace_style"] == "token_trace"


def test_build_selected_sft_balances_families_round_robin() -> None:
    dataset = build_selected_sft(
        [
            _make_example("cipher_1", source="official", signature="sig-c1", official_family="cipher", subtype="cipher_token_sub"),
            _make_example("cipher_2", source="official", signature="sig-c2", official_family="cipher", subtype="cipher_vocab"),
            _make_example("bit_1", source="official", signature="sig-b1", official_family="bit", subtype="bit_xor_mask"),
            _make_example("bit_2", source="official", signature="sig-b2", official_family="bit", subtype="bit_shift"),
            _make_example("numeral_1", source="official", signature="sig-n1", official_family="numeral", subtype="numeral_base"),
            _make_example("numeral_2", source="official", signature="sig-n2", official_family="numeral", subtype="numeral_base"),
        ],
        hard_triad_repeat_factor=1,
    )
    assert [row["id"] for row in dataset] == [
        "cipher_1",
        "bit_1",
        "numeral_1",
        "cipher_2",
        "bit_2",
        "numeral_2",
    ]


def test_build_selected_sft_enforces_signature_bucket_limit() -> None:
    dataset = build_selected_sft(
        [
            _make_example("cipher_1", source="official", signature="dup-sig", official_family="cipher"),
            _make_example("cipher_2", source="official", signature="dup-sig", official_family="cipher"),
            _make_example("bit_1", source="official", signature="bit-sig", official_family="bit", subtype="bit_xor_mask"),
        ],
        hard_triad_repeat_factor=1,
        max_per_signature_bucket=1,
    )
    assert [row["id"] for row in dataset] == ["cipher_1", "bit_1"]
    report = summarise_selected_sft(dataset)
    assert report["family_counts"] == {"bit": 1, "cipher": 1}
    assert report["hard_triad_ratio"] == 1.0


def test_build_selected_sft_passes_search_parameters_to_annotation(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, int] = {}

    def _fake_annotate(examples, *, beam_width, max_depth, top_k):
        seen.update({"beam_width": beam_width, "max_depth": max_depth, "top_k": top_k})
        return examples

    monkeypatch.setattr(builder_module, "_annotate_examples", _fake_annotate)
    build_selected_sft(
        [_make_example("official", source="official", signature="sig-official")],
        beam_width=10,
        max_depth=3,
        top_k=3,
    )

    assert seen == {"beam_width": 10, "max_depth": 3, "top_k": 3}


def test_build_repair_set_retains_error_fields(tmp_path: Path) -> None:
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
                    "predicted_signature": "wrong_sig",
                }
            ]
        },
    )
    dataset = build_repair_set(examples, repair_artifact=artifact, trace_style="short_trace")
    assert dataset[0]["error_type"] == "format_error"
    assert dataset[0]["predicted_signature"] == "wrong_sig"
    assert dataset[0]["repair_source"] == "repair"


def test_stage_datasets_roundtrip_with_jsonl_inputs(tmp_path: Path) -> None:
    input_path = tmp_path / "examples.jsonl"
    write_jsonl(input_path, [_make_example("official", source="official", signature="sig-official").to_dict()])
    rows = [PuzzleExample.from_dict(row) for row in load_jsonl(input_path)]
    assert rows[0].metadata.official_family == "cipher"


def test_export_split_subset_returns_filtered_examples(tmp_path: Path) -> None:
    split_path = tmp_path / "splits.json"
    write_json(
        split_path,
        {
            "hard_triad_rule_novelty": {
                "train_ids": ["keep-me"],
                "valid_ids": [],
            }
        },
    )
    rows = export_split_subset(
        [
            _make_example("keep-me", source="official", signature="sig-1"),
            _make_example("drop-me", source="official", signature="sig-2"),
        ],
        split_file=split_path,
        split_name="hard_triad_rule_novelty",
        split_role="train",
    )
    assert [row["id"] for row in rows] == ["keep-me"]


# --- Repair artifact schema guard ---------------------------------------------


def _repair_example() -> PuzzleExample:
    return _make_example("repair_me", source="official", signature="sig-repair")


def _write_artifact(path: Path, payload: object) -> Path:
    write_json(path, payload)
    return path


def test_build_repair_set_accepts_canonical_schema(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path / "baseline_eval.json",
        {
            "records": [
                {
                    "id": "repair_me",
                    "competition_correct": False,
                    "boxed_valid": False,
                    "predicted_signature": "wrong_sig",
                }
            ]
        },
    )
    dataset = build_repair_set([_repair_example()], repair_artifact=artifact)
    assert [row["id"] for row in dataset] == ["repair_me"]


def test_build_repair_set_legacy_rows_field_warns_but_works(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path / "baseline_eval.json",
        {
            "rows": [
                {
                    "id": "repair_me",
                    "competition_correct": False,
                    "boxed_valid": False,
                }
            ]
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dataset = build_repair_set([_repair_example()], repair_artifact=artifact)
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    assert [row["id"] for row in dataset] == ["repair_me"]


def test_build_repair_set_raises_when_competition_correct_missing(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path / "baseline_eval.json",
        {"records": [{"id": "repair_me", "prediction": "\\boxed{x}"}]},
    )
    with pytest.raises(RepairArtifactSchemaError, match="competition_correct"):
        build_repair_set([_repair_example()], repair_artifact=artifact)


def test_build_repair_set_raises_when_id_missing(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path / "baseline_eval.json",
        {"records": [{"competition_correct": False, "prediction": ""}]},
    )
    with pytest.raises(RepairArtifactSchemaError, match="'id'"):
        build_repair_set([_repair_example()], repair_artifact=artifact)


def test_build_repair_set_raises_on_missing_records_and_rows(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path / "baseline_eval.json", {"headline_metric": "competition_correct_rate"}
    )
    with pytest.raises(RepairArtifactSchemaError, match="missing both 'records' and 'rows'"):
        build_repair_set([_repair_example()], repair_artifact=artifact)


def test_build_repair_set_raises_on_empty_records(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "baseline_eval.json", {"records": []})
    with pytest.raises(RepairArtifactSchemaError, match="empty records list"):
        build_repair_set([_repair_example()], repair_artifact=artifact)


def test_build_repair_set_raises_when_records_not_list(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path / "baseline_eval.json", {"records": {"id": "x", "competition_correct": False}}
    )
    with pytest.raises(RepairArtifactSchemaError, match="non-list records"):
        build_repair_set([_repair_example()], repair_artifact=artifact)


def test_build_repair_set_raises_when_payload_not_dict(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path / "baseline_eval.json",
        [{"id": "repair_me", "competition_correct": False}],
    )
    with pytest.raises(RepairArtifactSchemaError, match="not a JSON object"):
        build_repair_set([_repair_example()], repair_artifact=artifact)


# --- chat_thinking prompt mode -----------------------------------------------


def test_build_sft_record_chat_thinking_requires_tokenizer() -> None:
    example = _make_example("ex", source="official", signature="sig")
    with pytest.raises(ValueError, match="chat_thinking"):
        build_sft_record(
            example,
            stage="stage1",
            prompt_mode=PROMPT_MODE_CHAT_THINKING,
            trace_style="answer_only",
        )


def test_build_sft_record_chat_thinking_wraps_prompt_and_completion() -> None:
    example = _make_example("ex", source="official", signature="sig-x")
    tokenizer = FakeNemotronTokenizer()
    record = build_sft_record(
        example,
        stage="stage2",
        prompt_mode=PROMPT_MODE_CHAT_THINKING,
        trace_style="short_trace",
        tokenizer=tokenizer,
    )
    assert record["prompt"].endswith("<think>\n")
    assert "Please put your final answer inside `\\boxed{}`" in record["prompt"]
    # Completion must close the thinking segment and box the answer exactly once
    assert "</think>" in record["completion"]
    assert record["completion"].count("\\boxed{") == 1
    assert record["completion"].endswith("\\boxed{seven eight}")
    # Trace body is preserved inside the thinking segment
    assert "family=cipher" in record["completion"]
    assert record["source_prompt_type"] == "chat_thinking"


def test_build_sft_record_chat_thinking_answer_only_produces_empty_thinking() -> None:
    example = _make_example("ex", source="official", signature="sig-x")
    record = build_sft_record(
        example,
        stage="stage1",
        prompt_mode=PROMPT_MODE_CHAT_THINKING,
        trace_style="answer_only",
        tokenizer=FakeNemotronTokenizer(),
    )
    assert record["completion"] == "</think>\n\\boxed{seven eight}"


def test_build_stage1_sft_passes_tokenizer_through() -> None:
    examples = [_make_example("official", source="official", signature="sig")]
    dataset = build_stage1_sft(
        examples,
        prompt_mode=PROMPT_MODE_CHAT_THINKING,
        tokenizer=FakeNemotronTokenizer(),
    )
    assert dataset[0]["prompt"].endswith("<think>\n")
    assert dataset[0]["completion"].endswith("\\boxed{seven eight}")


def test_build_selected_sft_passes_tokenizer_through() -> None:
    dataset = build_selected_sft(
        [_make_example("official", source="official", signature="sig-official")],
        prompt_mode=PROMPT_MODE_CHAT_THINKING,
        trace_style="token_trace",
        tokenizer=FakeNemotronTokenizer(),
    )
    assert dataset[0]["prompt"].endswith("<think>\n")
    assert "fam=cipher" in dataset[0]["completion"]


def test_build_repair_set_passes_tokenizer_through(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path / "baseline_eval.json",
        {
            "records": [
                {
                    "id": "repair_me",
                    "competition_correct": False,
                    "boxed_valid": False,
                }
            ]
        },
    )
    dataset = build_repair_set(
        [_repair_example()],
        repair_artifact=artifact,
        prompt_mode=PROMPT_MODE_CHAT_THINKING,
        trace_style="short_trace",
        tokenizer=FakeNemotronTokenizer(),
    )
    assert dataset[0]["prompt"].endswith("<think>\n")
    assert dataset[0]["completion"].endswith("\\boxed{seven eight}")


def test_build_repair_set_adds_replay_records_from_success_artifact(tmp_path: Path) -> None:
    repair_artifact = _write_artifact(
        tmp_path / "stage2_model_failures.json",
        {
            "records": [
                {"id": "repair_me", "competition_correct": False, "boxed_valid": False},
                {"id": "repair_me_2", "competition_correct": False, "boxed_valid": False},
            ]
        },
    )
    replay_artifact = _write_artifact(
        tmp_path / "stage2_model_successes.json",
        {
            "records": [
                {"id": "replay_me", "competition_correct": True, "boxed_valid": True},
                {"id": "replay_me_2", "competition_correct": True, "boxed_valid": True},
            ]
        },
    )
    examples = [
        _make_example("repair_me", source="official", signature="sig-r1", official_family="cipher"),
        _make_example("repair_me_2", source="official", signature="sig-r2", official_family="bit", subtype="bit_xor_mask"),
        _make_example("replay_me", source="official", signature="sig-s1", official_family="cipher"),
        _make_example("replay_me_2", source="official", signature="sig-s2", official_family="bit", subtype="bit_shift"),
    ]

    dataset = build_repair_set(
        examples,
        repair_artifact=repair_artifact,
        replay_input=replay_artifact,
        replay_ratio=0.5,
    )

    assert len(dataset) == 3
    assert [row["repair_source"] for row in dataset] == ["repair", "repair", "replay"]
    report = summarise_repair_sft(dataset)
    assert report["repair_count"] == 2
    assert report["replay_count"] == 1
