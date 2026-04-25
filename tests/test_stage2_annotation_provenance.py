from __future__ import annotations

from pathlib import Path

from src.teacher.stage2_annotation_provenance import (
    build_stage2_subset_provenance,
    sha256_file,
    stage2_provenance_matches_local,
)


def test_subset_provenance_records_selection_and_hashes(tmp_path: Path) -> None:
    source = tmp_path / "official_train_tagged.jsonl"
    source.write_text('{"id":"a"}\n', encoding="utf-8")
    output = tmp_path / "stage2_train.jsonl"
    output.write_text('{"id":"a"}\n', encoding="utf-8")
    splits = tmp_path / "splits.json"
    splits.write_text('{"rule_novelty_all":{"train_ids":["a"]}}\n', encoding="utf-8")

    provenance = build_stage2_subset_provenance(
        input_path=source,
        output_path=output,
        split_file=splits,
        selection={
            "split_name": "rule_novelty_all",
            "split_role": "train",
            "exclude_split_name": "hard_triad_rule_novelty",
            "exclude_split_role": "valid",
        },
    )

    assert provenance["provenance_type"] == "stage2_teacher_subset"
    assert provenance["input_jsonl_sha256"] == sha256_file(source)
    assert provenance["split_file_sha256"] == sha256_file(splits)
    assert provenance["output_jsonl_sha256"] == sha256_file(output)
    assert provenance["selection"] == {
        "split_name": "rule_novelty_all",
        "split_role": "train",
        "exclude_split_name": "hard_triad_rule_novelty",
        "exclude_split_role": "valid",
    }


def test_stage2_provenance_matches_local_checks_output_hash(tmp_path: Path) -> None:
    source = tmp_path / "official_train_tagged.jsonl"
    source.write_text('{"id":"a"}\n', encoding="utf-8")
    output = tmp_path / "stage2_train.jsonl"
    output.write_text('{"id":"a"}\n', encoding="utf-8")
    provenance = build_stage2_subset_provenance(
        input_path=source,
        output_path=output,
        split_file=tmp_path / "missing_splits.json",
        selection={"split_name": "rule_novelty_all", "split_role": "train"},
    )
    ok, _required, _found = stage2_provenance_matches_local(
        provenance,
        output_path=output,
    )
    assert ok is True

    output.write_text('{"id":"changed"}\n', encoding="utf-8")
    ok, required, found = stage2_provenance_matches_local(
        provenance,
        output_path=output,
    )
    assert ok is False
    assert required["output_jsonl_sha256"] != found["output_jsonl_sha256"]
