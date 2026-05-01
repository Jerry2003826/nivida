from __future__ import annotations

from pathlib import Path

from scripts.hydrate_stage2_support_cache import hydrate_support_cache
from src.common.io import load_jsonl, read_json, write_json, write_jsonl
from src.teacher.stage2_annotation_provenance import sha256_file


def test_hydrator_adds_support_pairs_and_updates_provenance(tmp_path: Path) -> None:
    source = tmp_path / "stage2.jsonl"
    output = tmp_path / "stage2_hydrated.jsonl"
    provenance = tmp_path / "stage2.jsonl.provenance.json"
    write_jsonl(
        source,
        [
            {
                "id": "reverse",
                "raw_prompt": "",
                "official_instruction": "",
                "parsed_examples": [{"input": "abc", "output": "cba"}],
                "query": "stun",
                "target_answer": "nuts",
                "metadata": {
                    "official_family": "cipher",
                    "subtype": "toy",
                    "extras": {},
                },
            }
        ],
    )
    write_json(provenance, {"output_jsonl_sha256": "stale"})

    summary = hydrate_support_cache(
        input_jsonl=source,
        output_jsonl=output,
        provenance_json=provenance,
    )

    rows = load_jsonl(output)
    extras = rows[0]["metadata"]["extras"]
    updated_provenance = read_json(provenance)
    assert summary["hydrated_rows"] == 1
    assert summary["remaining_missing_rows"] == 0
    assert summary["complete"] is True
    assert extras["support_pairs"] == [{"input": "abc", "target": "cba", "prediction": "cba"}]
    assert isinstance(extras["query_prediction"], str)
    assert updated_provenance["output_jsonl_sha256"] == sha256_file(output)
    assert updated_provenance["support_pair_cache"]["complete"] is True


def test_hydrator_limit_leaves_remaining_rows_visible(tmp_path: Path) -> None:
    source = tmp_path / "stage2.jsonl"
    output = tmp_path / "stage2_hydrated.jsonl"
    write_jsonl(
        source,
        [
            {
                "id": "a",
                "parsed_examples": [{"input": "ab", "output": "ba"}],
                "query": "cd",
                "target_answer": "dc",
                "metadata": {"official_family": "cipher", "extras": {}},
            },
            {
                "id": "b",
                "parsed_examples": [{"input": "xy", "output": "yx"}],
                "query": "pq",
                "target_answer": "qp",
                "metadata": {"official_family": "cipher", "extras": {}},
            },
        ],
    )

    summary = hydrate_support_cache(
        input_jsonl=source,
        output_jsonl=output,
        limit=1,
    )

    rows = load_jsonl(output)
    assert summary["hydrated_rows"] == 1
    assert summary["remaining_missing_rows"] == 1
    assert summary["complete"] is False
    assert "support_pairs" in rows[0]["metadata"]["extras"]
    assert "support_pairs" not in rows[1]["metadata"]["extras"]
