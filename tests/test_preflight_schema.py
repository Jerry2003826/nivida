from __future__ import annotations

from pathlib import Path

from src.common.io import write_json, write_jsonl
from src.student.preflight import run_training_preflight


def _write_dataset(path: Path) -> Path:
    write_jsonl(
        path,
        [
            {
                "id": "ex-1",
                "prompt": "alpha beta",
                "completion": "gamma",
                "official_family": "bit",
                "subtype": "bit_xor_mask",
            }
        ],
    )
    return path


def _base_config(tmp_path: Path) -> dict[str, object]:
    dataset_path = _write_dataset(tmp_path / "train.jsonl")
    tokenizer_dir = tmp_path / "artifacts" / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    write_json(
        tmp_path / "artifacts" / "chat_template_probe.json",
        {
            "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
            "tokenizer_path": "artifacts/tokenizer",
            "chat_template_sha16": "ab7813c3abdd9cb6",
        },
    )
    return {
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "model_source": "huggingface",
        "trust_remote_code": False,
        "tokenizer_path": "artifacts/tokenizer",
        "lora": {"rank": 32, "target_modules": ["in_proj", "out_proj"]},
        "training": {
            "output_dir": str(tmp_path / "artifacts" / "adapter"),
            "dataset_path": str(dataset_path),
        },
    }


def test_preflight_schema_contract_exposes_acceptance_fields(tmp_path: Path) -> None:
    report = run_training_preflight(_base_config(tmp_path), dry_run=True, repo_root=tmp_path)

    assert set(("status", "checks", "disk_free_gb", "chat_template_sha16")).issubset(report)
    assert isinstance(report["checks"], dict)
    assert report["status"] == "ok"
    assert report["chat_template_sha16"] == "ab7813c3abdd9cb6"
