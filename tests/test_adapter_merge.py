from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from src.research.adapter_merge import (
    TensorValue,
    _read_safetensors,
    _write_safetensors,
    merge_lora_adapters,
    parse_adapter_spec,
)
from src.student.package_submission import build_submission_zip


def _write_adapter(path: Path, *, a: np.ndarray, b: np.ndarray, rank: int = 2) -> Path:
    path.mkdir(parents=True)
    path.joinpath("adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": rank,
                "lora_alpha": rank,
                "target_modules": ["q_proj"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_safetensors(
        path / "adapter_model.safetensors",
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": TensorValue(a, "F32"),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": TensorValue(b, "F32"),
        },
    )
    return path


def test_parse_adapter_spec_uses_last_colon_for_weight() -> None:
    spec = parse_adapter_spec("a=C:/tmp/adapter:0.25")

    assert spec.name == "a"
    assert spec.path.as_posix().endswith("C:/tmp/adapter")
    assert spec.weight == 0.25


def test_linear_merge_writes_submit_safe_adapter_and_manifest(tmp_path: Path) -> None:
    a1 = _write_adapter(
        tmp_path / "a1",
        a=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        b=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    a2 = _write_adapter(
        tmp_path / "a2",
        a=np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        b=np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
    )

    manifest = merge_lora_adapters(
        adapters=[
            parse_adapter_spec(f"first={a1}:0.25"),
            parse_adapter_spec(f"second={a2}:0.75"),
        ],
        output_dir=tmp_path / "merged",
        method="linear",
    )

    assert manifest["submit_safe"] is True
    tensors = _read_safetensors(tmp_path / "merged" / "adapter_model.safetensors")
    merged_a = tensors["base_model.model.layers.0.self_attn.q_proj.lora_A.weight"].array
    np.testing.assert_allclose(
        merged_a,
        np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32),
    )
    zip_path = build_submission_zip(tmp_path / "merged", tmp_path / "submission.zip")
    with zipfile.ZipFile(zip_path, "r") as archive:
        assert sorted(archive.namelist()) == ["adapter_config.json", "adapter_model.safetensors"]


def test_svd_rank32_merge_compresses_lora_pair(tmp_path: Path) -> None:
    a1 = _write_adapter(
        tmp_path / "a1",
        a=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        b=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    a2 = _write_adapter(
        tmp_path / "a2",
        a=np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        b=np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    )

    manifest = merge_lora_adapters(
        adapters=[
            parse_adapter_spec(f"first={a1}:0.5"),
            parse_adapter_spec(f"second={a2}:0.5"),
        ],
        output_dir=tmp_path / "svd",
        method="svd-rank32",
        target_rank=1,
    )

    config = json.loads((tmp_path / "svd" / "adapter_config.json").read_text(encoding="utf-8"))
    tensors = _read_safetensors(tmp_path / "svd" / "adapter_model.safetensors")
    assert manifest["output_rank"] == 1
    assert config["r"] == 1
    assert tensors["base_model.model.layers.0.self_attn.q_proj.lora_A.weight"].array.shape == (1, 2)
    assert tensors["base_model.model.layers.0.self_attn.q_proj.lora_B.weight"].array.shape == (2, 1)
