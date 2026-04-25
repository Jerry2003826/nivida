from __future__ import annotations

import zipfile

import pytest

import src.student.package_submission as package_submission_module
from src.student.package_submission import build_submission_zip, read_adapter_rank, validate_adapter_dir


def test_validate_adapter_dir_requires_weight_file(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        validate_adapter_dir(adapter_dir)


def test_build_submission_zip_writes_root_level_adapter_files(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    (adapter_dir / "training_metadata.json").write_text("{}", encoding="utf-8")

    zip_path = tmp_path / "submission.zip"
    build_submission_zip(adapter_dir, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "adapter_config.json" in names
    assert "adapter_model.safetensors" in names
    assert "training_metadata.json" not in names
    assert names == {"adapter_config.json", "adapter_model.safetensors"}


def test_read_adapter_rank_extracts_rank_from_config(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"r": 32}', encoding="utf-8")
    assert read_adapter_rank(adapter_dir) == 32


def test_build_submission_zip_rejects_oversize_archive(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")

    monkeypatch.setattr(
        package_submission_module,
        "submission_zip_size_bytes",
        lambda _path: package_submission_module.KAGGLE_SINGLE_FILE_LIMIT_BYTES + 1,
    )

    with pytest.raises(ValueError, match="Kaggle single-file limit"):
        package_submission_module.build_submission_zip(
            adapter_dir,
            tmp_path / "submission.zip",
        )
