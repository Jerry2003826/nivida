from __future__ import annotations
from collections import namedtuple
from pathlib import Path

import pytest

import src.student.preflight as preflight
from src.common.io import write_json, write_jsonl
from src.student.lora_train import (
    apply_runtime_environment,
    build_training_arguments_kwargs,
    dry_run_manifest,
    ensure_generation_output_aliases,
    infer_recommended_max_seq_length,
    load_tokenizer,
    normalise_target_modules,
    requires_mamba_ssm,
    summarise_supervised_records,
    validate_lora_config,
)
from src.student.preflight import TrainingPreflightError, run_training_preflight


def _write_dataset(path: Path) -> Path:
    write_jsonl(
        path,
        [
            {
                "id": "ex-1",
                "prompt": "a b",
                "completion": "c",
                "official_family": "bit",
                "subtype": "bit_xor_mask",
            },
            {
                "id": "ex-2",
                "prompt": "x",
                "completion": "y z",
                "official_family": "cipher",
                "subtype": "cipher_vocab",
            },
        ],
    )
    return path


def _base_config(tmp_path: Path) -> dict[str, object]:
    dataset_path = _write_dataset(tmp_path / "train.jsonl")
    return {
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "model_source": "huggingface",
        "trust_remote_code": False,
        "lora": {"rank": 32, "target_modules": ["in_proj", "out_proj"]},
        "training": {
            "output_dir": str(tmp_path / "artifacts" / "adapter"),
            "dataset_path": str(dataset_path),
        },
    }


def _usage(total_gb: int, free_gb: int):
    usage_type = namedtuple("usage", ["total", "used", "free"])
    total = total_gb * 1024**3
    free = free_gb * 1024**3
    used = total - free
    return usage_type(total, used, free)


def test_validate_lora_config_rejects_large_rank() -> None:
    with pytest.raises(ValueError):
        validate_lora_config({"lora": {"rank": 64}})


def test_dry_run_manifest_contains_resolved_fields(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.update(
        {
            "model_source": "kagglehub",
            "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
            "environment": {"KAGGLEHUB_CACHE": "/workspace/.cache/kagglehub"},
        }
    )

    manifest = dry_run_manifest(config)

    assert manifest["status"] == "dry_run_ok"
    assert manifest["model_source"] == "kagglehub"
    assert manifest["model_handle"].endswith("/default")
    assert manifest["environment"]["KAGGLEHUB_CACHE"] == "/workspace/.cache/kagglehub"
    assert manifest["resolved_target_modules"] == ["in_proj", "out_proj"]
    assert manifest["preflight"]["status"] == "ok"


def test_requires_mamba_ssm_only_for_nemotron_like_configs() -> None:
    assert requires_mamba_ssm({"model_source": "kagglehub", "model_handle": "metric/nemotron-3/default"})
    assert requires_mamba_ssm({"base_model": "nvidia/Nemotron-3-Nano-30B"})
    assert not requires_mamba_ssm({"base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "trust_remote_code": False})


def test_build_training_arguments_kwargs_supports_old_and_new_transformers_names() -> None:
    class _OldArgs:
        def __init__(self, output_dir, evaluation_strategy=None, **kwargs):
            pass

    class _NewArgs:
        def __init__(self, output_dir, eval_strategy=None, **kwargs):
            pass

    old_kwargs = build_training_arguments_kwargs(
        type("T", (), {"TrainingArguments": _OldArgs}),
        {"training": {}},
        output_dir="artifacts/x",
        has_eval_dataset=True,
    )
    new_kwargs = build_training_arguments_kwargs(
        type("T", (), {"TrainingArguments": _NewArgs}),
        {"training": {}},
        output_dir="artifacts/x",
        has_eval_dataset=False,
    )

    assert old_kwargs["evaluation_strategy"] == "steps"
    assert "eval_strategy" not in old_kwargs
    assert new_kwargs["eval_strategy"] == "no"
    assert "evaluation_strategy" not in new_kwargs
    assert old_kwargs["warmup_ratio"] == 0.0
    assert old_kwargs["lr_scheduler_type"] == "linear"


def test_apply_runtime_environment_sets_non_empty_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLEHUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    applied = apply_runtime_environment(
        {
            "environment": {
                "KAGGLEHUB_CACHE": "/workspace/.cache/kagglehub",
                "HF_HOME": "/workspace/.cache/huggingface",
                "EMPTY_VALUE": "",
            }
        }
    )

    assert applied == {
        "KAGGLEHUB_CACHE": "/workspace/.cache/kagglehub",
        "HF_HOME": "/workspace/.cache/huggingface",
    }
    assert applied["KAGGLEHUB_CACHE"] == "/workspace/.cache/kagglehub"


def test_load_tokenizer_prefers_slow_for_nemotron_like_configs() -> None:
    calls: list[dict[str, object]] = []

    class _Tokenizer:
        pad_token = None
        eos_token = "</s>"

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            calls.append({"model_path": model_path, **kwargs})
            return _Tokenizer()

    tokenizer = load_tokenizer(
        type("T", (), {"AutoTokenizer": _AutoTokenizer}),
        "cached/model",
        {"base_model": "nvidia/Nemotron-3-Nano-30B", "trust_remote_code": True},
    )

    assert calls == [{"model_path": "cached/model", "trust_remote_code": True, "use_fast": False}]
    assert tokenizer.pad_token == "</s>"


def test_load_tokenizer_falls_back_to_slow_when_fast_fails() -> None:
    calls: list[dict[str, object]] = []

    class _Tokenizer:
        pad_token = "<pad>"
        eos_token = "</s>"

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            calls.append({"model_path": model_path, **kwargs})
            if kwargs.get("use_fast", True):
                raise ValueError("broken fast tokenizer")
            return _Tokenizer()

    tokenizer = load_tokenizer(
        type("T", (), {"AutoTokenizer": _AutoTokenizer}),
        "cached/model",
        {"base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "trust_remote_code": False},
    )

    assert calls == [
        {"model_path": "cached/model", "trust_remote_code": False, "use_fast": True},
        {"model_path": "cached/model", "trust_remote_code": False, "use_fast": False},
    ]
    assert tokenizer.pad_token == "<pad>"


def test_ensure_generation_output_aliases_backfills_missing_decoder_outputs() -> None:
    generation = type("Generation", (), {"GenerateDecoderOnlyOutput": object(), "TextStreamer": object()})()
    transformers_module = type("Transformers", (), {"generation": generation})

    aliases = ensure_generation_output_aliases(transformers_module)

    assert aliases == {
        "GreedySearchDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
        "SampleDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
    }
    assert generation.GreedySearchDecoderOnlyOutput is generation.GenerateDecoderOnlyOutput
    assert generation.SampleDecoderOnlyOutput is generation.GenerateDecoderOnlyOutput


def test_normalise_target_modules_supports_lists_and_regex() -> None:
    assert normalise_target_modules(["q_proj", "v_proj"]) == ["q_proj", "v_proj"]
    assert normalise_target_modules(".*proj$") == ".*proj$"


def test_infer_recommended_max_seq_length_uses_p95_and_floor() -> None:
    records = [
        type("Record", (), {"prompt": "a", "completion": " ".join(["x"] * 10)})(),
        type("Record", (), {"prompt": "a", "completion": " ".join(["x"] * 1100)})(),
    ]
    assert infer_recommended_max_seq_length(records, floor=1024, minimum_above_floor=1536, cap=2048) == 1536


def test_summarise_supervised_records_reports_distributions() -> None:
    records = [
        type("Record", (), {"prompt": "a b", "completion": "c", "official_family": "bit", "subtype": "bit_xor_mask"})(),
        type("Record", (), {"prompt": "a", "completion": "c d", "official_family": "cipher", "subtype": "cipher_vocab"})(),
    ]
    summary = summarise_supervised_records(records, max_seq_length=2)
    assert summary["num_samples"] == 2
    assert summary["family_distribution"]["bit"] == 1
    assert summary["subtype_distribution"]["cipher:cipher_vocab"] == 1


def test_preflight_missing_dataset_path_fails(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config["training"] = {"output_dir": str(tmp_path / "artifacts" / "adapter")}

    with pytest.raises(TrainingPreflightError, match="dataset_path"):
        run_training_preflight(config, dry_run=True, repo_root=tmp_path)


def test_preflight_missing_eval_path_fails_when_configured(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config["training"]["eval_path"] = str(tmp_path / "missing_eval.jsonl")

    with pytest.raises(TrainingPreflightError, match="eval_path"):
        run_training_preflight(config, dry_run=True, repo_root=tmp_path)


def test_preflight_missing_tokenizer_path_fails_when_configured(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config["tokenizer_path"] = str(tmp_path / "missing_tokenizer")

    with pytest.raises(TrainingPreflightError, match="tokenizer_path"):
        run_training_preflight(config, dry_run=True, repo_root=tmp_path)


def test_preflight_unwritable_output_dir_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_config(tmp_path)

    def _boom(_output_dir: Path) -> Path:
        raise TrainingPreflightError(
            "output_dir: Unable to write to output dir. Choose a writable output_dir."
        )

    monkeypatch.setattr(preflight, "_ensure_output_dir_writable", _boom)

    with pytest.raises(TrainingPreflightError, match="output_dir"):
        run_training_preflight(config, dry_run=True, repo_root=tmp_path)


def test_non_dry_kaggle_check_uses_tokenizer_config_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_config(tmp_path)
    config.update(
        {
            "base_model": "nvidia/Nemotron-3-Nano-30B",
            "model_source": "kagglehub",
            "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
            "trust_remote_code": True,
        }
    )
    bundle_root = tmp_path / "cache" / "bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)
    probe_file = bundle_root / "tokenizer_config.json"
    probe_file.write_text("{}", encoding="utf-8")
    (bundle_root / "model.safetensors").write_text("x", encoding="utf-8")

    calls: list[tuple[str, str]] = []

    class _FakeKaggleHub:
        @staticmethod
        def model_download(handle: str, path: str | None = None) -> str:
            calls.append((handle, str(path)))
            return str(probe_file)

    def _fake_import(module_name: str, remediation: str):
        if module_name == "kagglehub":
            return _FakeKaggleHub()
        return object()

    monkeypatch.setattr(preflight, "_import_or_raise", _fake_import)
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _path: _usage(200, 30))

    report = run_training_preflight(config, dry_run=False, repo_root=tmp_path)

    assert calls == [
        (
            "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
            "tokenizer_config.json",
        )
    ]
    assert report["kaggle_model_cached"] is True
    assert report["required_disk_gb"] == 20


def test_preflight_in_dry_run_mode_skips_remote_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_config(tmp_path)
    config.update(
        {
            "model_source": "kagglehub",
            "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
        }
    )

    def _boom(*args, **kwargs):
        raise AssertionError("dry-run preflight must not call remote kaggle checks")

    monkeypatch.setattr(preflight, "_download_kaggle_probe_file", _boom)

    report = run_training_preflight(config, dry_run=True, repo_root=tmp_path)

    assert report["checks"]["remote_runtime_checks"]["ok"] is True
    assert report["kaggle_model_cached"] is None
    assert report["required_disk_gb"] is None


def test_dry_run_manifest_embeds_preflight_report(tmp_path: Path) -> None:
    config = _base_config(tmp_path)

    manifest = dry_run_manifest(config)

    assert manifest["preflight"]["status"] == "ok"
    assert manifest["preflight"]["checks"]["dataset_path"]["ok"] is True


def test_preflight_uses_probe_sha_and_relative_tokenizer_path(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.update(
        {
            "model_source": "kagglehub",
            "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
            "tokenizer_path": "artifacts/tokenizer",
        }
    )
    tokenizer_dir = tmp_path / "artifacts" / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    write_json(
        tmp_path / "artifacts" / "chat_template_probe.json",
        {
            "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
            "tokenizer_path": "artifacts/tokenizer",
            "chat_template_sha16": "abcd1234ef567890",
        },
    )

    report = run_training_preflight(config, dry_run=True, repo_root=tmp_path)

    assert report["tokenizer_path"] == "artifacts/tokenizer"
    assert report["chat_template_sha16"] == "abcd1234ef567890"


def test_preflight_disk_threshold_is_20gb_when_model_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_config(tmp_path)
    config.update(
        {
            "base_model": "nvidia/Nemotron-3-Nano-30B",
            "model_source": "kagglehub",
            "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
            "trust_remote_code": True,
        }
    )
    bundle_root = tmp_path / "cache" / "cached_bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)
    probe_file = bundle_root / "tokenizer_config.json"
    probe_file.write_text("{}", encoding="utf-8")
    (bundle_root / "weights.safetensors").write_text("x", encoding="utf-8")

    def _fake_import(module_name: str, remediation: str):
        if module_name == "kagglehub":
            return type("FakeKaggle", (), {"model_download": staticmethod(lambda handle, path=None: str(probe_file))})()
        return object()

    monkeypatch.setattr(preflight, "_import_or_raise", _fake_import)
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _path: _usage(200, 25))

    report = run_training_preflight(config, dry_run=False, repo_root=tmp_path)

    assert report["kaggle_model_cached"] is True
    assert report["required_disk_gb"] == 20
    assert report["disk_free_gb"] == 25.0


def test_preflight_disk_threshold_is_80gb_when_model_not_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_config(tmp_path)
    config.update(
        {
            "base_model": "nvidia/Nemotron-3-Nano-30B",
            "model_source": "kagglehub",
            "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
            "trust_remote_code": True,
        }
    )
    bundle_root = tmp_path / "cache" / "tokenizer_only_bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)
    probe_file = bundle_root / "tokenizer_config.json"
    probe_file.write_text("{}", encoding="utf-8")

    def _fake_import(module_name: str, remediation: str):
        if module_name == "kagglehub":
            return type("FakeKaggle", (), {"model_download": staticmethod(lambda handle, path=None: str(probe_file))})()
        return object()

    monkeypatch.setattr(preflight, "_import_or_raise", _fake_import)
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _path: _usage(200, 90))

    report = run_training_preflight(config, dry_run=False, repo_root=tmp_path)

    assert report["kaggle_model_cached"] is False
    assert report["required_disk_gb"] == 80
    assert report["disk_free_gb"] == 90.0
