from __future__ import annotations

import pytest

from src.student.lora_train import (
    apply_runtime_environment,
    build_training_arguments_kwargs,
    dry_run_manifest,
    ensure_generation_output_aliases,
    infer_recommended_max_seq_length,
    load_tokenizer,
    normalise_target_modules,
    requires_mamba_ssm,
    validate_lora_config,
)


def test_validate_lora_config_rejects_large_rank() -> None:
    with pytest.raises(ValueError):
        validate_lora_config({"lora": {"rank": 64}})


def test_dry_run_manifest_contains_resolved_fields() -> None:
    manifest = dry_run_manifest(
        {
            "base_model": "nvidia/Nemotron-3-Nano-30B",
            "model_source": "kagglehub",
            "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
            "environment": {"KAGGLEHUB_CACHE": "/workspace/.cache/kagglehub"},
            "lora": {"rank": 32, "target_modules": ["in_proj", "out_proj"]},
            "training": {"output_dir": "artifacts/adapter"},
        }
    )
    assert manifest["status"] == "dry_run_ok"
    assert manifest["model_source"] == "kagglehub"
    assert manifest["model_handle"].endswith("/default")
    assert manifest["environment"]["KAGGLEHUB_CACHE"] == "/workspace/.cache/kagglehub"
    assert manifest["resolved_target_modules"] == ["in_proj", "out_proj"]


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
