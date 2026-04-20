from __future__ import annotations
from collections import namedtuple
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pytest
from safetensors.numpy import save_file

import src.student.lora_train as lora_train_module
import src.student.preflight as preflight
from src.common.io import write_json, write_jsonl
from src.student.adapter_submission_budget import _HYPOTHETICAL_OVER_LIMIT_TARGET_REGEX
from src.student.lora_train import (
    apply_runtime_environment,
    assert_saved_lora_artifact_healthy,
    build_training_arguments_kwargs,
    configure_model_for_training,
    dry_run_manifest,
    ensure_lora_parameters_trainable,
    ensure_generation_output_aliases,
    infer_recommended_max_seq_length,
    initialise_lora_model,
    LoraTrainingHealth,
    load_tokenizer,
    normalise_target_modules,
    probe_saved_lora_artifact,
    requires_mamba_ssm,
    sanitize_nonfinite_lora_gradients,
    should_capture_loss_outputs,
    should_require_strict_divergence_check,
    summarise_lora_runtime_state,
    summarise_model_losses,
    summarise_supervised_records,
    summarise_lora_gradients,
    validate_init_adapter_compatibility,
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


def _write_adapter_safetensors(
    path: Path,
    *,
    lora_a_value: float = 0.25,
    lora_b_value: float = 0.0,
    copies: int = 3,
) -> Path:
    tensors = {}
    for idx in range(copies):
        prefix = f"base_model.model.backbone.layers.{idx}.mixer.in_proj"
        tensors[f"{prefix}.lora_A.weight"] = np.full((2, 4), lora_a_value + idx, dtype=np.float32)
        tensors[f"{prefix}.lora_B.weight"] = np.full((4, 2), lora_b_value, dtype=np.float32)
    save_file(tensors, str(path))
    return path


def test_validate_lora_config_rejects_large_rank() -> None:
    with pytest.raises(ValueError):
        validate_lora_config({"lora": {"rank": 64}})


def test_validate_lora_config_rejects_submission_unsafe_target_modules() -> None:
    with pytest.raises(ValueError, match="submission-safe"):
        validate_lora_config(
            {
                "base_model": "nvidia/Nemotron-3-Nano-30B",
                "model_source": "kagglehub",
                "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
                "trust_remote_code": True,
                "lora": {
                    "rank": 32,
                    "target_modules": _HYPOTHETICAL_OVER_LIMIT_TARGET_REGEX,
                },
            }
        )


def test_validate_lora_config_strict_on_unknown_model() -> None:
    config = {
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "model_source": "huggingface",
        "trust_remote_code": False,
        "lora": {"rank": 32, "target_modules": ["in_proj", "out_proj"]},
    }

    with pytest.raises(ValueError, match="cannot be estimated"):
        validate_lora_config(config)

    validate_lora_config(config, allow_unknown_model=True)


def test_dry_run_manifest_contains_resolved_fields(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.update(
        {
            "base_model": "nvidia/Nemotron-3-Nano-30B",
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
    assert manifest["submission_budget"]["status"] == "ok"


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


def test_configure_model_for_training_disables_use_cache_with_gradient_checkpointing() -> None:
    class _DummyModel:
        def __init__(self) -> None:
            self.config = type("Config", (), {"use_cache": True})()
            self.generation_config = type("GenerationConfig", (), {"use_cache": True})()
            self.gradient_checkpointing_enabled = False

        def gradient_checkpointing_enable(self) -> None:
            self.gradient_checkpointing_enabled = True

    model = _DummyModel()

    configured = configure_model_for_training(model, {"training": {"gradient_checkpointing": True}})

    assert configured is model
    assert model.config.use_cache is False
    assert model.generation_config.use_cache is False
    assert model.gradient_checkpointing_enabled is True


def test_ensure_lora_parameters_trainable_reenables_frozen_lora_params() -> None:
    class _FakeParameter:
        def __init__(self, *, numel: int, requires_grad: bool) -> None:
            self._numel = numel
            self.requires_grad = requires_grad
            self.grad = None

        def numel(self) -> int:
            return self._numel

    class _DummyModel:
        def __init__(self) -> None:
            self._params = [
                ("base_model.model.layers.0.mixer.in_proj.lora_A.weight", _FakeParameter(numel=4, requires_grad=False)),
                ("base_model.model.layers.0.mixer.in_proj.lora_B.weight", _FakeParameter(numel=4, requires_grad=False)),
                ("base_model.model.layers.0.mixer.in_proj.weight", _FakeParameter(numel=4, requires_grad=False)),
            ]

        def named_parameters(self):
            return list(self._params)

    model = _DummyModel()

    summary = ensure_lora_parameters_trainable(model)

    assert summary["lora_parameter_tensors"] == 2
    assert summary["trainable_lora_parameter_tensors"] == 2
    assert summary["reenabled_lora_parameter_tensors"] == 2
    assert all(parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" in name)
    assert not next(parameter for name, parameter in model.named_parameters() if "lora_" not in name).requires_grad


def test_summarise_lora_gradients_reports_nonzero_gradients() -> None:
    class _FakeNorm:
        def __init__(self, value: float) -> None:
            self.value = value

        def item(self) -> float:
            return self.value

    class _FakeGrad:
        def __init__(self, norm_value: float) -> None:
            self.norm_value = norm_value

        def detach(self):
            return self

        def float(self):
            return self

        def norm(self):
            return _FakeNorm(self.norm_value)

    class _FakeParameter:
        def __init__(self, *, requires_grad: bool, grad: _FakeGrad | None) -> None:
            self.requires_grad = requires_grad
            self.grad = grad

        def numel(self) -> int:
            return 4

    class _DummyModel:
        def __init__(self) -> None:
            self._params = [
                ("adapter.lora_A.weight", _FakeParameter(requires_grad=True, grad=_FakeGrad(1.0))),
                ("adapter.lora_B.weight", _FakeParameter(requires_grad=True, grad=_FakeGrad(3.0))),
            ]

        def named_parameters(self):
            return list(self._params)

    summary = summarise_lora_gradients(_DummyModel())

    assert summary["lora_parameter_tensors"] == 2
    assert summary["trainable_lora_parameter_tensors"] == 2
    assert summary["lora_tensors_with_grad"] == 2
    assert summary["mean_grad_norm"] == pytest.approx((1.0 + 3.0) / 2.0)
    assert summary["max_grad_norm"] == pytest.approx(3.0)
    assert summary["nonfinite_grad_tensors"] == 0
    assert summary["nonfinite_grad_elements"] == 0


def test_sanitize_nonfinite_lora_gradients_zeroes_only_nonfinite_entries() -> None:
    torch = pytest.importorskip("torch")

    class _Parameter:
        def __init__(self, grad: torch.Tensor) -> None:
            self.requires_grad = True
            self.grad = grad

        def numel(self) -> int:
            return int(self.grad.numel())

    class _DummyModel:
        def __init__(self) -> None:
            self._params = [
                (
                    "adapter.lora_A.weight",
                    _Parameter(torch.tensor([[1.0, float("inf")], [float("nan"), -2.0]], dtype=torch.float32)),
                ),
                (
                    "adapter.lora_B.weight",
                    _Parameter(torch.tensor([[0.5, -0.25]], dtype=torch.float32)),
                ),
            ]

        def named_parameters(self):
            return list(self._params)

    model = _DummyModel()

    summary = sanitize_nonfinite_lora_gradients(model)

    assert summary["lora_tensors_with_grad"] == 2
    assert summary["nonfinite_grad_tensors"] == 1
    assert summary["nonfinite_grad_elements"] == 2
    assert summary["sanitized_parameter_names"] == ["adapter.lora_A.weight"]
    assert torch.equal(
        model._params[0][1].grad,
        torch.tensor([[1.0, 0.0], [0.0, -2.0]], dtype=torch.float32),
    )
    assert torch.equal(
        model._params[1][1].grad,
        torch.tensor([[0.5, -0.25]], dtype=torch.float32),
    )


def test_summarise_lora_gradients_reports_nonfinite_counts_without_inf_norm() -> None:
    torch = pytest.importorskip("torch")

    class _Parameter:
        def __init__(self, grad: torch.Tensor) -> None:
            self.requires_grad = True
            self.grad = grad

        def numel(self) -> int:
            return int(self.grad.numel())

    class _DummyModel:
        def __init__(self) -> None:
            self._params = [
                (
                    "adapter.lora_A.weight",
                    _Parameter(torch.tensor([[3.0, float("inf")]], dtype=torch.float32)),
                ),
                (
                    "adapter.lora_B.weight",
                    _Parameter(torch.tensor([[4.0]], dtype=torch.float32)),
                ),
            ]

        def named_parameters(self):
            return list(self._params)

    summary = summarise_lora_gradients(_DummyModel())

    assert summary["lora_tensors_with_grad"] == 2
    assert summary["nonfinite_grad_tensors"] == 1
    assert summary["nonfinite_grad_elements"] == 1
    assert summary["mean_grad_norm"] == pytest.approx((3.0 + 4.0) / 2.0)
    assert summary["max_grad_norm"] == pytest.approx(4.0)


def test_lora_training_health_emits_diagnostics_to_stdout_and_file(capsys: pytest.CaptureFixture[str]) -> None:
    temp_dir = Path(tempfile.mkdtemp(dir=Path.cwd()))
    try:
        health = LoraTrainingHealth(
            output_dir=temp_dir,
            initial_probe=None,
            probe_tensor_limit=8,
            strict_divergence_after_step=250,
            strict_divergence_after_warmup_multiplier=2.0,
        )

        health.log_trainable_summary(
            {
                "trainable_lora_parameter_tensors": 2,
                "lora_parameter_tensors": 2,
                "trainable_lora_parameter_numel": 8,
                "lora_parameter_numel": 8,
                "reenabled_lora_parameter_tensors": 0,
            }
        )

        captured = capsys.readouterr()
        expected = "[LoRA] trainable tensors=2/2 numel=8/8 reenabled=0"
        assert expected in captured.out
        assert (temp_dir / "training_diagnostics.log").read_text(encoding="utf-8").splitlines() == [expected]
    finally:
        shutil.rmtree(temp_dir)


def test_probe_saved_lora_artifact_reports_zero_lora_b(tmp_path: Path) -> None:
    adapter_model_path = _write_adapter_safetensors(tmp_path / "adapter_model.safetensors")

    probe = probe_saved_lora_artifact(adapter_model_path, max_tensors=8)

    assert probe["lora_b_tensor_count"] == 3
    assert probe["sampled_lora_b_zero_fraction"] == 1.0
    assert probe["sampled_lora_b_max_abs"] == 0.0
    assert len(probe["sampled_tensor_digest"]) == 64


def test_probe_saved_lora_artifact_digest_normalises_storage_dtype(tmp_path: Path) -> None:
    fp32_path = tmp_path / "adapter_fp32.safetensors"
    fp16_path = tmp_path / "adapter_fp16.safetensors"
    tensors_fp32 = {
        "base_model.model.backbone.layers.0.mixer.in_proj.lora_A.weight": np.full((2, 4), 0.5, dtype=np.float32),
        "base_model.model.backbone.layers.0.mixer.in_proj.lora_B.weight": np.full((4, 2), 0.125, dtype=np.float32),
    }
    tensors_fp16 = {
        key: value.astype(np.float16)
        for key, value in tensors_fp32.items()
    }
    save_file(tensors_fp32, str(fp32_path))
    save_file(tensors_fp16, str(fp16_path))

    fp32_probe = probe_saved_lora_artifact(fp32_path, max_tensors=8)
    fp16_probe = probe_saved_lora_artifact(fp16_path, max_tensors=8)

    assert fp32_probe["sampled_tensor_digest"] == fp16_probe["sampled_tensor_digest"]
    assert fp32_probe["sampled_lora_b_zero_fraction"] == fp16_probe["sampled_lora_b_zero_fraction"]


def test_probe_saved_lora_artifact_supports_bfloat16_storage_dtype(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_torch_file = pytest.importorskip("safetensors.torch").save_file

    bf16_path = tmp_path / "adapter_bf16.safetensors"
    tensors = {
        "base_model.model.backbone.layers.0.mixer.in_proj.lora_A.weight": torch.full(
            (2, 4), 0.5, dtype=torch.bfloat16
        ),
        "base_model.model.backbone.layers.0.mixer.in_proj.lora_B.weight": torch.full(
            (4, 2), 0.125, dtype=torch.bfloat16
        ),
    }
    save_torch_file(tensors, str(bf16_path))

    probe = probe_saved_lora_artifact(bf16_path, max_tensors=8)

    assert probe["sampled_tensor_count"] == 2
    assert probe["sampled_lora_b_zero_fraction"] == 0.0
    assert probe["sampled_lora_b_max_abs"] == pytest.approx(0.125)


def test_assert_saved_lora_artifact_healthy_rejects_all_zero_lora_b(tmp_path: Path) -> None:
    adapter_model_path = _write_adapter_safetensors(tmp_path / "adapter_model.safetensors")

    with pytest.raises(RuntimeError, match="all-zero"):
        assert_saved_lora_artifact_healthy(adapter_model_path, step_label="checkpoint-100")


def test_assert_saved_lora_artifact_healthy_rejects_unchanged_init_adapter(tmp_path: Path) -> None:
    init_path = _write_adapter_safetensors(
        tmp_path / "init_adapter_model.safetensors",
        lora_b_value=0.125,
    )
    current_path = _write_adapter_safetensors(
        tmp_path / "current_adapter_model.safetensors",
        lora_b_value=0.125,
    )

    init_probe = probe_saved_lora_artifact(init_path, max_tensors=8)

    with pytest.raises(RuntimeError, match="byte-identical"):
        assert_saved_lora_artifact_healthy(
            current_path,
            step_label="checkpoint-125",
            initial_probe=init_probe,
            max_tensors=8,
        )


def test_assert_saved_lora_artifact_checkpoint_mode_allows_init_identical_digest(tmp_path: Path) -> None:
    init_path = _write_adapter_safetensors(
        tmp_path / "init_adapter_model.safetensors",
        lora_b_value=0.125,
    )
    checkpoint_path = _write_adapter_safetensors(
        tmp_path / "checkpoint_adapter_model.safetensors",
        lora_b_value=0.125,
    )

    init_probe = probe_saved_lora_artifact(init_path, max_tensors=8)

    probe = assert_saved_lora_artifact_healthy(
        checkpoint_path,
        step_label="checkpoint-125",
        initial_probe=init_probe,
        max_tensors=8,
        require_divergence_from_initial=False,
    )

    assert probe["sampled_tensor_digest"] == init_probe["sampled_tensor_digest"]


def test_summarise_model_losses_extracts_named_components() -> None:
    class _TensorLike:
        def __init__(self, value: float) -> None:
            self.value = value

        def detach(self):
            return self

        def float(self):
            return self

        def item(self) -> float:
            return self.value

    outputs = type(
        "Outputs",
        (),
        {
            "loss": _TensorLike(2.5),
            "aux_loss": _TensorLike(0.75),
            "router_aux_loss": _TensorLike(0.125),
        },
    )()

    summary = summarise_model_losses(outputs)

    assert summary == {
        "loss": pytest.approx(2.5),
        "aux_loss": pytest.approx(0.75),
        "router_aux_loss": pytest.approx(0.125),
    }


def test_summarise_lora_runtime_state_reports_disabled_and_active_adapters() -> None:
    class _LoraModule:
        def __init__(self, *, disabled: bool, merged: bool, active_adapters) -> None:
            self.lora_A = {"default": object()}
            self.lora_B = {"default": object()}
            self.disable_adapters = disabled
            self.merged = merged
            self.active_adapters = active_adapters

    class _DummyModel:
        def named_modules(self):
            return [
                ("", self),
                ("layers.0.mixer.in_proj", _LoraModule(disabled=False, merged=False, active_adapters=["default"])),
                ("layers.0.mixer.out_proj", _LoraModule(disabled=True, merged=True, active_adapters=["repair"])),
                ("layers.0.mixer.core", object()),
            ]

    summary = summarise_lora_runtime_state(_DummyModel())

    assert summary == {
        "lora_module_count": 2,
        "disabled_lora_module_count": 1,
        "merged_lora_module_count": 1,
        "active_adapters": ["default", "repair"],
    }


def test_should_capture_loss_outputs_only_on_logging_steps_by_default() -> None:
    assert should_capture_loss_outputs(global_step=8, logging_steps=10, return_outputs=False) is False
    assert should_capture_loss_outputs(global_step=9, logging_steps=10, return_outputs=False) is True
    assert should_capture_loss_outputs(global_step=0, logging_steps=10, return_outputs=True) is True


def test_should_require_strict_divergence_check_triggers_after_early_threshold() -> None:
    assert should_require_strict_divergence_check(
        global_step=100,
        warmup_steps=50,
        floor_step=250,
        warmup_multiplier=2.0,
    ) is True
    assert should_require_strict_divergence_check(
        global_step=72,
        warmup_steps=36,
        floor_step=250,
        warmup_multiplier=2.0,
    ) is True
    assert should_require_strict_divergence_check(
        global_step=70,
        warmup_steps=36,
        floor_step=250,
        warmup_multiplier=2.0,
    ) is False
    assert should_require_strict_divergence_check(
        global_step=249,
        warmup_steps=500,
        floor_step=250,
        warmup_multiplier=2.0,
    ) is False
    assert should_require_strict_divergence_check(
        global_step=250,
        warmup_steps=500,
        floor_step=250,
        warmup_multiplier=2.0,
    ) is True


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


def test_initialise_lora_model_raises_when_no_modules_match() -> None:
    class _FakeModel:
        @staticmethod
        def named_modules():
            return [("decoder.block", object())]

    class _FakePeft:
        class TaskType:
            CAUSAL_LM = "causal"

        class LoraConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        @staticmethod
        def get_peft_model(model, _config):
            return model

    with pytest.raises(ValueError, match="No modules matched"):
        initialise_lora_model(
            _FakeModel(),
            _FakePeft,
            {"lora": {"rank": 32, "target_modules": ["in_proj"]}, "training": {}},
        )


def test_validate_init_adapter_compatibility_rejects_target_module_mismatch(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    (adapter_dir / "adapter_config.json").write_text(
        '{"r": 32, "target_modules": ["q_proj"]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target_modules"):
        validate_init_adapter_compatibility(
            {"lora": {"rank": 32, "target_modules": ["in_proj"]}},
            adapter_dir,
        )


def test_initialise_lora_model_loads_trainable_init_adapter(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    (adapter_dir / "adapter_config.json").write_text(
        '{"r": 32, "target_modules": ["in_proj", "out_proj"]}',
        encoding="utf-8",
    )

    class _FakeModel:
        @staticmethod
        def named_modules():
            return [("layers.0.in_proj", object()), ("layers.0.out_proj", object())]

    calls: list[tuple[str, bool | None]] = []

    class _FakePeft:
        class TaskType:
            CAUSAL_LM = "causal"

        class LoraConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class PeftModel:
            @staticmethod
            def from_pretrained(model, adapter_dir_arg, is_trainable=True):
                calls.append((str(adapter_dir_arg), is_trainable))
                return model

        @staticmethod
        def get_peft_model(model, _config):
            raise AssertionError("fresh LoRA path should not be used for continuation")

    model, metadata = initialise_lora_model(
        _FakeModel(),
        _FakePeft,
        {
            "lora": {"rank": 32, "target_modules": ["in_proj", "out_proj"]},
            "training": {"init_adapter_dir": str(adapter_dir)},
        },
    )

    assert isinstance(model, _FakeModel)
    assert calls == [(str(adapter_dir), True)]
    assert metadata["init_adapter_dir"] == str(adapter_dir)
    assert metadata["num_matched_target_modules"] == 2


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
    assert report["disk_check_path"] is None
    assert report["required_disk_gb"] is None


def test_dry_run_manifest_embeds_preflight_report(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.update(
        {
            "base_model": "nvidia/Nemotron-3-Nano-30B",
            "model_source": "kagglehub",
            "model_handle": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
        }
    )

    manifest = dry_run_manifest(config)

    assert manifest["preflight"]["status"] == "ok"
    assert manifest["preflight"]["checks"]["dataset_path"]["ok"] is True


def test_main_defaults_to_real_training_when_config_omits_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_config(tmp_path)
    config["training"].pop("dry_run", None)
    calls: list[str] = []

    monkeypatch.setattr(lora_train_module, "read_yaml", lambda _path: config)
    monkeypatch.setattr(lora_train_module, "validate_lora_config", lambda _config: None)
    monkeypatch.setattr(lora_train_module, "train_lora", lambda _config: {"status": "trained"})
    monkeypatch.setattr(lora_train_module, "write_json", lambda path, payload: calls.append(str(path)))
    monkeypatch.setattr(sys, "argv", ["lora_train", "--config", "dummy.yaml"])

    lora_train_module.main()

    assert calls == [str(Path(config["training"]["output_dir"]) / "last_run_summary.json")]


def test_main_force_train_overrides_config_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_config(tmp_path)
    config["training"]["dry_run"] = True
    trained: list[dict[str, object]] = []

    monkeypatch.setattr(lora_train_module, "read_yaml", lambda _path: config)
    monkeypatch.setattr(lora_train_module, "validate_lora_config", lambda _config: None)
    monkeypatch.setattr(lora_train_module, "train_lora", lambda _config: trained.append(_config) or {"status": "trained"})
    monkeypatch.setattr(lora_train_module, "write_json", lambda path, payload: None)
    monkeypatch.setattr(sys, "argv", ["lora_train", "--config", "dummy.yaml", "--force-train"])

    lora_train_module.main()

    assert trained == [config]


def test_main_dry_run_still_wins_over_force_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_config(tmp_path)
    config["training"]["dry_run"] = False
    dry_run_calls: list[dict[str, object]] = []

    monkeypatch.setattr(lora_train_module, "read_yaml", lambda _path: config)
    monkeypatch.setattr(lora_train_module, "validate_lora_config", lambda _config: None)
    monkeypatch.setattr(lora_train_module, "dry_run_manifest", lambda _config: dry_run_calls.append(_config) or {"status": "dry"})
    monkeypatch.setattr(lora_train_module, "train_lora", lambda _config: (_ for _ in ()).throw(AssertionError("should not train")))
    monkeypatch.setattr(lora_train_module, "write_json", lambda path, payload: None)
    monkeypatch.setattr(sys, "argv", ["lora_train", "--config", "dummy.yaml", "--dry-run", "--force-train"])

    lora_train_module.main()

    assert dry_run_calls == [config]


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
    assert report["disk_check_path"] == "artifacts/adapter"
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
            "environment": {"KAGGLEHUB_CACHE": "workspace/kagglehub-cache"},
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

    disk_usage_paths: list[Path] = []

    def _fake_disk_usage(path: Path):
        disk_usage_paths.append(Path(path))
        return _usage(200, 90)

    monkeypatch.setattr(preflight.shutil, "disk_usage", _fake_disk_usage)

    report = run_training_preflight(config, dry_run=False, repo_root=tmp_path)

    assert report["kaggle_model_cached"] is False
    assert report["disk_check_path"] == "workspace/kagglehub-cache"
    assert report["required_disk_gb"] == 80
    assert report["disk_free_gb"] == 90.0
    assert disk_usage_paths == [tmp_path / "workspace" / "kagglehub-cache"]
