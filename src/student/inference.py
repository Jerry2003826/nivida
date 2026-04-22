from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from src.common.io import load_jsonl, read_yaml, write_jsonl
from src.competition.answer_extract import extract_single_boxed_answer
from src.competition.harness_prompt import build_chat_thinking_prompt
from src.competition.prompt_templates import (
    PROMPT_MODE_CHAT_THINKING,
    PROMPT_MODE_RAW_WITH_GUARD,
    build_competition_prompt,
)
from src.competition.schema import PuzzleExample
from src.student.format_guard import wrap_boxed
from src.student.lora_train import _load_torch_dtype, resolve_model_path


def _import_or_raise(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Missing optional dependency '{module_name}'. Install training extras before running inference."
        ) from exc


def postprocess_generation(text: str) -> str:
    result = extract_single_boxed_answer(text)
    if result.is_valid and result.answer is not None:
        return wrap_boxed(result.answer)
    return text.strip()


def load_adapter_model(config: dict[str, Any], adapter_dir: str | Path) -> tuple[Any, Any]:
    _import_or_raise("mamba_ssm")
    torch = _import_or_raise("torch")
    transformers = _import_or_raise("transformers")
    peft = _import_or_raise("peft")

    model_path = resolve_model_path(config)
    dtype = _load_torch_dtype(torch, config.get("torch_dtype", "bfloat16"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=config.get("device_map", "auto"),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        torch_dtype=dtype,
    )
    model = peft.PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer


def generate_single_prediction(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> tuple[str, str]:
    torch = _import_or_raise("torch")
    encoded = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][encoded["input_ids"].shape[1] :]
    raw_text = tokenizer.decode(generated, skip_special_tokens=True)
    return postprocess_generation(raw_text), raw_text


# === Metric notebook defaults (NOT authoritative for LB selection) ===
# These mirror the ``score()`` signature inside
# 官方资料/kaggle_metric/nvidia-nemotron-metric.ipynb. Ryan Holbrook (Kaggle
# Staff) confirmed on discussion #687798 that these defaults are NOT what
# the official runner uses. Kept for legacy parity and debugging only.
OFFICIAL_EVAL_PARAMS = {
    "max_new_tokens": 3584,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "prompt_mode": PROMPT_MODE_CHAT_THINKING,
}

# === Kaggle runtime contract (AUTHORITATIVE) ===
# Source: Kaggle competition Overview/Evaluation tab table, confirmed by
# Ryan Holbrook (Kaggle Staff) in discussion #687798. Mirrors
# ``RUNTIME_SAMPLING_KWARGS`` in src.competition.official_metric_contract.
# Use ``runtime_eval=True`` (preferred) for any checkpoint selection or
# LB-aligned local inference.
RUNTIME_EVAL_PARAMS = {
    "max_new_tokens": 7680,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "prompt_mode": PROMPT_MODE_CHAT_THINKING,
}


def run_inference(
    config: dict[str, Any],
    *,
    input_path: str | Path,
    adapter_dir: str | Path,
    output_path: str | Path,
    max_new_tokens: int | None = None,
    official_eval: bool = False,
    runtime_eval: bool = False,
) -> Path:
    if official_eval and runtime_eval:
        raise ValueError(
            "official_eval and runtime_eval are mutually exclusive; "
            "prefer runtime_eval=True for LB-aligned selection."
        )
    inference_config = dict(config.get("inference", {}))
    if runtime_eval:
        inference_config.update(RUNTIME_EVAL_PARAMS)
    elif official_eval:
        inference_config.update(OFFICIAL_EVAL_PARAMS)
    prompt_mode = str(inference_config.get("prompt_mode", PROMPT_MODE_RAW_WITH_GUARD))
    max_tokens = int(max_new_tokens or inference_config.get("max_new_tokens", 128))
    do_sample = bool(inference_config.get("do_sample", False))
    temperature = float(inference_config.get("temperature", 1.0))
    top_p = float(inference_config.get("top_p", 1.0))
    repetition_penalty = float(inference_config.get("repetition_penalty", 1.0))
    save_raw_generations = bool(inference_config.get("save_raw_generations", False))

    examples = [PuzzleExample.from_dict(row) for row in load_jsonl(input_path)]
    model, tokenizer = load_adapter_model(config, adapter_dir)
    rows = []
    for example in examples:
        prompt = _build_prompt_for_inference(example, prompt_mode=prompt_mode, tokenizer=tokenizer)
        prediction, raw_generation = generate_single_prediction(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        row = {"id": example.id, "prediction": prediction}
        if save_raw_generations:
            row["raw_generation"] = raw_generation
        rows.append(row)
    write_jsonl(output_path, rows)
    return Path(output_path)


def _build_prompt_for_inference(
    example: PuzzleExample,
    *,
    prompt_mode: str,
    tokenizer: Any,
) -> str:
    """Wrap the prompt to match the training distribution selected by ``prompt_mode``.

    For ``chat_thinking`` this delegates to
    :func:`src.competition.harness_prompt.build_chat_thinking_prompt`, which is
    byte-identical to the vLLM harness input (modulo the ``<think>`` seed that
    the model fills in). For other modes the pre-harness-alignment text path is
    kept so legacy dry-runs and TinyLlama smoke tests remain untouched.
    """
    if prompt_mode == PROMPT_MODE_CHAT_THINKING:
        return build_chat_thinking_prompt(example, tokenizer)
    return build_competition_prompt(example, mode=prompt_mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a LoRA adapter and run local inference on canonical JSONL.")
    parser.add_argument("--config", default="configs/train_lora.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", default="data/processed/inference_predictions.jsonl")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--official-eval",
        action="store_true",
        help=(
            "Legacy metric-notebook defaults (do_sample=True, T=1.0, "
            "max_new_tokens=3584). NOT authoritative for LB selection; "
            "retained for parity fingerprinting only."
        ),
    )
    parser.add_argument(
        "--runtime-eval",
        action="store_true",
        help=(
            "Mirror the Kaggle Overview-tab runtime contract: chat_thinking "
            "prompt, do_sample=False (greedy), T=0.0, max_new_tokens=7680. "
            "This is the authoritative setting for LB-aligned local inference."
        ),
    )
    args = parser.parse_args()
    if args.official_eval and args.runtime_eval:
        parser.error("--official-eval and --runtime-eval are mutually exclusive")

    config = read_yaml(args.config)
    run_inference(
        config,
        input_path=args.input,
        adapter_dir=args.adapter_dir,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        official_eval=args.official_eval,
        runtime_eval=args.runtime_eval,
    )


if __name__ == "__main__":
    main()
