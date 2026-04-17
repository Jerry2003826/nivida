from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from src.common.io import load_jsonl, read_yaml, write_jsonl
from src.competition.answer_extract import extract_single_boxed_answer
from src.competition.prompt_templates import build_competition_prompt
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
) -> str:
    torch = _import_or_raise("torch")
    encoded = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][encoded["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return postprocess_generation(text)


def run_inference(
    config: dict[str, Any],
    *,
    input_path: str | Path,
    adapter_dir: str | Path,
    output_path: str | Path,
    max_new_tokens: int = 128,
) -> Path:
    examples = [PuzzleExample.from_dict(row) for row in load_jsonl(input_path)]
    model, tokenizer = load_adapter_model(config, adapter_dir)
    rows = []
    for example in examples:
        prompt = build_competition_prompt(example)
        prediction = generate_single_prediction(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        rows.append({"id": example.id, "prediction": prediction})
    write_jsonl(output_path, rows)
    return Path(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a LoRA adapter and run local inference on canonical JSONL.")
    parser.add_argument("--config", default="configs/train_lora.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", default="data/processed/inference_predictions.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    config = read_yaml(args.config)
    run_inference(
        config,
        input_path=args.input,
        adapter_dir=args.adapter_dir,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
