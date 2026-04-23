"""vLLM-backed batched inference for LoRA adapters.

This module mirrors :func:`src.student.inference.run_inference` in both
signature and output schema so it can replace the HuggingFace generate loop
anywhere runtime-eval speed matters (Stage 2/3 bestproxy selection,
Round-1 large-scale decoding).

Key differences vs. the HF path:

* Inference backend is vLLM with ``LoRARequest`` — bit-exact to the Kaggle
  runtime contract (see ``src.competition.official_metric_contract``).
* Decoding is batched over all examples in one ``llm.generate`` call.
* ``runtime_eval=True`` (the only supported value for now) pins the
  contract to ``RUNTIME_LLM_KWARGS`` / ``RUNTIME_SAMPLING_KWARGS``.

Output JSONL rows are byte-compatible with the HF path:

    {"id": <id>, "prediction": <boxed answer text>,
     # when save_raw_generations or write_raw_generations is set:
     "raw_generation": <full generation>}
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

from src.common.io import load_jsonl, read_yaml, write_jsonl
from src.competition.harness_prompt import build_chat_thinking_prompt
from src.competition.official_metric_contract import (
    RUNTIME_LLM_KWARGS,
    RUNTIME_SAMPLING_KWARGS,
)
from src.competition.prompt_templates import (
    PROMPT_MODE_CHAT_THINKING,
    build_competition_prompt,
)
from src.competition.schema import PuzzleExample
from src.student.inference import postprocess_generation
from src.student.lora_train import resolve_model_path
from src.student.package_submission import validate_adapter_dir


def _import_or_raise(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Missing optional dependency '{module_name}'. "
            f"Install vLLM (pip install vllm) before running vllm inference."
        ) from exc


def _instantiate_vllm(base_model_path: str, llm_kwargs: dict[str, Any]) -> Any:
    vllm = _import_or_raise("vllm")
    return vllm.LLM(model=str(base_model_path), **llm_kwargs)


def _build_sampling_params(sampling_kwargs: dict[str, Any]) -> Any:
    vllm = _import_or_raise("vllm")
    # Match eval_official_vllm_proxy: no seed / stop / stop_token_ids overrides
    # so the runtime contract fingerprint stays stable.
    return vllm.SamplingParams(**sampling_kwargs)


def _build_lora_request(adapter_dir: str | Path) -> Any:
    # Lazy import matches eval_official_vllm_proxy for a single code path.
    lora_request_module = _import_or_raise("vllm.lora.request")
    return lora_request_module.LoRARequest("adapter", 1, str(adapter_dir))


def _build_prompt_for_inference(
    example: PuzzleExample,
    *,
    prompt_mode: str,
    tokenizer: Any,
) -> str:
    """Render the prompt consistently with the HF path.

    For ``chat_thinking`` (the only LB-aligned mode) we call
    :func:`build_chat_thinking_prompt` — byte-identical to the harness.
    Other modes fall back to ``build_competition_prompt`` so legacy dry-runs
    (TinyLlama smoke, raw_with_guard) behave identically to the HF code.
    """
    if prompt_mode == PROMPT_MODE_CHAT_THINKING:
        return build_chat_thinking_prompt(example, tokenizer)
    return build_competition_prompt(example, mode=prompt_mode)


def run_inference_vllm(
    config: dict[str, Any],
    *,
    input_path: str | Path,
    adapter_dir: str | Path,
    output_path: str | Path,
    max_new_tokens: int | None = None,
    runtime_eval: bool = True,
    llm_kwargs_override: dict[str, Any] | None = None,
    sampling_kwargs_override: dict[str, Any] | None = None,
    save_raw_generations: bool | None = None,
) -> Path:
    """Run batched vLLM inference and write a prediction JSONL.

    The output schema matches :func:`src.student.inference.run_inference` so
    downstream evaluators (``stage2_proxy_runtime_eval`` and friends) do not
    need to branch on the backend.

    Args:
        config: Training/inference config dict (same shape as HF path). The
            ``inference`` sub-dict is consulted for ``prompt_mode`` and
            ``save_raw_generations``.
        input_path: Canonical JSONL with rows loadable via
            :meth:`PuzzleExample.from_dict`.
        adapter_dir: LoRA adapter directory (must contain
            ``adapter_config.json`` and ``adapter_model.safetensors``).
        output_path: Where to write the prediction JSONL.
        max_new_tokens: Optional override for ``max_tokens``; when ``None``
            the runtime-contract default (7680) is used.
        runtime_eval: Must be ``True`` for now; kept as a parameter for
            parity with ``run_inference``.
        llm_kwargs_override: Shallow-merged on top of
            ``RUNTIME_LLM_KWARGS``. Useful for smoke tests that need
            ``max_model_len=2048`` or similar.
        sampling_kwargs_override: Shallow-merged on top of
            ``RUNTIME_SAMPLING_KWARGS``.
        save_raw_generations: If ``True`` (or when the config sets it),
            ``raw_generation`` is included in each row.

    Returns:
        The ``output_path`` as a :class:`Path`.
    """
    if not runtime_eval:
        raise NotImplementedError(
            "inference_vllm only supports runtime_eval=True (the authoritative "
            "LB-aligned contract). Use src.student.inference for notebook-default "
            "or raw_with_guard modes."
        )

    inference_config = dict(config.get("inference", {}))
    prompt_mode = str(
        inference_config.get("prompt_mode", PROMPT_MODE_CHAT_THINKING)
    )
    if save_raw_generations is None:
        save_raw_generations = bool(
            inference_config.get("save_raw_generations", False)
        )

    llm_kwargs = dict(RUNTIME_LLM_KWARGS)
    if llm_kwargs_override:
        llm_kwargs.update(llm_kwargs_override)

    sampling_kwargs = dict(RUNTIME_SAMPLING_KWARGS)
    if sampling_kwargs_override:
        sampling_kwargs.update(sampling_kwargs_override)
    if max_new_tokens is not None:
        sampling_kwargs["max_tokens"] = int(max_new_tokens)

    validate_adapter_dir(adapter_dir)
    base_model_path = resolve_model_path(config)

    examples = [PuzzleExample.from_dict(row) for row in load_jsonl(input_path)]

    llm = _instantiate_vllm(str(base_model_path), llm_kwargs)
    tokenizer = llm.get_tokenizer()
    prompts = [
        _build_prompt_for_inference(example, prompt_mode=prompt_mode, tokenizer=tokenizer)
        for example in examples
    ]
    sampling_params = _build_sampling_params(sampling_kwargs)
    lora_request = _build_lora_request(adapter_dir)

    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    rows: list[dict[str, Any]] = []
    for example, output in zip(examples, outputs):
        choice = output.outputs[0]
        raw_generation = choice.text
        prediction = postprocess_generation(raw_generation)
        row: dict[str, Any] = {"id": example.id, "prediction": prediction}
        if save_raw_generations:
            row["raw_generation"] = raw_generation
        rows.append(row)

    write_jsonl(output_path, rows)
    return Path(output_path)


# Backwards-compatible alias so call sites can swap ``run_inference`` with
# ``run_inference_vllm`` by import path alone.
run_inference = run_inference_vllm


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batched vLLM inference mirroring src.student.inference output "
            "schema. Uses the runtime contract (greedy, max_tokens=7680) by "
            "default."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--config", default="configs/train_lora.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument(
        "--output",
        default="data/processed/inference_predictions.jsonl",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help=(
            "Override the runtime contract's max_tokens (7680). Intended "
            "for smoke tests only; production eval should leave this at "
            "the contract default."
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        help=(
            "Override max_model_len. Contract default is 8192; smoke tests "
            "can drop this to reduce KV-cache footprint on small GPUs."
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="Override gpu_memory_utilization (contract default 0.85).",
    )
    parser.add_argument(
        "--save-raw-generations",
        action="store_true",
        help="Include the full generation text alongside the boxed answer.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    config = read_yaml(args.config)

    llm_kwargs_override: dict[str, Any] = {}
    if args.max_model_len is not None:
        llm_kwargs_override["max_model_len"] = int(args.max_model_len)
    if args.gpu_memory_utilization is not None:
        llm_kwargs_override["gpu_memory_utilization"] = float(
            args.gpu_memory_utilization
        )

    run_inference_vllm(
        config,
        input_path=args.input,
        adapter_dir=args.adapter_dir,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        runtime_eval=True,
        llm_kwargs_override=llm_kwargs_override or None,
        save_raw_generations=bool(args.save_raw_generations),
    )


if __name__ == "__main__":
    sys.exit(main())
