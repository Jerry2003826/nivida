from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import load_jsonl, read_yaml, write_json, write_jsonl
from src.competition.official_metric_contract import (
    OFFICIAL_LLM_KWARGS,
    OFFICIAL_SAMPLING_KWARGS,
    RUNTIME_LLM_KWARGS,
    RUNTIME_SAMPLING_KWARGS,
    RUNTIME_CONTRACT_SOURCE,
    build_official_prompt,
    current_contract_fingerprint,
    extract_final_answer,
    verify,
)
from src.student.lora_train import resolve_model_path
from src.student.package_submission import validate_adapter_dir


def _select_llm_kwargs(contract: str) -> dict[str, Any]:
    if contract == "runtime":
        return dict(RUNTIME_LLM_KWARGS)
    if contract == "notebook":
        return dict(OFFICIAL_LLM_KWARGS)
    raise ValueError(
        f"Unknown eval contract {contract!r}. Expected 'runtime' or 'notebook'."
    )


def _select_sampling_kwargs(contract: str) -> dict[str, Any]:
    if contract == "runtime":
        return dict(RUNTIME_SAMPLING_KWARGS)
    if contract == "notebook":
        return dict(OFFICIAL_SAMPLING_KWARGS)
    raise ValueError(
        f"Unknown eval contract {contract!r}. Expected 'runtime' or 'notebook'."
    )


def _instantiate_vllm(base_model_path: str, llm_kwargs: dict[str, Any]):
    from vllm import LLM

    return LLM(model=str(base_model_path), **llm_kwargs)


def _build_sampling_params(sampling_kwargs: dict[str, Any]):
    from vllm import SamplingParams

    assert "seed" not in sampling_kwargs
    assert "stop" not in sampling_kwargs
    assert "stop_token_ids" not in sampling_kwargs
    return SamplingParams(**sampling_kwargs)


def _build_lora_request(adapter_dir: str | Path):
    from vllm.lora.request import LoRARequest

    return LoRARequest("adapter", 1, str(adapter_dir))


def _percentile(lengths: list[int], percentile: float) -> int:
    if not lengths:
        return 0
    ordered = sorted(lengths)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _repeat_stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean_value = sum(values) / len(values)
    if len(values) == 1:
        return mean_value, 0.0, 0.0
    std_value = statistics.stdev(values)
    return mean_value, std_value, std_value / math.sqrt(len(values))


def _by_family_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[bool]] = {}
    for row in rows:
        family = str(row.get("family", "unknown"))
        buckets.setdefault(family, []).append(bool(row["competition_correct"]))
    return {
        family: sum(values) / len(values)
        for family, values in sorted(buckets.items())
    }


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _row_family(row: dict[str, Any]) -> str:
    meta = _metadata(row)
    value = (
        row.get("official_family")
        or row.get("family")
        or meta.get("official_family")
        or meta.get("family")
    )
    return "unknown" if value is None else str(value)


def evaluate_official_vllm_proxy(
    *,
    adapter_dir: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
    num_repeats: int,
    write_raw_predictions: bool,
    raw_predictions_dir: str | Path | None,
    no_load_base_model: bool,
    contract: str = "runtime",
) -> dict[str, Any]:
    config = read_yaml(config_path)
    rows = load_jsonl(input_path)

    llm_kwargs = _select_llm_kwargs(contract)
    sampling_kwargs = _select_sampling_kwargs(contract)

    if contract == "runtime" and num_repeats > 1 and float(sampling_kwargs["temperature"]) == 0.0:
        # Runtime contract is greedy (temperature=0.0). Repeating greedy
        # generation is deterministic, so num_repeats>1 is pure wasted GPU.
        # Leave a loud warning in the result payload but still run once to
        # avoid silent downstream breakage.
        print(
            f"[eval_official_vllm_proxy] WARNING: contract=runtime is greedy "
            f"(temperature=0.0); requested num_repeats={num_repeats} will be "
            f"executed but yield identical outputs. Forcing num_repeats=1.",
            flush=True,
        )
        num_repeats = 1

    if not no_load_base_model:
        validate_adapter_dir(adapter_dir)
        base_model_path = resolve_model_path(config)
    else:
        base_model_path = "__no_load_base_model__"

    llm = _instantiate_vllm(str(base_model_path), llm_kwargs)
    tokenizer = llm.get_tokenizer()
    prompts = [
        build_official_prompt(str(row.get("prompt", row.get("raw_prompt", ""))), tokenizer)
        for row in rows
    ]
    sampling = _build_sampling_params(sampling_kwargs)
    lora_request = None if no_load_base_model else _build_lora_request(adapter_dir)

    raw_dir = None if raw_predictions_dir is None else Path(raw_predictions_dir)
    if write_raw_predictions and raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    repeat_payloads: list[dict[str, Any]] = []
    family_names = sorted({_row_family(row) for row in rows})

    for repeat_index in range(num_repeats):
        outputs = llm.generate(prompts, sampling_params=sampling, lora_request=lora_request)
        repeat_rows: list[dict[str, Any]] = []
        token_lengths: list[int] = []
        num_boxed_found = 0
        num_fallback_used = 0
        num_correct = 0

        for row, output in zip(rows, outputs):
            choice = output.outputs[0]
            generation = choice.text
            token_count = len(getattr(choice, "token_ids", []) or [])
            extracted_answer = extract_final_answer(generation)
            boxed_found = "\\boxed{" in generation
            fallback_used = not boxed_found
            competition_correct = verify(str(row.get("target_answer", "")), str(extracted_answer))

            if boxed_found:
                num_boxed_found += 1
            if fallback_used:
                num_fallback_used += 1
            if competition_correct:
                num_correct += 1
            token_lengths.append(token_count)

            repeat_rows.append(
                {
                    "id": str(row.get("id", "")),
                    "family": _row_family(row),
                    "prompt": str(row.get("prompt", row.get("raw_prompt", ""))),
                    "target_answer": str(row.get("target_answer", "")),
                    "generation": generation,
                    "token_count": token_count,
                    "hit_max_tokens": token_count >= int(sampling_kwargs["max_tokens"]),
                    "extracted_answer": extracted_answer,
                    "fallback_used": fallback_used,
                    "competition_correct": competition_correct,
                }
            )

        raw_predictions_path = None
        if write_raw_predictions and raw_dir is not None:
            raw_predictions_path = raw_dir / f"repeat_{repeat_index}.jsonl"
            write_jsonl(raw_predictions_path, repeat_rows)

        hit_max_tokens = sum(row["hit_max_tokens"] for row in repeat_rows)
        repeat_payloads.append(
            {
                "repeat_index": repeat_index,
                "competition_correct_rate": 0.0 if not repeat_rows else num_correct / len(repeat_rows),
                "num_correct": num_correct,
                "by_family_accuracy": _by_family_accuracy(repeat_rows),
                "generation_length_p50": _percentile(token_lengths, 0.50),
                "generation_length_p95": _percentile(token_lengths, 0.95),
                "generation_length_p99": _percentile(token_lengths, 0.99),
                "num_hit_max_tokens": hit_max_tokens,
                "truncate_rate": 0.0 if not repeat_rows else hit_max_tokens / len(repeat_rows),
                "mean_generation_length": 0.0 if not token_lengths else sum(token_lengths) / len(token_lengths),
                "num_boxed_found": num_boxed_found,
                "num_fallback_used": num_fallback_used,
                "boxed_rate": 0.0 if not repeat_rows else num_boxed_found / len(repeat_rows),
                "raw_predictions_path": None if raw_predictions_path is None else str(raw_predictions_path),
            }
        )

    competition_rates = [repeat["competition_correct_rate"] for repeat in repeat_payloads]
    mean_rate, std_rate, stderr_rate = _repeat_stats(competition_rates)
    by_family_mean: dict[str, float] = {}
    by_family_std: dict[str, float] = {}
    for family in family_names:
        values = [
            repeat["by_family_accuracy"].get(family, 0.0)
            for repeat in repeat_payloads
        ]
        family_mean, family_std, _family_stderr = _repeat_stats(values)
        by_family_mean[family] = family_mean
        by_family_std[family] = family_std

    payload = {
        "adapter_dir": str(adapter_dir),
        "input_path": str(input_path),
        "num_examples": len(rows),
        "num_repeats": num_repeats,
        "eval_contract": contract,
        "eval_contract_source": (
            RUNTIME_CONTRACT_SOURCE if contract == "runtime"
            else "Notebook default (NOT authoritative for LB selection)"
        ),
        "eval_llm_kwargs": llm_kwargs,
        "eval_sampling_kwargs": sampling_kwargs,
        "contract_fingerprint": current_contract_fingerprint().to_dict(),
        "repeats": repeat_payloads,
        "mean_competition_correct_rate": mean_rate,
        "std_competition_correct_rate": std_rate,
        "stderr_competition_correct_rate": stderr_rate,
        "min_competition_correct_rate": min(competition_rates) if competition_rates else 0.0,
        "max_competition_correct_rate": max(competition_rates) if competition_rates else 0.0,
        "by_family_mean_accuracy": by_family_mean,
        "by_family_std_accuracy": by_family_std,
    }
    write_json(output_path, payload)
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an adapter with the official vLLM prompt/sampling contract.",
        allow_abbrev=False,
    )
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--write-raw-predictions", action="store_true")
    parser.add_argument("--raw-predictions-dir")
    parser.add_argument("--no-load-base-model", action="store_true")
    parser.add_argument(
        "--contract",
        choices=["runtime", "notebook"],
        default="runtime",
        help=(
            "Which eval contract to use. 'runtime' (default) mirrors the "
            "Kaggle Overview tab (temperature=0.0, max_tokens=7680, "
            "max_model_len=8192) and is authoritative for LB-correlated "
            "checkpoint selection. 'notebook' mirrors the metric notebook "
            "defaults (temperature=1.0, max_tokens=3584) and is retained "
            "only for legacy parity fingerprinting."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    evaluate_official_vllm_proxy(
        adapter_dir=args.adapter_dir,
        input_path=args.input,
        output_path=args.output,
        config_path=args.config,
        num_repeats=max(1, int(args.num_repeats)),
        write_raw_predictions=bool(args.write_raw_predictions),
        raw_predictions_dir=args.raw_predictions_dir,
        no_load_base_model=bool(args.no_load_base_model),
        contract=args.contract,
    )


if __name__ == "__main__":
    main()
