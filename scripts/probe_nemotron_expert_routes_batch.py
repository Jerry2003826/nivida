from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import load_jsonl, read_yaml, write_json  # noqa: E402
from src.competition.official_metric_contract import build_official_prompt  # noqa: E402
from src.student.lora_train import resolve_model_path  # noqa: E402

from scripts.probe_nemotron_expert_routes import (  # noqa: E402
    ExpertRouteProbe,
    _subtract_probe_payload,
    _sum_probe_payloads,
)


def _read_inputs(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = load_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _parse_job(text: str) -> dict[str, Any]:
    # name=input:output:limit:scope:max_new_tokens
    name, rest = text.split("=", 1)
    parts = rest.split(":")
    if len(parts) < 3:
        raise ValueError(
            "--job must be name=input:output:limit[:scope[:max_new_tokens]]"
        )
    input_path = parts[0]
    output_path = parts[1]
    limit = None if parts[2].lower() in {"none", "all", ""} else int(parts[2])
    scope = parts[3] if len(parts) >= 4 and parts[3] else "prompt"
    max_new_tokens = int(parts[4]) if len(parts) >= 5 and parts[4] else 0
    return {
        "name": name,
        "input": Path(input_path),
        "output": Path(output_path),
        "limit": limit,
        "count_scope": scope,
        "max_new_tokens": max_new_tokens,
    }


def _build_prompt(row: dict[str, Any], tokenizer: Any) -> str:
    raw_prompt = str(row.get("raw_prompt") or "")
    if raw_prompt:
        return build_official_prompt(raw_prompt, tokenizer)
    prompt = str(row.get("prompt", ""))
    if not prompt.lstrip().startswith("<|im_start|>"):
        return build_official_prompt(prompt, tokenizer)
    return prompt


def _run_prompt(model: Any, batch: Any) -> None:
    model(**batch, use_cache=False)


def _run_generation(model: Any, batch: Any, *, max_new_tokens: int) -> None:
    model.generate(
        **batch,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )


def _run_scope(
    *,
    model: Any,
    probe: ExpertRouteProbe,
    batch: Any,
    count_scope: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    if count_scope == "auto":
        count_scope = "prompt_and_generation" if max_new_tokens > 0 else "prompt"
    if count_scope == "prompt_and_generation" and max_new_tokens <= 0:
        count_scope = "prompt"

    probe.reset()
    if count_scope == "prompt":
        _run_prompt(model, batch)
        return probe.to_json()
    if count_scope == "prompt_and_generation":
        _run_generation(model, batch, max_new_tokens=max_new_tokens)
        return probe.to_json()
    if count_scope == "generation_delta":
        if max_new_tokens <= 0:
            raise ValueError("generation_delta requires max_new_tokens > 0")
        _run_prompt(model, batch)
        prompt_payload = probe.to_json()
        probe.reset()
        _run_generation(model, batch, max_new_tokens=max_new_tokens)
        total_payload = probe.to_json()
        return _subtract_probe_payload(total_payload, prompt_payload)
    raise ValueError(f"unknown count_scope={count_scope!r}")


def _run_job(
    *,
    model: Any,
    tokenizer: Any,
    probe: ExpertRouteProbe,
    job: dict[str, Any],
    config_path: str,
    adapter_dir: str | None,
    num_experts: int,
    top_k: int,
) -> None:
    rows = _read_inputs(job["input"], limit=job["limit"])
    prompt_token_lengths: list[int] = []
    examples: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        prompt = _build_prompt(row, tokenizer)
        batch = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_token_lengths.append(int(batch["input_ids"].shape[-1]))
        with torch.inference_mode():
            example_payload = _run_scope(
                model=model,
                probe=probe,
                batch=batch,
                count_scope=str(job["count_scope"]),
                max_new_tokens=int(job["max_new_tokens"]),
            )
        examples.append(
            {
                "row_index": row_index,
                "row_id": row.get("id"),
                "prompt_token_length": prompt_token_lengths[-1],
                **example_payload,
            }
        )
    route_payload = _sum_probe_payloads(examples)
    payload = {
        "input": str(job["input"]),
        "config": config_path,
        "adapter_dir": adapter_dir,
        "num_rows": len(rows),
        "num_experts": num_experts,
        "top_k": top_k,
        "max_new_tokens": int(job["max_new_tokens"]),
        "count_scope": str(job["count_scope"]),
        "record_examples": True,
        "prompt_token_lengths": prompt_token_lengths,
        **route_payload,
        "examples": examples,
    }
    write_json(job["output"], payload)
    print(json.dumps({"job": job["name"], "output": str(job["output"]), "num_rows": len(rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple Nemotron route probes after loading model+adapter once."
    )
    parser.add_argument("--config", default="configs/train_stage2_thin.yaml")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--job",
        action="append",
        required=True,
        help="name=input:output:limit[:scope[:max_new_tokens]]. Repeatable.",
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = read_yaml(args.config)
    model_path = resolve_model_path(config)
    tokenizer_path = config.get("tokenizer_path") or model_path
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    if args.adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
    model.eval()

    probe = ExpertRouteProbe(num_experts=args.num_experts, top_k=args.top_k)
    probe.attach(model)
    try:
        for job_text in args.job:
            _run_job(
                model=model,
                tokenizer=tokenizer,
                probe=probe,
                job=_parse_job(job_text),
                config_path=args.config,
                adapter_dir=args.adapter_dir,
                num_experts=args.num_experts,
                top_k=args.top_k,
            )
    finally:
        probe.close()


if __name__ == "__main__":
    main()
