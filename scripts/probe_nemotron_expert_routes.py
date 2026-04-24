from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import load_jsonl, read_yaml, write_json  # noqa: E402
from src.competition.official_metric_contract import build_official_prompt  # noqa: E402
from src.student.lora_train import resolve_model_path  # noqa: E402


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def _layer_from_name(name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return None if match is None else int(match.group(1))


def _read_inputs(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = load_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _summarize_route_counts(
    *,
    module_counts: dict[str, Counter[int]],
    module_top1_counts: dict[str, Counter[int]],
    module_token_events: Counter[str],
    module_calls: Counter[str],
) -> dict[str, Any]:
    modules: list[dict[str, Any]] = []
    by_layer: dict[int, Counter[int]] = defaultdict(Counter)
    by_layer_top1: dict[int, Counter[int]] = defaultdict(Counter)
    for name, counts in sorted(module_counts.items()):
        layer = _layer_from_name(name)
        if layer is not None:
            by_layer[layer].update(counts)
            by_layer_top1[layer].update(module_top1_counts[name])
        total = sum(counts.values())
        modules.append(
            {
                "module": name,
                "layer": layer,
                "calls": int(module_calls[name]),
                "token_events": int(module_token_events[name]),
                "topk_total": int(total),
                "topk_counts": {str(k): int(v) for k, v in counts.most_common()},
                "top1_counts": {
                    str(k): int(v)
                    for k, v in module_top1_counts[name].most_common()
                },
                "topk_head": [
                    {"expert": int(k), "count": int(v), "fraction": float(v / total)}
                    for k, v in counts.most_common(16)
                ]
                if total
                else [],
            }
        )

    layers = []
    for layer, counts in sorted(by_layer.items()):
        total = sum(counts.values())
        layers.append(
            {
                "layer": int(layer),
                "topk_total": int(total),
                "topk_counts": {str(k): int(v) for k, v in counts.most_common()},
                "top1_counts": {
                    str(k): int(v)
                    for k, v in by_layer_top1[layer].most_common()
                },
                "topk_head": [
                    {"expert": int(k), "count": int(v), "fraction": float(v / total)}
                    for k, v in counts.most_common(16)
                ]
                if total
                else [],
            }
        )
    return {"modules": modules, "layers": layers}


def _counter_from_counts(raw: dict[str, Any]) -> Counter[int]:
    counter: Counter[int] = Counter()
    for key, value in raw.items():
        count = int(value)
        if count > 0:
            counter[int(key)] += count
    return counter


def _sum_probe_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    module_counts: dict[str, Counter[int]] = defaultdict(Counter)
    module_top1_counts: dict[str, Counter[int]] = defaultdict(Counter)
    module_token_events: Counter[str] = Counter()
    module_calls: Counter[str] = Counter()
    for payload in payloads:
        for row in payload.get("modules", []):
            name = str(row["module"])
            module_counts[name].update(_counter_from_counts(row.get("topk_counts") or {}))
            module_top1_counts[name].update(_counter_from_counts(row.get("top1_counts") or {}))
            module_token_events[name] += int(row.get("token_events") or 0)
            module_calls[name] += int(row.get("calls") or 0)
    return _summarize_route_counts(
        module_counts=module_counts,
        module_top1_counts=module_top1_counts,
        module_token_events=module_token_events,
        module_calls=module_calls,
    )


def _subtract_probe_payload(total: dict[str, Any], minus: dict[str, Any]) -> dict[str, Any]:
    minus_by_module = {str(row["module"]): row for row in minus.get("modules", [])}
    module_counts: dict[str, Counter[int]] = defaultdict(Counter)
    module_top1_counts: dict[str, Counter[int]] = defaultdict(Counter)
    module_token_events: Counter[str] = Counter()
    module_calls: Counter[str] = Counter()

    for row in total.get("modules", []):
        name = str(row["module"])
        remove = minus_by_module.get(name, {})
        topk = _counter_from_counts(row.get("topk_counts") or {})
        top1 = _counter_from_counts(row.get("top1_counts") or {})
        topk.subtract(_counter_from_counts(remove.get("topk_counts") or {}))
        top1.subtract(_counter_from_counts(remove.get("top1_counts") or {}))
        module_counts[name].update({k: max(0, int(v)) for k, v in topk.items() if v > 0})
        module_top1_counts[name].update({k: max(0, int(v)) for k, v in top1.items() if v > 0})
        module_token_events[name] += max(
            0, int(row.get("token_events") or 0) - int(remove.get("token_events") or 0)
        )
        module_calls[name] += max(
            0, int(row.get("calls") or 0) - int(remove.get("calls") or 0)
        )

    return _summarize_route_counts(
        module_counts=module_counts,
        module_top1_counts=module_top1_counts,
        module_token_events=module_token_events,
        module_calls=module_calls,
    )


class ExpertRouteProbe:
    def __init__(self, *, num_experts: int, top_k: int) -> None:
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.module_counts: dict[str, Counter[int]] = defaultdict(Counter)
        self.module_top1_counts: dict[str, Counter[int]] = defaultdict(Counter)
        self.module_token_events: Counter[str] = Counter()
        self.module_calls: Counter[str] = Counter()
        self.handles: list[Any] = []

    def reset(self) -> None:
        self.module_counts.clear()
        self.module_top1_counts.clear()
        self.module_token_events.clear()
        self.module_calls.clear()

    def _hook(self, name: str):
        def handle(module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if hasattr(module, "n_routed_experts") and isinstance(output, (tuple, list)) and output:
                indices = output[0]
                if isinstance(indices, torch.Tensor) and not indices.is_floating_point():
                    flat = indices.detach().long().reshape(-1)
                    if flat.numel() > 0:
                        valid = flat[(flat >= 0) & (flat < self.num_experts)].cpu()
                        if valid.numel() > 0:
                            top1 = indices.detach().long().reshape(-1, int(indices.shape[-1]))[:, 0].cpu()
                            self.module_calls[name] += 1
                            self.module_token_events[name] += int(indices.reshape(-1, int(indices.shape[-1])).shape[0])
                            self.module_counts[name].update(int(x) for x in valid.tolist())
                            self.module_top1_counts[name].update(
                                int(x) for x in top1[(top1 >= 0) & (top1 < self.num_experts)].tolist()
                            )
                            return

            candidates = []
            for tensor in _iter_tensors(output):
                if tensor.ndim < 2:
                    continue
                if int(tensor.shape[-1]) != self.num_experts:
                    continue
                if not tensor.is_floating_point():
                    continue
                candidates.append(tensor)
            if not candidates:
                return
            # Count only the first router-shaped tensor emitted by a module to
            # avoid double-counting modules that return (logits, probs).
            scores = candidates[0].detach().float().reshape(-1, self.num_experts)
            if scores.numel() == 0:
                return
            k = min(self.top_k, self.num_experts)
            top = torch.topk(scores, k=k, dim=-1).indices.cpu()
            top1 = top[:, 0]
            self.module_calls[name] += 1
            self.module_token_events[name] += int(top.shape[0])
            self.module_top1_counts[name].update(int(x) for x in top1.tolist())
            self.module_counts[name].update(int(x) for x in top.reshape(-1).tolist())

        return handle

    def attach(self, model: Any) -> None:
        for name, module in model.named_modules():
            # Start broad but not reckless: router-shaped outputs are filtered
            # inside the hook, and names keep the result auditable.
            lowered = name.lower()
            if not any(token in lowered for token in ("router", "gate", "moe", "expert")):
                continue
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def to_json(self) -> dict[str, Any]:
        return _summarize_route_counts(
            module_counts=self.module_counts,
            module_top1_counts=self.module_top1_counts,
            module_token_events=self.module_token_events,
            module_calls=self.module_calls,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Nemotron MoE router outputs and count selected experts."
    )
    parser.add_argument("--config", default="configs/train_stage2_thin.yaml")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--limit", type=int, default=35)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument(
        "--count-scope",
        choices=["auto", "prompt", "prompt_and_generation", "generation_delta"],
        default="auto",
        help=(
            "What route events to count. generation_delta runs one prompt-only pass "
            "and subtracts it from a prompt+generation pass as an approximate diagnostic."
        ),
    )
    parser.add_argument(
        "--record-examples",
        action="store_true",
        help="Include per-row module/layer route counts and aggregate from those rows.",
    )
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
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
    rows = _read_inputs(args.input, limit=args.limit)
    prompt_token_lengths: list[int] = []
    examples: list[dict[str, Any]] = []

    def run_prompt(prompt_batch: Any) -> None:
        model(**prompt_batch, use_cache=False)

    def run_prompt_and_generation(prompt_batch: Any) -> None:
        model.generate(
            **prompt_batch,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    def run_selected_scope(prompt_batch: Any) -> dict[str, Any] | None:
        scope = args.count_scope
        if scope == "auto":
            scope = "prompt_and_generation" if args.max_new_tokens > 0 else "prompt"
        if scope == "prompt_and_generation" and args.max_new_tokens <= 0:
            scope = "prompt"

        if args.record_examples:
            probe.reset()
        if scope == "prompt":
            run_prompt(prompt_batch)
            return probe.to_json() if args.record_examples else None
        if scope == "prompt_and_generation":
            run_prompt_and_generation(prompt_batch)
            return probe.to_json() if args.record_examples else None
        if scope == "generation_delta":
            if args.max_new_tokens <= 0:
                raise ValueError("--count-scope generation_delta requires --max-new-tokens > 0")
            probe.reset()
            run_prompt(prompt_batch)
            prompt_payload = probe.to_json()
            probe.reset()
            run_prompt_and_generation(prompt_batch)
            total_payload = probe.to_json()
            return _subtract_probe_payload(total_payload, prompt_payload)
        raise AssertionError(scope)

    try:
        for row_index, row in enumerate(rows):
            raw_prompt = str(row.get("raw_prompt") or "")
            if raw_prompt:
                prompt = build_official_prompt(raw_prompt, tokenizer)
            else:
                prompt = str(row.get("prompt", ""))
                if not prompt.lstrip().startswith("<|im_start|>"):
                    prompt = build_official_prompt(prompt, tokenizer)
            batch = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_token_lengths.append(int(batch["input_ids"].shape[-1]))
            with torch.inference_mode():
                example_payload = run_selected_scope(batch)
            if args.record_examples and example_payload is not None:
                examples.append(
                    {
                        "row_index": row_index,
                        "row_id": row.get("id"),
                        "prompt_token_length": prompt_token_lengths[-1],
                        **example_payload,
                    }
                )
    finally:
        probe.close()

    route_payload = _sum_probe_payloads(examples) if args.record_examples else probe.to_json()
    payload = {
        "input": str(args.input),
        "config": str(args.config),
        "adapter_dir": args.adapter_dir,
        "num_rows": len(rows),
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "count_scope": args.count_scope,
        "record_examples": args.record_examples,
        "prompt_token_lengths": prompt_token_lengths,
        **route_payload,
    }
    if args.record_examples:
        payload["examples"] = examples
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
