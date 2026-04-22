"""profile_moe_routing.py
===========================
Instrumentation-only profiler for Nemotron-3-Nano-30B-A3B MoE router.

Purpose
-------
Before considering hot-expert LoRA (deferred from Round 1 per GPT-5.4 Pro
Round 3 decisions), we need empirical evidence of router concentration:

  * For each MoE layer, which experts carry what fraction of the routed
    mass on our competition distribution?
  * Is routing stable enough across prompt families that a "hot" subset
    (top-25% of experts, or top-K by mass80 rule) would cover most of
    the signal?

This script walks 2–5 k prompts from the Nemotron-Math-v2 loader,
captures the router logits / top-k gating weights at every MoE layer,
and emits:

  * ``per_layer_expert_mass.json`` — for each layer, the fraction of
    routed tokens assigned to each expert (both top-1 vote count and
    gating-weight sum).
  * ``per_layer_top25.json`` — the top-25% of experts by routed mass
    per layer, with cumulative coverage.
  * ``per_layer_mass80.json`` — smallest set of experts covering ≥80%
    of routed mass per layer.

These artifacts feed the Round-1-next-milestone decision on whether to
target hot-expert LoRA.  **This script does NOT train, backprop, or
modify any weights.**

Implementation notes
--------------------
* The router itself is frozen; we read the top-k gating weights from the
  model's forward pass using PyTorch hooks.
* We never materialise the full experts output — hooks run on the router
  (usually called ``block_sparse_moe`` or ``router`` or ``gate`` in
  Nemotron).
* Nemotron-3-Nano has 128 routed experts + 1 shared expert per MoE
  layer.  Shared expert is reported separately (always contributes).
* Set ``--num-prompts`` between 2000 and 5000 for statistically stable
  top-25% estimates across 23 MoE layers × 128 experts.

Dry-run
-------
``--dry-run`` loads no model and synthesises router output for 5 fake
prompts × 2 layers × 8 experts — validates the aggregation code.

Usage
-----
::

    # Dry run (no GPU needed)
    python scripts/profile_moe_routing.py --dry-run \\
        --output artifacts/moe_profile_dryrun

    # Full run on GPU with the base model
    python scripts/profile_moe_routing.py \\
        --model /workspace/models/nemotron-3-nano-30b-a3b \\
        --prompts data/processed/round1_nmath_v2.jsonl \\
        --num-prompts 3000 \\
        --output artifacts/moe_profile_round1

Output schema (per layer)
-------------------------
::

    {
      "layer_idx": int,
      "num_experts_routed": int,
      "num_experts_shared": int,
      "expert_token_count": list[float],   # len = num_experts_routed
      "expert_gate_mass":   list[float],   # sum of top-k weights per expert
      "top25_expert_ids":   list[int],
      "top25_cumulative_mass": float,
      "mass80_expert_ids":   list[int],
      "mass80_num_experts":  int,
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Aggregation (pure numpy — no torch needed)
# ---------------------------------------------------------------------------

def aggregate_layer(
    token_counts,     # np.ndarray [num_experts], int32
    gate_mass,        # np.ndarray [num_experts], float32
    mass80_cutoff: float = 0.80,
) -> dict[str, Any]:
    """Compute top-25% and mass80 statistics for a single layer."""
    import numpy as np  # local import

    num_experts = int(gate_mass.shape[0])
    total_mass = float(gate_mass.sum())
    if total_mass <= 0.0:
        total_mass = 1e-12  # avoid divide-by-zero
    mass_frac = gate_mass / total_mass

    order = np.argsort(-gate_mass)  # descending
    sorted_mass = mass_frac[order]
    cumulative = np.cumsum(sorted_mass)

    top25_count = max(1, int(round(num_experts * 0.25)))
    top25_ids = [int(i) for i in order[:top25_count]]
    top25_cum = float(cumulative[top25_count - 1])

    mass80_pos = int(np.searchsorted(cumulative, mass80_cutoff) + 1)
    mass80_pos = min(mass80_pos, num_experts)
    mass80_ids = [int(i) for i in order[:mass80_pos]]

    return {
        "num_experts_routed": num_experts,
        "expert_token_count": [int(c) for c in token_counts.tolist()],
        "expert_gate_mass": [float(m) for m in gate_mass.tolist()],
        "top25_expert_ids": top25_ids,
        "top25_cumulative_mass": top25_cum,
        "mass80_expert_ids": mass80_ids,
        "mass80_num_experts": mass80_pos,
    }


# ---------------------------------------------------------------------------
# Torch / model integration (lazy import so dry-run works without torch)
# ---------------------------------------------------------------------------

def _install_router_hooks(
    model,                 # transformers.PreTrainedModel
    expert_counts: dict[int, "np.ndarray"],  # will be populated in-place
    expert_mass: dict[int, "np.ndarray"],
) -> list:
    """Attach forward hooks on each MoE layer's router.

    The exact attribute name for the router varies by HF implementation.
    For Nemotron-3-Nano we expect ``model.layers[i].block_sparse_moe.gate``
    or ``model.layers[i].mlp.router`` — the probe below covers both.

    Each hook receives the router's output (top-k weights and indices),
    accumulates:
      * token count per expert  (from top-1 index)
      * gate-weight mass per expert (sum of top-k weights across tokens)

    Returns the list of hook handles (caller must call .remove()).
    """
    # pylint: disable=import-outside-toplevel
    import numpy as np
    import torch  # noqa: F401

    handles = []

    def make_hook(layer_idx: int, num_experts: int):
        # Lazily initialise accumulators
        expert_counts[layer_idx] = np.zeros(num_experts, dtype=np.int64)
        expert_mass[layer_idx]   = np.zeros(num_experts, dtype=np.float64)

        def _hook(_module, _inputs, output):
            # Most HF MoE routers return (weights, indices) or a tuple
            # containing them; we attempt both shapes.
            if isinstance(output, tuple) and len(output) >= 2:
                weights, indices = output[0], output[1]
            else:
                # Some routers return a dict
                weights = output.get("routing_weights", None)
                indices = output.get("expert_indices", None)
            if weights is None or indices is None:
                return
            # weights: [B, T, top_k]; indices: [B, T, top_k]
            w_np = weights.detach().to("cpu").float().numpy().reshape(-1)
            i_np = indices.detach().to("cpu").long().numpy().reshape(-1)
            # Aggregate gate mass per expert
            np.add.at(expert_mass[layer_idx], i_np, w_np)
            # Top-1 token count — use only the first top-k slot
            # weights shape reshape: [B*T, top_k]
            topk = indices.shape[-1]
            top1_flat = i_np.reshape(-1, topk)[:, 0]
            counts = np.bincount(top1_flat, minlength=num_experts)
            expert_counts[layer_idx] += counts

        return _hook

    # Walk modules looking for MoE blocks
    moe_layer_idx = 0
    for _, module in model.named_modules():
        # Heuristic: router modules expose a `top_k` attribute and a
        # `num_experts` (or `num_local_experts`) field.
        if hasattr(module, "top_k") and (
            hasattr(module, "num_experts") or hasattr(module, "num_local_experts")
        ):
            num_experts = getattr(
                module, "num_experts", getattr(module, "num_local_experts", None)
            )
            if num_experts is None:
                continue
            h = module.register_forward_hook(make_hook(moe_layer_idx, int(num_experts)))
            handles.append(h)
            moe_layer_idx += 1
    print(
        f"[profile_moe_routing] installed hooks on {moe_layer_idx} MoE layers",
        flush=True,
    )
    return handles


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def _dry_run(output_dir: Path) -> None:
    """Synthesise 5 fake prompts × 2 layers × 8 experts to validate
    aggregation + report output."""
    import numpy as np  # local import

    output_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic heavy-tailed mass: 3 of 8 experts get 90% of mass
    layer0_counts = np.array([40, 30, 30, 5, 3, 2, 2, 1], dtype=np.int64)
    layer0_mass = np.array(
        [0.30, 0.25, 0.20, 0.08, 0.07, 0.05, 0.03, 0.02], dtype=np.float64
    )
    layer1_counts = np.array([10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int64)
    layer1_mass = np.array(
        [0.125] * 8, dtype=np.float64
    )

    reports = [
        {"layer_idx": 0, **aggregate_layer(layer0_counts, layer0_mass)},
        {"layer_idx": 1, **aggregate_layer(layer1_counts, layer1_mass)},
    ]
    (output_dir / "per_layer_expert_mass.json").write_text(
        json.dumps(reports, indent=2)
    )
    print(
        f"[profile_moe_routing] dry-run: wrote {len(reports)} layer reports "
        f"to {output_dir}",
        flush=True,
    )
    # Sanity: layer 0's top-25% (2 of 8) should capture > layer 1's top-25%
    assert reports[0]["top25_cumulative_mass"] > reports[1]["top25_cumulative_mass"]
    assert reports[0]["mass80_num_experts"] < reports[1]["mass80_num_experts"]
    print("[profile_moe_routing] OK: dry-run assertions passed.", flush=True)


# ---------------------------------------------------------------------------
# Main run (stub: integration point for vLLM / HF forward pass)
# ---------------------------------------------------------------------------

def run_profile(
    model_path: str,
    prompts_jsonl: Path,
    num_prompts: int,
    output_dir: Path,
) -> None:
    """Run the profiler on *num_prompts* from *prompts_jsonl*.

    # TODO: integration point —
    #   1. Load tokenizer and HF model (device_map='auto', torch_dtype=bf16).
    #   2. Install hooks via _install_router_hooks().
    #   3. For each prompt, build the official chat-template input, run
    #      ``model.generate(max_new_tokens=256, do_sample=False)`` once
    #      (256 tokens is enough to profile routing without wasting compute).
    #   4. Remove hooks, serialise expert_counts / expert_mass per layer.
    """
    raise NotImplementedError(
        "run_profile() is not wired yet — see TODO comment.  Use --dry-run "
        "for now; full model integration lands in a follow-up commit."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Instrumentation-only MoE router profiler for Nemotron-3-Nano. "
            "Emits top-25% and mass80 expert sets per MoE layer.  Does NOT "
            "train or modify weights."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--model", default=None, help="Path to HF model dir.")
    parser.add_argument(
        "--prompts",
        default=None,
        help="JSONL with input prompts (output of round1_prepare_nemotron_math_v2.py).",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=3000,
        dest="num_prompts",
        help="Number of prompts to profile (default 3000).",
    )
    parser.add_argument(
        "--output",
        default="artifacts/moe_profile_round1",
        help="Output directory for per-layer reports.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Run numpy-only aggregation self-test.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output)

    if args.dry_run:
        _dry_run(output_dir)
        return

    if args.model is None or args.prompts is None:
        raise SystemExit(
            "--model and --prompts are required unless --dry-run is set."
        )
    run_profile(
        model_path=args.model,
        prompts_jsonl=Path(args.prompts),
        num_prompts=args.num_prompts,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
