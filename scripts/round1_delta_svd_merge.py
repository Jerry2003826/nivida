"""round1_delta_svd_merge.py
=================================
Delta-space SVD merge for LoRA checkpoints.

Purpose
-------
Produce a single **rank-32 LoRA adapter** that merges multiple independently
fine-tuned LoRA checkpoints (e.g. Stage-2-bestproxy and Stage-3-longtrace-
repair) via **delta-space** SVD truncation.

This is the adapter used for Kaggle submit #2 (the "merged" candidate)
alongside submit #1 (the best single checkpoint).  Expected uplift per
GPT-5.4 Pro Round 3 research: +0.005 – +0.02 LB over the best single
adapter.

Why delta-space SVD (and not raw A/B averaging, TIES, or DARE)
----------------------------------------------------------------
* Raw A/B averaging is algebraically wrong: ``(A1 + A2)/2 * (B1 + B2)/2``
  is NOT equal to ``(A1 B1 + A2 B2)/2``.  The cross-terms inflate rank
  and distort directions.
* TIES/DARE target full-model deltas and scale-invariant parameters;
  LoRA adapters are already low-rank additive deltas, so the magnitude
  pruning step does more harm than good in our rank regime.
* Delta-space SVD is exact up to the rank truncation error:
    1. For each adapter i, compute the full delta
       ``ΔW_i = B_i @ A_i``  (shape [out, in])
    2. Sum the deltas: ``ΔW = Σ_i w_i * ΔW_i``  (weights w_i configurable;
       defaults to uniform 1/N).
    3. Compute a rank-R truncated SVD of ΔW:
       ``ΔW ≈ U[:, :R] * Σ[:R] * V^T[:R, :]``.
    4. Factorise back into LoRA A/B at rank R:
       ``B = U[:, :R] * sqrt(Σ[:R])``,  ``A = sqrt(Σ[:R]) * V^T[:R, :]``.
       The resulting (A, B) satisfies ``B @ A = ΔW_truncated`` exactly.

The rank-R ceiling matches the competition's LoRA rank ≤ 32 rule.

Usage
-----
::

    # Merge two adapters, uniform weights, rank 32 (competition max):
    python scripts/round1_delta_svd_merge.py \\
        --input artifacts/adapter_stage2_bestproxy \\
                artifacts/adapter_stage3_bestproxy \\
        --output artifacts/adapter_merged_svd32 \\
        --rank 32

    # Weighted merge (stage3 counts 1.5x):
    python scripts/round1_delta_svd_merge.py \\
        --input  artifacts/adapter_stage2_bestproxy \\
                 artifacts/adapter_stage3_bestproxy \\
        --weights 1.0 1.5 \\
        --output artifacts/adapter_merged_svd32_w \\
        --rank 32

    # Dry run on two 8×4 → 4×16 synthetic adapters (no torch needed):
    python scripts/round1_delta_svd_merge.py --dry-run

Output
------
The output directory mirrors the standard PEFT LoRA format:
  - ``adapter_config.json``    (copied from the first input; rank overridden)
  - ``adapter_model.safetensors``  (merged SVD-rank weights)
  - ``merge_report.json``     (per-layer SV spectra + residual errors)

Only ``adapter_config.json`` + ``adapter_model.safetensors`` are packaged
into submission.zip — matches the Kaggle submission contract.

Limitations / caveats
---------------------
* All input adapters MUST share identical target-module lists and base
  model.  The script asserts this.
* Mixing different LoRA ranks is supported (each is merged in its own
  full-delta space; the only ceiling is the output --rank).
* Scaling factor (alpha / r) from each input is applied before summing;
  the output alpha is set so alpha/r matches the merged effective scale.
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
# Core SVD merge (pure numpy, so it is testable without torch)
# ---------------------------------------------------------------------------

def _svd_rank_truncate(
    delta,  # np.ndarray [out_dim, in_dim]
    rank: int,
):
    """Rank-truncated SVD factorisation.

    Returns (B, A, singular_values) where B @ A ≈ delta and B has rank
    columns, A has rank rows.

    Factorisation: delta ≈ U[:, :r] @ diag(S[:r]) @ Vt[:r, :]
    We split sqrt(S) to both sides so B = U * sqrt(S), A = sqrt(S) * Vt.
    """
    import numpy as np  # local import

    U, S, Vt = np.linalg.svd(delta, full_matrices=False)
    r = min(rank, len(S))
    sqrt_s = np.sqrt(S[:r])
    B = U[:, :r] * sqrt_s[None, :]          # [out, r]
    A = sqrt_s[:, None] * Vt[:r, :]          # [r, in]
    return B, A, S


def _compute_merged_delta(
    deltas: list,                # list of np.ndarray [out, in], one per input adapter
    weights: list[float],
):
    """Weighted sum of per-adapter deltas in the full-delta space."""
    import numpy as np  # local import

    assert len(deltas) == len(weights), "deltas and weights length mismatch"
    stacked = np.stack(
        [w * d for w, d in zip(weights, deltas)], axis=0
    )
    return stacked.sum(axis=0)


def merge_one_layer(
    adapter_AB_list: list[tuple],    # list of (A, B) numpy arrays per adapter
    alpha_over_r_list: list[float],   # per-adapter scaling factor α/r
    weights: list[float],
    out_rank: int,
    out_alpha: float,
) -> dict[str, Any]:
    """Merge a single linear layer's LoRA deltas across adapters.

    Parameters
    ----------
    adapter_AB_list:
        For each input adapter, the ``(A, B)`` LoRA matrices with shapes
        A=[r_i, in_dim] and B=[out_dim, r_i].
    alpha_over_r_list:
        Each adapter's PEFT ``alpha/r`` scaling factor.  Applied as a
        scalar multiplier on the delta before summing.
    weights:
        Merge weights (same length as adapter_AB_list).  Typically all
        1.0 for an unweighted merge.
    out_rank:
        Target rank of the merged adapter.
    out_alpha:
        Target ``alpha`` for the merged adapter.  The effective scaling
        on the merged delta will be ``out_alpha / out_rank``.

    Returns
    -------
    dict with keys:
        merged_A: np.ndarray [out_rank, in_dim]
        merged_B: np.ndarray [out_dim, out_rank]
        residual_frob: float   (Frobenius norm of the reconstruction error)
        spectrum: list[float]  (top-min(rank*2, len(S)) singular values)
    """
    import numpy as np  # local import

    deltas = []
    for (A, B), aor in zip(adapter_AB_list, alpha_over_r_list):
        # PEFT effective delta: (α/r) * B @ A
        deltas.append(float(aor) * (B @ A))

    merged_delta = _compute_merged_delta(deltas, weights)

    # Compensate for the output scaling — the final (A, B) we produce
    # will be multiplied by (out_alpha/out_rank) at inference, so we
    # first un-scale the target delta by that factor.
    out_scale = float(out_alpha) / float(out_rank)
    target = merged_delta / out_scale

    B, A, S = _svd_rank_truncate(target, out_rank)
    reconstructed = B @ A
    residual = float(np.linalg.norm(target - reconstructed))
    keep = min(out_rank * 2, len(S))
    spectrum = [float(s) for s in S[:keep]]

    return {
        "merged_A": A,
        "merged_B": B,
        "residual_frob": residual,
        "spectrum": spectrum,
    }


# ---------------------------------------------------------------------------
# PEFT / safetensors I/O (lazy import so dry-run works without torch)
# ---------------------------------------------------------------------------

def _load_peft_adapter(adapter_dir: Path) -> dict[str, Any]:
    """Load PEFT adapter config + safetensor weights.

    Returns a dict:
        {"config": <adapter_config dict>,
         "weights": {tensor_name: np.ndarray, ...},
         "rank": int, "alpha": float, "target_modules": list[str]}
    """
    import numpy as np  # local import
    from safetensors.numpy import load_file  # local import

    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Missing adapter files in {adapter_dir}: "
            f"need both adapter_config.json and adapter_model.safetensors"
        )
    config = json.loads(config_path.read_text())
    weights = load_file(str(weights_path))
    return {
        "config": config,
        "weights": weights,
        "rank": int(config.get("r", 0) or config.get("lora_r", 0) or 0),
        "alpha": float(
            config.get("lora_alpha", 0.0)
            or config.get("alpha", 0.0)
            or 0.0
        ),
        "target_modules": list(config.get("target_modules", [])),
    }


def _collect_layer_pairs(weights: dict, rank: int) -> dict[str, dict]:
    """Walk the safetensor dict and group ``lora_A``/``lora_B`` tensors per
    layer name.

    PEFT layout:
        base_model.model.<path>.lora_A.weight -> [r, in_dim]
        base_model.model.<path>.lora_B.weight -> [out_dim, r]

    Returns {layer_path: {"A": ndarray, "B": ndarray}} with one entry per
    fused A/B pair.
    """
    pairs: dict[str, dict] = {}
    for name, tensor in weights.items():
        if "lora_A" in name:
            key = name.replace(".lora_A.weight", "").replace(".lora_A", "")
            pairs.setdefault(key, {})["A"] = tensor
        elif "lora_B" in name:
            key = name.replace(".lora_B.weight", "").replace(".lora_B", "")
            pairs.setdefault(key, {})["B"] = tensor
    # Drop incomplete pairs (should not happen in a well-formed adapter)
    return {k: v for k, v in pairs.items() if "A" in v and "B" in v}


def run_merge(
    input_dirs: list[Path],
    output_dir: Path,
    out_rank: int,
    out_alpha: float | None,
    weights: list[float] | None,
) -> dict[str, Any]:
    """End-to-end merge of N LoRA adapters into a single rank-R adapter."""
    import numpy as np  # local import
    from safetensors.numpy import save_file  # local import

    # Load all adapters
    loaded = [_load_peft_adapter(Path(d)) for d in input_dirs]
    if not loaded:
        raise ValueError("No input adapters supplied.")

    # Validate target_modules + base model consistency
    tm0 = sorted(loaded[0]["target_modules"])
    base0 = loaded[0]["config"].get("base_model_name_or_path")
    for i, ad in enumerate(loaded[1:], start=1):
        if sorted(ad["target_modules"]) != tm0:
            raise ValueError(
                f"target_modules mismatch between adapter 0 and {i}: "
                f"{tm0} vs {sorted(ad['target_modules'])}"
            )
        if ad["config"].get("base_model_name_or_path") != base0:
            raise ValueError(
                f"base_model mismatch between adapter 0 and {i}: "
                f"{base0} vs {ad['config'].get('base_model_name_or_path')}"
            )

    if weights is None:
        weights = [1.0] * len(loaded)
    if len(weights) != len(loaded):
        raise ValueError("--weights length must equal --input length")

    # Normalise weights so they sum to the number of adapters (keeps scale
    # close to any single adapter)
    w_sum = sum(weights)
    weights = [w * len(loaded) / w_sum for w in weights]

    # Build per-adapter layer pairs
    per_adapter_pairs = [
        _collect_layer_pairs(ad["weights"], ad["rank"]) for ad in loaded
    ]
    # Intersection of layer names (must be identical across adapters)
    common_layers = set(per_adapter_pairs[0].keys())
    for pairs in per_adapter_pairs[1:]:
        common_layers &= set(pairs.keys())
    if len(common_layers) != len(per_adapter_pairs[0]):
        missing = set(per_adapter_pairs[0].keys()) - common_layers
        print(
            f"[delta_svd_merge] WARNING: {len(missing)} layers missing in some "
            f"adapters, skipping: {sorted(missing)[:5]}... ",
            flush=True,
        )

    alpha_over_r = [ad["alpha"] / max(ad["rank"], 1) for ad in loaded]
    if out_alpha is None:
        # Match the mean scale of the inputs
        out_alpha = out_rank * (sum(alpha_over_r) / len(alpha_over_r))

    merged_weights: dict[str, Any] = {}
    per_layer_report: list[dict[str, Any]] = []

    for layer_key in sorted(common_layers):
        adapter_AB_list = [
            (pairs[layer_key]["A"], pairs[layer_key]["B"])
            for pairs in per_adapter_pairs
        ]
        res = merge_one_layer(
            adapter_AB_list=adapter_AB_list,
            alpha_over_r_list=alpha_over_r,
            weights=weights,
            out_rank=out_rank,
            out_alpha=out_alpha,
        )
        merged_weights[f"{layer_key}.lora_A.weight"] = res["merged_A"].astype(
            np.float32
        )
        merged_weights[f"{layer_key}.lora_B.weight"] = res["merged_B"].astype(
            np.float32
        )
        per_layer_report.append(
            {
                "layer": layer_key,
                "residual_frob": res["residual_frob"],
                "top_spectrum": res["spectrum"][:8],
            }
        )

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_config = dict(loaded[0]["config"])
    out_config["r"] = out_rank
    out_config["lora_alpha"] = out_alpha
    (output_dir / "adapter_config.json").write_text(
        json.dumps(out_config, indent=2)
    )
    save_file(merged_weights, str(output_dir / "adapter_model.safetensors"))
    report = {
        "num_input_adapters": len(loaded),
        "input_dirs": [str(d) for d in input_dirs],
        "weights_normalised": weights,
        "out_rank": out_rank,
        "out_alpha": out_alpha,
        "num_layers_merged": len(per_layer_report),
        "per_layer": per_layer_report[:16],  # truncate for readability
    }
    (output_dir / "merge_report.json").write_text(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# Dry-run / self-test
# ---------------------------------------------------------------------------

def _dry_run() -> None:
    """Run merge_one_layer on two synthetic adapters, verifying the
    factorisation B @ A reconstructs the target delta.

    This validates the numerics without touching safetensors/torch.
    """
    import numpy as np  # local import

    rng = np.random.default_rng(seed=42)
    out_dim, in_dim = 32, 16
    r1, r2 = 8, 8

    A1 = rng.standard_normal((r1, in_dim)).astype(np.float32)
    B1 = rng.standard_normal((out_dim, r1)).astype(np.float32)
    A2 = rng.standard_normal((r2, in_dim)).astype(np.float32)
    B2 = rng.standard_normal((out_dim, r2)).astype(np.float32)

    # simulate alpha/r = 1.0 for both
    res = merge_one_layer(
        adapter_AB_list=[(A1, B1), (A2, B2)],
        alpha_over_r_list=[1.0, 1.0],
        weights=[1.0, 1.0],
        out_rank=8,
        out_alpha=8.0,
    )
    B = res["merged_B"]
    A = res["merged_A"]
    assert A.shape == (8, in_dim), f"A shape wrong: {A.shape}"
    assert B.shape == (out_dim, 8), f"B shape wrong: {B.shape}"

    # Ground truth delta (with out_scale=1.0, same as input)
    expected = B1 @ A1 + B2 @ A2
    reconstructed = B @ A  # because out_alpha/out_rank = 1.0
    err = float(np.linalg.norm(expected - reconstructed))
    print(
        f"[dry-run] rank-8 merge of two rank-8 adapters: "
        f"residual Frobenius norm = {err:.4g}",
        flush=True,
    )
    # Residual error should be moderate (we truncate from rank-16 to rank-8)
    assert err < np.linalg.norm(expected), (
        f"SVD residual ({err}) not smaller than naive zero-reconstruction "
        f"({np.linalg.norm(expected)})"
    )

    # Rank-16 should give a near-exact reconstruction
    res_full = merge_one_layer(
        adapter_AB_list=[(A1, B1), (A2, B2)],
        alpha_over_r_list=[1.0, 1.0],
        weights=[1.0, 1.0],
        out_rank=16,
        out_alpha=16.0,
    )
    B_full = res_full["merged_B"]
    A_full = res_full["merged_A"]
    err_full = float(np.linalg.norm(expected - B_full @ A_full))
    print(
        f"[dry-run] rank-16 merge of two rank-8 adapters: "
        f"residual Frobenius norm = {err_full:.4g} (should be ~0)",
        flush=True,
    )
    assert err_full < 1e-4, f"Full-rank SVD did not reconstruct (err={err_full})"
    print("[dry-run] OK: delta-space SVD factorisation validated.", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Delta-space SVD merge of N PEFT LoRA adapters into a single "
            "rank-R adapter.  Used to produce Kaggle submit #2 (merged)."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help="Two or more adapter directories to merge.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional merge weights, one per --input.  Normalised to sum to "
            "len(input).  Default: uniform 1.0 per adapter."
        ),
    )
    parser.add_argument(
        "--output",
        default="artifacts/adapter_merged_svd32",
        help="Output adapter directory (default: %(default)s).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help=(
            "Output LoRA rank (default 32 = Kaggle rank cap).  Must be ≤ 32 "
            "for competition compliance."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Output LoRA alpha.  Defaults to rank * mean(alpha_i/r_i) so the "
            "effective scaling matches the inputs."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Run numpy-only self-test on synthetic adapters.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.dry_run or args.input is None:
        _dry_run()
        return

    if args.rank > 32:
        raise SystemExit(
            f"--rank {args.rank} exceeds Kaggle LoRA rank cap of 32. "
            "The competition rejects adapters with rank > 32."
        )
    if len(args.input) < 2:
        raise SystemExit("--input must list at least 2 adapter dirs.")

    report = run_merge(
        input_dirs=[Path(p) for p in args.input],
        output_dir=Path(args.output),
        out_rank=args.rank,
        out_alpha=args.alpha,
        weights=args.weights,
    )
    print(
        "[delta_svd_merge] merged "
        f"{report['num_input_adapters']} adapters into "
        f"{args.output} (rank={report['out_rank']}, alpha={report['out_alpha']:.2f}, "
        f"layers={report['num_layers_merged']}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
