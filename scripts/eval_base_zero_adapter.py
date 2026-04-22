"""Baseline proxy eval: base Nemotron model with an identity (zero) adapter.

Purpose
-------
When stage2 finishes we have no idea whether our LoRA adapter actually
*helps* on the proxy set. We have been comparing checkpoint-vs-checkpoint
deltas, but the absolute floor — "how well does the base model do under the
Kaggle runtime contract when the adapter contributes nothing?" — has never
been measured for this pipeline. The historical 0.54 Kaggle LB was an early
smoke, not a rigorous base-zero baseline.

This script creates a deterministic zero-initialised LoRA adapter that is
structurally identical to the real one (same rank, same target modules,
same rank pattern) and evaluates it through the same vLLM harness used for
every production checkpoint. Any finetuned adapter that cannot beat this
baseline is, by definition, worse than no adapter.

Design choices
--------------
* The adapter weights are all zeros → ``x + B @ A @ x == x``. vLLM happily
  accepts this and applies the LoRA; the base model is observationally
  unchanged.
* We copy the real adapter's ``adapter_config.json`` verbatim so rank,
  target modules, alpha, and any target-modules-specific metadata match.
  This keeps the eval identical to the real-adapter path in every way
  except the weights.
* We call :func:`scripts.eval_official_vllm_proxy.evaluate_official_vllm_proxy`
  directly so the contract / payload schema is identical to a normal
  checkpoint eval.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_official_vllm_proxy import evaluate_official_vllm_proxy  # noqa: E402
from src.student.package_submission import validate_adapter_dir  # noqa: E402


def _resolve_reference_adapter(reference_adapter_dir: Path) -> Path:
    """Validate the reference adapter has the files we need."""
    validate_adapter_dir(str(reference_adapter_dir))
    cfg = reference_adapter_dir / "adapter_config.json"
    weights_st = reference_adapter_dir / "adapter_model.safetensors"
    weights_bin = reference_adapter_dir / "adapter_model.bin"
    if not cfg.exists():
        raise SystemExit(
            f"reference adapter at {reference_adapter_dir} has no adapter_config.json"
        )
    if not (weights_st.exists() or weights_bin.exists()):
        raise SystemExit(
            f"reference adapter at {reference_adapter_dir} has no adapter_model weights"
        )
    return reference_adapter_dir


def _zero_out_safetensors(source_path: Path, dest_path: Path) -> None:
    """Load a safetensors file, zero every tensor, rewrite to ``dest_path``."""
    import torch  # local import: heavy optional dep
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(source_path))
    zeroed = {
        name: torch.zeros_like(tensor) for name, tensor in tensors.items()
    }
    save_file(zeroed, str(dest_path))


def _zero_out_bin(source_path: Path, dest_path: Path) -> None:
    import torch  # local import

    state = torch.load(str(source_path), map_location="cpu")
    zeroed = {
        key: (torch.zeros_like(val) if hasattr(val, "zero_") else val)
        for key, val in state.items()
    }
    torch.save(zeroed, str(dest_path))


def materialize_zero_adapter(
    reference_adapter_dir: Path, output_dir: Path
) -> Path:
    """Build a zero-weight adapter that structurally matches ``reference_adapter_dir``.

    Returns the path to ``output_dir``.
    """
    reference_adapter_dir = _resolve_reference_adapter(reference_adapter_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # adapter_config.json — verbatim copy so rank/targets/alpha align.
    shutil.copy(
        reference_adapter_dir / "adapter_config.json",
        output_dir / "adapter_config.json",
    )

    # Weights: zero everything while preserving keys and dtypes.
    st_src = reference_adapter_dir / "adapter_model.safetensors"
    bin_src = reference_adapter_dir / "adapter_model.bin"
    if st_src.exists():
        _zero_out_safetensors(st_src, output_dir / "adapter_model.safetensors")
    elif bin_src.exists():
        _zero_out_bin(bin_src, output_dir / "adapter_model.bin")
    else:  # pragma: no cover - validated above, defensive
        raise SystemExit(
            f"reference adapter at {reference_adapter_dir} has no weight file"
        )

    # Copy tokenizer / module-map auxiliaries when present (PEFT sometimes
    # writes these alongside the adapter).
    for aux_name in (
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "peft_version.json",
        "training_args.json",
    ):
        src = reference_adapter_dir / aux_name
        if src.exists():
            shutil.copy(src, output_dir / aux_name)

    # Final validation to catch any path drift.
    validate_adapter_dir(str(output_dir))
    return output_dir


def run_baseline_eval(
    *,
    reference_adapter_dir: Path,
    input_path: Path,
    output_path: Path,
    config_path: Path,
    workdir: Path,
    contract: str,
    num_repeats: int,
    write_raw_predictions: bool,
    raw_predictions_dir: Path | None,
) -> dict[str, Any]:
    zero_adapter_dir = workdir / "zero_adapter"
    materialize_zero_adapter(reference_adapter_dir, zero_adapter_dir)
    payload = evaluate_official_vllm_proxy(
        adapter_dir=zero_adapter_dir,
        input_path=input_path,
        output_path=output_path,
        config_path=config_path,
        num_repeats=max(1, int(num_repeats)),
        write_raw_predictions=bool(write_raw_predictions),
        raw_predictions_dir=(
            None if raw_predictions_dir is None else str(raw_predictions_dir)
        ),
        no_load_base_model=False,
        contract=contract,
    )
    # Tag the payload so downstream selection can't mistake a base-zero
    # baseline for a real checkpoint's eval.
    payload["baseline_kind"] = "base_zero_adapter"
    payload["reference_adapter_dir"] = str(reference_adapter_dir)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the base Nemotron model with a zero-weight LoRA adapter "
            "(structurally identical to a reference adapter) under the Kaggle "
            "runtime contract. Produces the absolute floor that any real "
            "finetuned adapter must beat."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--reference-adapter-dir",
        type=Path,
        required=True,
        help=(
            "Path to a valid finetuned adapter (e.g. stage2_bestproxy). Only "
            "its adapter_config.json and weight-file structure are reused; "
            "the values are zeroed out."
        ),
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("artifacts/baseline_base_zero_adapter"),
        help=(
            "Directory to materialise the zero-weight adapter. Reused across "
            "runs; contents are overwritten."
        ),
    )
    parser.add_argument(
        "--contract",
        choices=["runtime", "notebook"],
        default="runtime",
        help="Which Kaggle eval contract to use. Defaults to runtime (authoritative).",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=1,
        help=(
            "Number of sampled generations. Forced to 1 under the runtime "
            "contract (greedy)."
        ),
    )
    parser.add_argument("--write-raw-predictions", action="store_true")
    parser.add_argument("--raw-predictions-dir", type=Path)
    args = parser.parse_args(argv)

    run_baseline_eval(
        reference_adapter_dir=args.reference_adapter_dir,
        input_path=args.input,
        output_path=args.output,
        config_path=args.config,
        workdir=args.workdir,
        contract=args.contract,
        num_repeats=args.num_repeats,
        write_raw_predictions=args.write_raw_predictions,
        raw_predictions_dir=args.raw_predictions_dir,
    )


if __name__ == "__main__":
    main()
