"""Validate that a smoke training run produced healthy gradients and loss.

The Mamba `in_proj` overflow bug silently continued the run with
grad_norm == inf and loss == garbage.  This gate reads
``trainer_state.json`` (written by HF Trainer on every checkpoint)
and fails fast if any logged optimizer step shows:

* non-finite grad_norm (inf / nan)
* grad_norm above ``--max-grad-norm`` (default 1e10)
* training loss above ``--max-loss`` (default 10.0)
* fewer than ``--min-steps`` optimizer steps were logged

Run AFTER ``python -m src.student.lora_train --config
configs/smoke/train_stage1_smoke.yaml``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _iter_log_history(state_path: Path) -> list[dict]:
    if not state_path.is_file():
        raise SystemExit(f"missing trainer_state.json: {state_path}")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    history = payload.get("log_history", [])
    if not isinstance(history, list):
        raise SystemExit("log_history must be a list")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-dir",
        required=True,
        help="Smoke adapter output dir, e.g. artifacts/smoke/adapter_stage1_format",
    )
    parser.add_argument("--max-loss", type=float, default=10.0)
    parser.add_argument("--max-grad-norm", type=float, default=1e10)
    parser.add_argument("--min-steps", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    # Prefer the canonical trainer_state.json produced during the run.
    # Fall back to the latest checkpoint-*/trainer_state.json if the root copy
    # was rotated by save_total_limit.
    candidates = [adapter_dir / "trainer_state.json"]
    checkpoints = sorted(
        adapter_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[-1]),
    )
    candidates.extend(checkpoints)
    state_path = next((p for p in candidates if p.is_file()), None)
    if state_path is None:
        raise SystemExit(
            f"smoke health: no trainer_state.json under {adapter_dir}"
        )

    history = _iter_log_history(state_path)
    losses = []
    grad_norms = []
    bad_rows = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        loss = entry.get("loss")
        grad_norm = entry.get("grad_norm")
        if loss is None and grad_norm is None:
            # eval rows / epoch markers are fine to skip
            continue
        step = entry.get("step")
        row = {"step": step, "loss": loss, "grad_norm": grad_norm}
        if loss is not None:
            try:
                loss_f = float(loss)
            except (TypeError, ValueError):
                loss_f = float("nan")
            losses.append(loss_f)
            if not math.isfinite(loss_f) or loss_f > args.max_loss:
                bad_rows.append({**row, "reason": f"loss={loss_f} exceeds {args.max_loss}"})
        if grad_norm is not None:
            try:
                gn_f = float(grad_norm)
            except (TypeError, ValueError):
                gn_f = float("nan")
            grad_norms.append(gn_f)
            if not math.isfinite(gn_f) or gn_f > args.max_grad_norm:
                bad_rows.append(
                    {**row, "reason": f"grad_norm={gn_f} exceeds {args.max_grad_norm}"}
                )

    steps_seen = len(losses)
    summary = {
        "state_path": str(state_path),
        "optimizer_steps_logged": steps_seen,
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
        "max_grad_norm": max(grad_norms) if grad_norms else None,
        "threshold_max_loss": args.max_loss,
        "threshold_max_grad_norm": args.max_grad_norm,
        "threshold_min_steps": args.min_steps,
        "bad_rows": bad_rows,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if steps_seen < args.min_steps:
        raise SystemExit(
            f"smoke health: only {steps_seen} optimizer steps logged, "
            f"need at least {args.min_steps}"
        )
    if bad_rows:
        raise SystemExit(
            f"smoke health: {len(bad_rows)} bad row(s) detected (see bad_rows above). "
            "Training is NOT healthy - do not proceed to stage1. "
            "Check target_modules / LR / dtype settings."
        )
    print("smoke health: OK")


if __name__ == "__main__":
    main()
