"""Shared adapter / checkpoint proxy comparison helpers.

Both ``scripts.select_final_adapter`` (stage2 vs stage3) and
``scripts.select_best_proxy_checkpoint`` (checkpoint-level selection inside a
stage) compare two candidates by:

1. primary: all-family proxy ``competition_correct_rate``
2. tie-break: hard-triad proxy ``competition_correct_rate``
3. tiebreak_default on complete tie

Tolerance is half a sample at the proxy's resolution. Centralising the logic
prevents the two call sites from drifting apart.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def load_proxy_eval(path: str | Path) -> dict[str, Any]:
    """Load a proxy eval JSON and enforce the coverage and field contracts.

    Raises ``SystemExit`` on any of:

    - non-object payload
    - missing ``competition_correct_rate`` / ``num_examples``
    - ``coverage.num_missing`` / ``num_unexpected`` / ``num_duplicate`` != 0
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a JSON object")

    coverage = payload.get("coverage", {})
    if not isinstance(coverage, dict):
        raise SystemExit(f"{path}: 'coverage' must be a dict")
    for field in ("num_missing", "num_unexpected", "num_duplicate"):
        value = coverage.get(field, 0)
        if value != 0:
            raise SystemExit(
                f"{path}: coverage.{field} = {value}, refusing to select from a "
                "proxy artifact whose prediction coverage is not complete."
            )

    if "competition_correct_rate" not in payload:
        raise SystemExit(f"{path}: missing 'competition_correct_rate'")
    if "num_examples" not in payload:
        raise SystemExit(f"{path}: missing 'num_examples'")

    return {
        "path": str(path),
        "competition_correct_rate": float(payload["competition_correct_rate"]),
        "num_examples": int(payload["num_examples"]),
        "coverage": coverage,
    }


def compare_proxy_pairs(
    *,
    left_name: str,
    left_all: dict[str, Any],
    left_hard: dict[str, Any],
    right_name: str,
    right_all: dict[str, Any],
    right_hard: dict[str, Any],
    tiebreak_default: str,
) -> dict[str, Any]:
    """Compare two candidates and return the winner plus the rule that fired.

    Returned dict fields:

    - ``winner``: one of ``left_name`` / ``right_name``
    - ``rule``: ``'all_family_primary'`` / ``'hard_triad_tiebreak'`` /
      ``'prefer_default_on_tie'``
    - ``all_delta``, ``hard_delta``, ``all_tol``, ``hard_tol``
    """
    if left_all["num_examples"] != right_all["num_examples"]:
        raise SystemExit(
            "all-family proxy num_examples mismatch: "
            f"{left_name}={left_all['num_examples']} vs "
            f"{right_name}={right_all['num_examples']}"
        )
    if left_hard["num_examples"] != right_hard["num_examples"]:
        raise SystemExit(
            "hard-triad proxy num_examples mismatch: "
            f"{left_name}={left_hard['num_examples']} vs "
            f"{right_name}={right_hard['num_examples']}"
        )
    if tiebreak_default not in (left_name, right_name):
        raise ValueError(
            f"tiebreak_default must be one of ({left_name!r}, {right_name!r}), "
            f"got {tiebreak_default!r}"
        )

    all_tol = 0.5 / max(1, left_all["num_examples"])
    hard_tol = 0.5 / max(1, left_hard["num_examples"])
    all_delta = (
        right_all["competition_correct_rate"] - left_all["competition_correct_rate"]
    )
    hard_delta = (
        right_hard["competition_correct_rate"] - left_hard["competition_correct_rate"]
    )

    base = {
        "all_delta": all_delta,
        "hard_delta": hard_delta,
        "all_tol": all_tol,
        "hard_tol": hard_tol,
    }
    if all_delta > all_tol:
        return {"winner": right_name, "rule": "all_family_primary", **base}
    if all_delta < -all_tol:
        return {"winner": left_name, "rule": "all_family_primary", **base}
    if hard_delta > hard_tol:
        return {"winner": right_name, "rule": "hard_triad_tiebreak", **base}
    if hard_delta < -hard_tol:
        return {"winner": left_name, "rule": "hard_triad_tiebreak", **base}
    return {"winner": tiebreak_default, "rule": "prefer_default_on_tie", **base}


def copy_adapter_dir(src: str | Path, dst: str | Path) -> None:
    """Copy an adapter directory to ``dst``, replacing any existing target."""
    src_path = Path(src)
    if not src_path.is_dir():
        raise SystemExit(f"adapter source not found: {src}")
    dst_path = Path(dst)
    if dst_path.exists():
        shutil.rmtree(dst_path)
    shutil.copytree(src_path, dst_path)
