"""risk_score_gate.py
=========================
Stage-2 (and Stage-3) early-stop judge that does NOT rely on
``eval_loss``.

Round 4 GPT-5.4 Pro guidance: greedy ``competition_correct_rate`` plus
format/length health tell a very different story from token-level CE.
Concretely, Stage-2 CE can keep drifting down while the model learns to
produce longer, more template-y traces that miss the ``\\boxed{}`` guard
more often — LB goes flat or drops.

This script consumes a sequence of per-checkpoint eval reports and
emits one of:

    CONTINUE
    STOP_STAGE2_PROMOTE_BEST
    STOP_STAGE2_REGRESSION

It also prints a table with the four secondary metrics (parse_fail,
no_final, avg_output_tokens, truncation_rate) and the composite
``risk_score``.

Input schema
------------
A directory containing per-step eval JSONs, e.g.::

    artifacts/adapter_stage2_bestproxy/eval_step_1000.json
    artifacts/adapter_stage2_bestproxy/eval_step_1100.json
    ...

Each JSON must include::

    {
      "step": 1100,
      "eval_loss": 0.2554731,
      "competition_correct_rate": 0.715,
      "parse_fail_rate":           0.012,
      "no_final_rate":             0.034,
      "truncation_rate":           0.007,
      "avg_output_tokens":         1823.4
    }

The first three fields are mandatory; the rest default to 0 when
missing but that will make the gate noisy, so you should always emit
them.

Decision logic (Round 4)
------------------------
Stop Stage-2 when **any 2** of the following hold across the most recent
two checkpoints (t, t-1)::

    1. relative_loss_slope_100 < 0.003  (over last 300 steps)
    2. correct_rate_t - correct_rate_{t-1} < +0.005
       AND McNemar p-value for (t, t-1) > 0.10
       (if a paired eval JSON is present at step t)
    3. avg_output_tokens_t / avg_output_tokens_{t-1} > 1.08
       AND correct_rate_t <= correct_rate_{t-1}
    4. parse_fail_rate_t >= parse_fail_rate_{t-1}
       AND no_final_rate_t >= no_final_rate_{t-1}
    5. risk_score_t <= risk_score_{t-1}

``risk_score`` is defined as::

    risk_score = correct_rate
               - 1.5 * parse_fail_rate
               - 1.0 * no_final_rate
               - 1.0 * truncation_rate
               - 0.00002 * avg_output_tokens

(the last term is dimensionless when ``avg_output_tokens`` is in tokens;
~0.04 penalty at 2000 tokens.)

Usage
-----
::

    # Dry run on synthetic eval reports
    python scripts/risk_score_gate.py --dry-run

    # Real run — point at Stage-2 artifacts directory
    python scripts/risk_score_gate.py \
        --eval-dir artifacts/adapter_stage2_bestproxy \
        --min-steps 900 --max-steps 2000

Exit codes::

    0  CONTINUE           — no rule fired, keep training
    10 STOP_STAGE2_PROMOTE_BEST   — stop, promote the highest risk_score checkpoint
    20 STOP_STAGE2_REGRESSION     — stop, latest checkpoint is worse than previous
    40 INSUFFICIENT_DATA  — fewer than 2 eval reports; caller should continue
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


def compute_risk_score(report: dict[str, Any]) -> float:
    return (
        float(report.get("competition_correct_rate", 0.0))
        - 1.5 * float(report.get("parse_fail_rate", 0.0))
        - 1.0 * float(report.get("no_final_rate", 0.0))
        - 1.0 * float(report.get("truncation_rate", 0.0))
        - 0.00002 * float(report.get("avg_output_tokens", 0.0))
    )


def load_reports(eval_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for p in sorted(eval_dir.glob("eval_step_*.json")):
        try:
            data = json.loads(p.read_text())
            if "step" not in data:
                # Parse step from filename
                stem = p.stem.replace("eval_step_", "")
                data["step"] = int(stem)
            reports.append(data)
        except (json.JSONDecodeError, ValueError):
            continue
    reports.sort(key=lambda r: int(r["step"]))
    return reports


def relative_loss_slope_per_100(reports: list[dict[str, Any]], window: int = 300) -> float | None:
    """Relative loss drop per 100 steps over the last ``window`` steps.

    Returns ``None`` when the window doesn't span enough reports.
    """
    if len(reports) < 2:
        return None
    last = reports[-1]
    step_last = int(last["step"])
    loss_last = float(last.get("eval_loss", 0.0) or 0.0)
    if loss_last <= 0:
        return None
    # Find the closest report at step_last - window
    target_step = step_last - window
    best = None
    best_delta = None
    for r in reports[:-1]:
        delta = abs(int(r["step"]) - target_step)
        if best is None or delta < best_delta:
            best = r
            best_delta = delta
    if best is None or best_delta is None or best_delta > window // 2:
        return None
    loss_prev = float(best.get("eval_loss", 0.0) or 0.0)
    step_prev = int(best["step"])
    num_100 = max((step_last - step_prev) / 100.0, 1.0)
    abs_slope = (loss_prev - loss_last) / num_100
    return abs_slope / max(loss_last, 1e-9)


def evaluate_gate(
    reports: list[dict[str, Any]],
    min_steps: int,
    max_steps: int,
    mcnemar_pvalue_latest: float | None = None,
) -> dict[str, Any]:
    """Return the gate decision + rule table."""
    if len(reports) < 2:
        return {
            "decision": "INSUFFICIENT_DATA",
            "exit_code": 40,
            "reports_seen": len(reports),
        }

    latest = reports[-1]
    prev = reports[-2]
    step_latest = int(latest["step"])

    rs_latest = compute_risk_score(latest)
    rs_prev = compute_risk_score(prev)

    rules: dict[str, bool] = {}

    # Rule 1: flat loss
    rel_slope = relative_loss_slope_per_100(reports, window=300)
    rules["rule1_flat_loss"] = bool(rel_slope is not None and rel_slope < 0.003)

    # Rule 2: no correct_rate gain + McNemar insignificant
    cr_latest = float(latest.get("competition_correct_rate", 0.0))
    cr_prev = float(prev.get("competition_correct_rate", 0.0))
    cr_gain = cr_latest - cr_prev
    rules["rule2_no_correct_rate_gain"] = bool(
        cr_gain < 0.005
        and (mcnemar_pvalue_latest is None or mcnemar_pvalue_latest > 0.10)
    )

    # Rule 3: output bloat without accuracy gain
    tok_latest = float(latest.get("avg_output_tokens", 0.0) or 0.0)
    tok_prev = float(prev.get("avg_output_tokens", 0.0) or 0.0)
    bloat_ratio = (tok_latest / tok_prev) if tok_prev > 0 else 1.0
    rules["rule3_output_bloat"] = bool(bloat_ratio > 1.08 and cr_latest <= cr_prev)

    # Rule 4: format health not improving
    pf_latest = float(latest.get("parse_fail_rate", 0.0))
    pf_prev = float(prev.get("parse_fail_rate", 0.0))
    nf_latest = float(latest.get("no_final_rate", 0.0))
    nf_prev = float(prev.get("no_final_rate", 0.0))
    rules["rule4_format_not_improving"] = bool(
        pf_latest >= pf_prev and nf_latest >= nf_prev
    )

    # Rule 5: risk_score regressing
    rules["rule5_risk_score_regressing"] = bool(rs_latest <= rs_prev)

    n_fired = sum(rules.values())

    # Min steps guard — don't stop before min_steps regardless
    if step_latest < min_steps:
        decision = "CONTINUE"
        exit_code = 0
    elif step_latest >= max_steps:
        decision = "STOP_STAGE2_PROMOTE_BEST"
        exit_code = 10
    elif n_fired >= 2:
        # Regression variant: both rule3 AND rule5 fired
        if rules["rule3_output_bloat"] and rules["rule5_risk_score_regressing"]:
            decision = "STOP_STAGE2_REGRESSION"
            exit_code = 20
        else:
            decision = "STOP_STAGE2_PROMOTE_BEST"
            exit_code = 10
    else:
        decision = "CONTINUE"
        exit_code = 0

    return {
        "decision": decision,
        "exit_code": exit_code,
        "step_latest": step_latest,
        "step_prev": int(prev["step"]),
        "correct_rate_latest": cr_latest,
        "correct_rate_prev": cr_prev,
        "correct_rate_gain": cr_gain,
        "relative_loss_slope_per_100": rel_slope,
        "avg_output_tokens_ratio": bloat_ratio,
        "risk_score_latest": rs_latest,
        "risk_score_prev": rs_prev,
        "n_rules_fired": n_fired,
        "rules": rules,
        "mcnemar_pvalue_latest": mcnemar_pvalue_latest,
    }


# ---------------------------------------------------------------------------
# Dry-run fixtures
# ---------------------------------------------------------------------------

_DRY_REPORTS = [
    {  # healthy improvement
        "step": 800,
        "eval_loss": 0.320,
        "competition_correct_rate": 0.680,
        "parse_fail_rate": 0.025,
        "no_final_rate": 0.048,
        "truncation_rate": 0.012,
        "avg_output_tokens": 1620.0,
    },
    {
        "step": 900,
        "eval_loss": 0.291,
        "competition_correct_rate": 0.702,
        "parse_fail_rate": 0.019,
        "no_final_rate": 0.041,
        "truncation_rate": 0.010,
        "avg_output_tokens": 1670.0,
    },
    {
        "step": 1000,
        "eval_loss": 0.256,
        "competition_correct_rate": 0.714,
        "parse_fail_rate": 0.013,
        "no_final_rate": 0.036,
        "truncation_rate": 0.008,
        "avg_output_tokens": 1723.0,
    },
    {  # stage2 plateau + bloat
        "step": 1100,
        "eval_loss": 0.255,
        "competition_correct_rate": 0.716,
        "parse_fail_rate": 0.013,
        "no_final_rate": 0.036,
        "truncation_rate": 0.008,
        "avg_output_tokens": 1870.0,
    },
]


def _dry_run() -> int:
    decision = evaluate_gate(_DRY_REPORTS, min_steps=900, max_steps=2000)
    print("[risk_score_gate] dry-run decision:")
    print(json.dumps(decision, indent=2))
    assert decision["decision"] in (
        "CONTINUE",
        "STOP_STAGE2_PROMOTE_BEST",
        "STOP_STAGE2_REGRESSION",
    ), f"unexpected decision {decision['decision']}"
    return decision["exit_code"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-2 early-stop gate based on risk_score + 5-rule logic. "
            "Does NOT use eval_loss as the primary criterion."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--eval-dir", default=None, help="Directory with eval_step_*.json files.")
    parser.add_argument("--min-steps", type=int, default=900, dest="min_steps")
    parser.add_argument("--max-steps", type=int, default=2000, dest="max_steps")
    parser.add_argument(
        "--mcnemar-pvalue",
        type=float,
        default=None,
        dest="mcnemar_pvalue",
        help="Optional paired-eval McNemar p-value for (latest, prev).",
    )
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    args = parser.parse_args(argv)

    if args.dry_run or args.eval_dir is None:
        sys.exit(_dry_run())

    reports = load_reports(Path(args.eval_dir))
    decision = evaluate_gate(
        reports,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        mcnemar_pvalue_latest=args.mcnemar_pvalue,
    )
    print(json.dumps(decision, indent=2))
    sys.exit(decision["exit_code"])


if __name__ == "__main__":
    main()
