"""paired_eval_mcnemar.py
==============================
Paired statistical comparison of two eval runs on the same question set.

Round 4 fix for the Round 1-A gate: ``+0.005`` / ``+0.010`` absolute-rate
thresholds on a 500-question proxy are below the binomial noise floor
(paired SE ~1.7pp at n=500), so they can't be used as hard gates.
This script replaces that with:

* **McNemar's exact test** on discordant pairs (n01 vs n10)
* **Paired bootstrap** (default 10,000 resamples) over the per-question
  score vector to get 10% / 50% / 90% quantiles of ``Δacc``

Input format
------------
Each eval run supplies a JSONL where every row is::

    {"id": "<question_id>", "correct": 0_or_1, "output_tokens": int, ...}

Rows are aligned by ``id``; questions missing in either file are dropped
and counted.

Decision tiers (aligned with Round 4 guidance)
----------------------------------------------
* **500-question triage** (``--sample-size quick``):
    * PROMOTE if ``delta >= +3.0pp`` AND McNemar p < 0.10
    * REJECT  if ``delta <= -2.0pp`` AND secondary metrics worse
    * Else: ESCALATE to 1500+ question gate
* **1500–2000 question decision** (``--sample-size full``):
    * PROMOTE if ``delta >= +1.5pp`` AND 10%-bootstrap > 0
      AND parse_fail/no_final don't regress
    * STRONG_PROMOTE if ``delta >= +2.0pp`` AND McNemar p < 0.05
    * REJECT if ``delta <= -1.0pp`` AND secondary regressions
    * Else: INCONCLUSIVE

Usage
-----
::

    # Dry run (no deps)
    python scripts/paired_eval_mcnemar.py --dry-run

    # Real run — compare two evals on the same proxy
    python scripts/paired_eval_mcnemar.py \
        --a run_A_baseline/eval_per_question.jsonl \
        --b run_C_gcd_round3/eval_per_question.jsonl \
        --sample-size full \
        --output artifacts/round1a/paired_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Paired stats
# ---------------------------------------------------------------------------

def mcnemar_exact_pvalue(n01: int, n10: int) -> float:
    """Two-sided exact McNemar's test (binomial) on discordant pairs.

    For the discordant count ``n = n01 + n10`` and observed extreme ``k =
    min(n01, n10)``, p-value = 2 * binomCDF(k; n, 0.5), clamped at 1.0.
    """
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    # Binomial CDF at k with p=0.5
    cdf = 0.0
    for i in range(k + 1):
        cdf += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * cdf)


def paired_bootstrap(
    a_scores: list[int],
    b_scores: list[int],
    num_resamples: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap the per-question ``b - a`` mean.  Returns quantiles."""
    rng = random.Random(seed)
    n = len(a_scores)
    if n == 0:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "n": 0}
    deltas: list[float] = []
    for _ in range(num_resamples):
        sum_d = 0
        for _j in range(n):
            idx = rng.randrange(n)
            sum_d += b_scores[idx] - a_scores[idx]
        deltas.append(sum_d / n)
    deltas.sort()

    def q(p: float) -> float:
        i = max(0, min(len(deltas) - 1, int(round(p * (len(deltas) - 1)))))
        return deltas[i]

    return {
        "mean": sum(deltas) / len(deltas),
        "p10": q(0.10),
        "p50": q(0.50),
        "p90": q(0.90),
        "n": n,
    }


def align_runs(a_rows: list[dict], b_rows: list[dict]) -> tuple[list[int], list[int], int]:
    """Align per-id and return (a_scores, b_scores, n_dropped)."""
    a_map = {str(r["id"]): int(r.get("correct", 0)) for r in a_rows}
    b_map = {str(r["id"]): int(r.get("correct", 0)) for r in b_rows}
    common = sorted(set(a_map) & set(b_map))
    dropped = len(a_map) + len(b_map) - 2 * len(common)
    return [a_map[k] for k in common], [b_map[k] for k in common], dropped


def compute_paired_stats(a_scores: list[int], b_scores: list[int]) -> dict[str, Any]:
    assert len(a_scores) == len(b_scores)
    n = len(a_scores)
    n11 = sum(1 for i in range(n) if a_scores[i] == 1 and b_scores[i] == 1)
    n00 = sum(1 for i in range(n) if a_scores[i] == 0 and b_scores[i] == 0)
    n10 = sum(1 for i in range(n) if a_scores[i] == 1 and b_scores[i] == 0)
    n01 = sum(1 for i in range(n) if a_scores[i] == 0 and b_scores[i] == 1)
    delta = (n01 - n10) / max(n, 1)
    pvalue = mcnemar_exact_pvalue(n01, n10)
    boot = paired_bootstrap(a_scores, b_scores, num_resamples=10000)
    return {
        "n": n,
        "n11_both_correct": n11,
        "n00_both_wrong": n00,
        "n10_only_a_correct": n10,
        "n01_only_b_correct": n01,
        "delta_mean": delta,
        "delta_absolute_pp": delta * 100,
        "mcnemar_pvalue_two_sided": pvalue,
        "bootstrap": boot,
    }


def decide(
    stats: dict[str, Any],
    sample_size: str,
    secondary_delta: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply the Round 4 tiered decision rules."""
    delta = stats["delta_mean"]
    pval = stats["mcnemar_pvalue_two_sided"]
    boot_p10 = stats["bootstrap"]["p10"]

    secondary_bad = False
    secondary_good = False
    if secondary_delta is not None:
        parse_worse = secondary_delta.get("parse_fail_delta", 0.0) >= 0.005
        nofinal_worse = secondary_delta.get("no_final_delta", 0.0) >= 0.005
        tok_bloat = secondary_delta.get("avg_output_tokens_ratio", 1.0) > 1.15
        secondary_bad = parse_worse or nofinal_worse or tok_bloat
        secondary_good = (
            secondary_delta.get("parse_fail_delta", 0.0) < 0
            or secondary_delta.get("no_final_delta", 0.0) < 0
        )

    if sample_size == "quick":
        if delta >= 0.030 and pval < 0.10:
            return {"decision": "PROMOTE", "reason": "Δ≥+3.0pp AND McNemar p<0.10"}
        if delta <= -0.020 and secondary_bad:
            return {"decision": "REJECT", "reason": "Δ≤-2.0pp AND secondary regressions"}
        return {"decision": "ESCALATE", "reason": "quick gate inconclusive, run full 1500–2000"}

    # full gate
    if delta >= 0.020 and pval < 0.05:
        return {"decision": "STRONG_PROMOTE", "reason": "Δ≥+2.0pp AND McNemar p<0.05"}
    if delta >= 0.015 and boot_p10 > 0.0 and not secondary_bad:
        return {"decision": "PROMOTE", "reason": "Δ≥+1.5pp AND bootstrap-10%>0 AND secondary clean"}
    if delta <= -0.010 and secondary_bad:
        return {"decision": "REJECT", "reason": "Δ≤-1.0pp AND secondary regressions"}
    return {"decision": "INCONCLUSIVE", "reason": "between thresholds"}


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def _dry_run() -> int:
    random.seed(42)
    n = 2000
    # Synthetic: A at 72%, B at 74.5%, with some paired structure
    a = [1 if random.random() < 0.72 else 0 for _ in range(n)]
    b = []
    for ai in a:
        if ai == 1:
            b.append(1 if random.random() < 0.95 else 0)
        else:
            b.append(1 if random.random() < 0.35 else 0)
    stats = compute_paired_stats(a, b)
    decision = decide(stats, sample_size="full", secondary_delta={
        "parse_fail_delta": -0.002,
        "no_final_delta": -0.003,
        "avg_output_tokens_ratio": 1.04,
    })
    out = {"stats": stats, "decision": decision}
    print(json.dumps(out, indent=2))
    assert stats["n"] == n
    assert 0.0 <= stats["mcnemar_pvalue_two_sided"] <= 1.0
    assert decision["decision"] in ("STRONG_PROMOTE", "PROMOTE", "INCONCLUSIVE", "REJECT")
    print("[paired_eval_mcnemar] dry-run OK", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_jsonl(p: Path) -> list[dict]:
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Paired McNemar + bootstrap comparison of two eval runs. "
            "Replaces absolute-delta hard gates with statistically sound "
            "decision tiers per Round 4."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--a", default=None, help="Baseline per-question JSONL.")
    parser.add_argument("--b", default=None, help="Candidate per-question JSONL.")
    parser.add_argument(
        "--sample-size",
        choices=["quick", "full"],
        default="full",
        dest="sample_size",
    )
    parser.add_argument("--output", default=None, help="Write JSON report here.")
    parser.add_argument("--secondary-a", default=None, help="Optional summary JSON for A.")
    parser.add_argument("--secondary-b", default=None, help="Optional summary JSON for B.")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    args = parser.parse_args(argv)

    if args.dry_run or args.a is None or args.b is None:
        sys.exit(_dry_run())

    a_rows = load_jsonl(Path(args.a))
    b_rows = load_jsonl(Path(args.b))
    a_scores, b_scores, dropped = align_runs(a_rows, b_rows)
    stats = compute_paired_stats(a_scores, b_scores)
    stats["n_dropped_from_alignment"] = dropped

    secondary_delta = None
    if args.secondary_a and args.secondary_b:
        sa = json.loads(Path(args.secondary_a).read_text())
        sb = json.loads(Path(args.secondary_b).read_text())
        secondary_delta = {
            "parse_fail_delta": float(sb.get("parse_fail_rate", 0.0))
            - float(sa.get("parse_fail_rate", 0.0)),
            "no_final_delta": float(sb.get("no_final_rate", 0.0))
            - float(sa.get("no_final_rate", 0.0)),
            "avg_output_tokens_ratio": (
                float(sb.get("avg_output_tokens", 1.0))
                / max(float(sa.get("avg_output_tokens", 1.0)), 1.0)
            ),
        }

    decision = decide(stats, sample_size=args.sample_size, secondary_delta=secondary_delta)
    out = {"stats": stats, "decision": decision, "secondary_delta": secondary_delta}
    text = json.dumps(out, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
