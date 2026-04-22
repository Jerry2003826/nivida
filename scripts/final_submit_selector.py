"""final_submit_selector.py
===========================
Pick the two final Kaggle submissions from a pool of candidate adapters.

Kaggle allows exactly two final selections per team per competition.  The
naive choice is "top two by public LB", but that maximises correlation
between the two picks and wastes the second slot — if the slot-1 pick
happens to over-fit the public split, the second pick is very likely to
over-fit in the same direction.

Round 4 (GPT-5.4 Pro, 2026-04-22) prescribes a decorrelated allocation:

  * **Slot #1 — risk-adjusted best.**  Rank candidates by ``risk_score``
    ``= correct_rate − 1.5·parse_fail_rate − 1.0·no_final_rate
       − 1.0·truncation_rate − 2e-5·avg_output_tokens``.
    The argmax is the single most robust candidate on the held-out proxy.
    This matches the stage2 early-stop gate (risk_score_gate.py) so the
    candidate that was selected by the gate is automatically the slot-1
    pick.
  * **Slot #2 — merged private-hedge.**  Among candidates whose
    ``risk_score`` is within ``--top-band`` (default 0.005) of the slot-1
    winner, pick the one that maximises the **paired hedge score**
    ``private_hedge_score = wins_where_slot1_wrong − losses_where_slot1_right``
    computed from per-prompt correctness vectors.  Intuitively: among
    near-equal candidates, we want the one that gets different prompts
    right — so when the public split over-estimates slot-1, slot-2 covers
    the private split.

This replaces the legacy "two stage2 checkpoints N steps apart" heuristic,
which produced near-identical correctness vectors and therefore no
private-split diversification.

Per-prompt correctness inputs
-----------------------------
For each candidate adapter we expect a JSON file produced by
``paired_eval_mcnemar.py`` (or an equivalent harness) at path
``<candidate_dir>/eval/per_prompt.json`` with schema::

    {
      "prompt_ids": ["p0", "p1", ...],
      "correct":    [true,  false, ...]   // one bool per prompt_id
    }

The selector aligns candidates by ``prompt_ids`` (must be identical across
all candidates; the tool asserts this).

Candidate metadata inputs
-------------------------
Each candidate directory must also contain ``metrics.json`` with at least::

    {
      "correct_rate":        0.724,
      "parse_fail_rate":     0.011,
      "no_final_rate":       0.006,
      "truncation_rate":     0.003,
      "avg_output_tokens":   1180.0
    }

These are the same fields computed by ``risk_score_gate.py`` so a single
offline eval run feeds both gates.

CLI
---
    python scripts/final_submit_selector.py \\
        --candidates-root artifacts/ \\
        --candidate adapter_stage2_bestproxy \\
        --candidate adapter_stage2_ckpt_1050 \\
        --candidate adapter_stage3_bestproxy \\
        --output data/final_selection.json

The output is a JSON document recording slot assignments, the rejected
pool, every score, and the hedge matrix used.  Exit code 0 on success;
non-zero if the candidate pool is empty or metadata is missing.

Dry-run
-------
``--dry-run`` fabricates 3 synthetic candidates with known correctness
patterns and validates that the selector:
  * picks the candidate with the highest risk_score for slot #1
  * picks a decorrelated candidate (not the second-highest risk_score)
    for slot #2 when the hedge score favours it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def risk_score(
    correct_rate: float,
    parse_fail_rate: float,
    no_final_rate: float,
    truncation_rate: float,
    avg_output_tokens: float,
    *,
    w_parse: float = 1.5,
    w_no_final: float = 1.0,
    w_truncation: float = 1.0,
    w_tokens: float = 2e-5,
) -> float:
    """Composite risk-adjusted score.

    Matches ``risk_score_gate.py`` so the stage2 early-stop gate and the
    final selector agree on the slot-1 argmax candidate.
    """
    return (
        correct_rate
        - w_parse * parse_fail_rate
        - w_no_final * no_final_rate
        - w_truncation * truncation_rate
        - w_tokens * float(avg_output_tokens)
    )


def paired_hedge_score(
    slot1_correct: list[bool],
    candidate_correct: list[bool],
) -> dict[str, int]:
    """Return the hedge accounting for *candidate* vs *slot1*.

    For two aligned correctness vectors of equal length:
      * ``wins``   — candidate correct AND slot1 wrong  (private-split lift)
      * ``losses`` — candidate wrong   AND slot1 correct (private-split drag)
      * ``both``   — both correct (no hedge value either direction)
      * ``neither``— both wrong
      * ``score``  — wins − losses (scalar used for ranking)

    The score is intentionally asymmetric: a hedge candidate is useful
    only when it *covers* prompts slot-1 misses, not merely when it
    diverges.
    """
    if len(slot1_correct) != len(candidate_correct):
        raise ValueError(
            f"length mismatch: slot1={len(slot1_correct)} "
            f"candidate={len(candidate_correct)}"
        )
    wins = losses = both = neither = 0
    for s, c in zip(slot1_correct, candidate_correct):
        if c and not s:
            wins += 1
        elif s and not c:
            losses += 1
        elif s and c:
            both += 1
        else:
            neither += 1
    return {
        "wins": wins,
        "losses": losses,
        "both": both,
        "neither": neither,
        "score": wins - losses,
        "n": len(slot1_correct),
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_candidate(
    root: Path, name: str
) -> dict[str, Any]:
    cand_dir = root / name
    metrics_path = cand_dir / "metrics.json"
    per_prompt_path = cand_dir / "eval" / "per_prompt.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing {metrics_path}")
    if not per_prompt_path.exists():
        raise FileNotFoundError(f"missing {per_prompt_path}")
    metrics = json.loads(metrics_path.read_text())
    per_prompt = json.loads(per_prompt_path.read_text())
    prompt_ids = [str(p) for p in per_prompt["prompt_ids"]]
    correct = [bool(c) for c in per_prompt["correct"]]
    if len(prompt_ids) != len(correct):
        raise ValueError(
            f"{name}: prompt_ids ({len(prompt_ids)}) vs correct "
            f"({len(correct)}) length mismatch"
        )
    return {
        "name": name,
        "dir": str(cand_dir),
        "metrics": metrics,
        "prompt_ids": prompt_ids,
        "correct": correct,
        "risk_score": risk_score(
            correct_rate=float(metrics.get("correct_rate", 0.0)),
            parse_fail_rate=float(metrics.get("parse_fail_rate", 0.0)),
            no_final_rate=float(metrics.get("no_final_rate", 0.0)),
            truncation_rate=float(metrics.get("truncation_rate", 0.0)),
            avg_output_tokens=float(metrics.get("avg_output_tokens", 0.0)),
        ),
    }


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_two_submissions(
    candidates: list[dict[str, Any]],
    *,
    top_band: float = 0.005,
    require_min_hedge: int = 1,
) -> dict[str, Any]:
    """Pick slot-1 (risk-argmax) and slot-2 (paired hedge argmax).

    Parameters
    ----------
    candidates:
        List of dicts produced by ``_load_candidate``; must all share the
        same ``prompt_ids`` vector (asserted).
    top_band:
        Slot-2 candidates must have ``risk_score >= slot1.risk_score - top_band``.
        Default 0.005 (= half-a-point on a 0..1 correctness scale) matches
        the typical within-seed variance of the proxy.
    require_min_hedge:
        Minimum hedge score (``wins − losses``) for a slot-2 pick to be
        considered "useful".  When no candidate meets this threshold the
        selector falls back to the second-highest risk_score candidate and
        flags ``fallback_hedge=True`` in the report (slot-2 still fills,
        but the decorrelation was weak).
    """
    if not candidates:
        raise ValueError("candidate pool is empty")
    # Align prompt_ids across all candidates
    reference_ids = candidates[0]["prompt_ids"]
    for c in candidates[1:]:
        if c["prompt_ids"] != reference_ids:
            raise ValueError(
                f"candidate {c['name']} prompt_ids do not match reference "
                f"(first mismatch index: {_first_mismatch(c['prompt_ids'], reference_ids)})"
            )

    # Slot 1: risk-argmax
    sorted_by_risk = sorted(
        candidates, key=lambda c: c["risk_score"], reverse=True
    )
    slot1 = sorted_by_risk[0]

    # Slot 2: hedge-argmax within risk_score top-band
    cutoff = slot1["risk_score"] - top_band
    band = [c for c in sorted_by_risk[1:] if c["risk_score"] >= cutoff]

    hedge_details: list[dict[str, Any]] = []
    for cand in band:
        h = paired_hedge_score(slot1["correct"], cand["correct"])
        hedge_details.append({
            "name": cand["name"],
            "risk_score": cand["risk_score"],
            "hedge": h,
        })

    slot2: dict[str, Any] | None = None
    fallback_hedge = False
    if hedge_details:
        hedge_details.sort(key=lambda h: h["hedge"]["score"], reverse=True)
        top_hedge = hedge_details[0]
        if top_hedge["hedge"]["score"] >= require_min_hedge:
            slot2 = next(c for c in band if c["name"] == top_hedge["name"])
        else:
            # Fallback: second-highest risk_score (already in band or
            # falling back to sorted_by_risk[1] when band is empty)
            fallback_hedge = True
            slot2 = sorted_by_risk[1] if len(sorted_by_risk) > 1 else None
    elif len(sorted_by_risk) > 1:
        fallback_hedge = True
        slot2 = sorted_by_risk[1]

    rejected = [
        {"name": c["name"], "risk_score": c["risk_score"]}
        for c in sorted_by_risk
        if slot2 is None or c["name"] not in {slot1["name"], slot2["name"]}
    ]

    return {
        "slot1": {
            "name": slot1["name"],
            "dir": slot1["dir"],
            "risk_score": slot1["risk_score"],
            "metrics": slot1["metrics"],
            "selection_reason": "risk_score argmax",
        },
        "slot2": (
            {
                "name": slot2["name"],
                "dir": slot2["dir"],
                "risk_score": slot2["risk_score"],
                "metrics": slot2["metrics"],
                "selection_reason": (
                    "fallback: second-highest risk_score "
                    "(no candidate passed the hedge threshold)"
                    if fallback_hedge
                    else "paired hedge argmax within top-band"
                ),
                "hedge_vs_slot1": paired_hedge_score(
                    slot1["correct"], slot2["correct"]
                ),
            }
            if slot2 is not None
            else None
        ),
        "top_band": top_band,
        "require_min_hedge": require_min_hedge,
        "fallback_hedge": fallback_hedge,
        "hedge_candidates": hedge_details,
        "rejected": rejected,
        "n_candidates": len(candidates),
        "n_prompts": len(reference_ids),
    }


def _first_mismatch(a: list[str], b: list[str]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


# ---------------------------------------------------------------------------
# Dry-run fixtures
# ---------------------------------------------------------------------------

def _dry_run_candidates() -> list[dict[str, Any]]:
    """Three synthetic candidates exercising the selector paths.

    Prompt layout (n=20):
      - idx  0..14: all three candidates correct  (shared easy bucket)
      - idx 15..17: only Stage3 correct           (Stage3 private hedge)
      - idx     18: only Stage2A correct          (Stage2A hedge vs Stage3)
      - idx     19: only Stage2B correct

    Expected:
      - correct_rates: S3=0.90, S2A=0.80, S2B=0.80
      - risk_scores   dominated by S3 once modest error penalties are added
      - slot1 = S3 (highest risk_score)
      - slot2 hedge: S2A wins 1 loss 3 score -2  / S2B wins 1 loss 3 score -2
        → no positive hedge; falls back to second-highest risk_score
    """
    prompt_ids = [f"p{i:02d}" for i in range(20)]

    s3_correct = [True] * 15 + [True, True, True, False, False]
    s2a_correct = [True] * 15 + [False, False, False, True, False]
    s2b_correct = [True] * 15 + [False, False, False, False, True]

    def _mk(
        name: str,
        correct: list[bool],
        parse_fail: float,
        no_final: float,
        trunc: float,
        avg_tokens: float,
    ) -> dict[str, Any]:
        rate = sum(correct) / len(correct)
        metrics = {
            "correct_rate": rate,
            "parse_fail_rate": parse_fail,
            "no_final_rate": no_final,
            "truncation_rate": trunc,
            "avg_output_tokens": avg_tokens,
        }
        return {
            "name": name,
            "dir": f"artifacts/{name}",
            "metrics": metrics,
            "prompt_ids": list(prompt_ids),
            "correct": list(correct),
            "risk_score": risk_score(
                correct_rate=rate,
                parse_fail_rate=parse_fail,
                no_final_rate=no_final,
                truncation_rate=trunc,
                avg_output_tokens=avg_tokens,
            ),
        }

    return [
        _mk("adapter_stage3_bestproxy", s3_correct, 0.010, 0.004, 0.002, 1150.0),
        _mk("adapter_stage2_bestproxy", s2a_correct, 0.012, 0.006, 0.003, 1180.0),
        _mk("adapter_stage2_ckpt_1050", s2b_correct, 0.014, 0.008, 0.004, 1200.0),
    ]


def _dry_run_candidates_hedge_wins() -> list[dict[str, Any]]:
    """Variant where the hedge candidate genuinely wins the slot-2 fight.

    This scenario exploits the risk-score vs correct-rate divergence:
    the hedge candidate has higher correct_rate but pays higher parse /
    truncation / token penalties, so the clean candidate still wins
    slot-1 on risk_score.  The hedge's extra corrects happen to cover
    prompts slot-1 missed, producing a positive hedge score.

    Prompt layout (n=20):
      - Slot-1 correct on  idx 0..13               (14/20 = 0.70)
      - Hedge  correct on  idx 2..15, 16, 17       (16/20 = 0.80)
        Overlap idx 2..13 (12 both-correct);
        Slot-1-only idx 0,1            → losses = 2
        Hedge-only  idx 14,15,16,17    → wins   = 4
        → hedge_score = wins - losses = +2
    """
    prompt_ids = [f"p{i:02d}" for i in range(20)]

    s1_correct = [i in set(range(0, 14)) for i in range(20)]
    hedge_correct = [i in (set(range(2, 16)) | {16, 17}) for i in range(20)]
    low_correct = [True] * 10 + [False] * 10  # 10/20 = 0.50 (clearly worse)

    def _mk(name, correct, pf, nf, tr, at):
        rate = sum(correct) / len(correct)
        metrics = {
            "correct_rate": rate,
            "parse_fail_rate": pf,
            "no_final_rate": nf,
            "truncation_rate": tr,
            "avg_output_tokens": at,
        }
        return {
            "name": name,
            "dir": f"artifacts/{name}",
            "metrics": metrics,
            "prompt_ids": list(prompt_ids),
            "correct": list(correct),
            "risk_score": risk_score(
                correct_rate=rate,
                parse_fail_rate=pf,
                no_final_rate=nf,
                truncation_rate=tr,
                avg_output_tokens=at,
            ),
        }

    # slot1: clean metrics (low penalties) → risk_score ~= 0.70 - 0.02 - ~0.02 = ~0.656
    # hedge: higher correctness but noisy outputs → risk_score ~= 0.80 - 0.17 - ~0.04 = ~0.59
    return [
        _mk("adapter_stage3_bestproxy", s1_correct, 0.010, 0.004, 0.002, 1100.0),
        _mk("adapter_stage2_bestproxy", hedge_correct, 0.080, 0.030, 0.020, 2000.0),
        _mk("adapter_stage2_ckpt_1050", low_correct, 0.010, 0.004, 0.002, 1100.0),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pick the two final Kaggle submissions from a candidate pool."
            "  Slot-1 = risk_score argmax; Slot-2 = paired hedge argmax"
            " within the top-band (GPT-5.4 Pro Round 4)."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--candidates-root",
        help="Directory containing all candidate adapter subdirectories.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        dest="candidates",
        help=(
            "Name of a candidate directory under --candidates-root."
            "  Repeat to add multiple candidates (e.g. `--candidate a"
            " --candidate b`)."
        ),
    )
    parser.add_argument(
        "--output",
        default="final_selection.json",
        help="Destination for the selection report JSON.",
    )
    parser.add_argument(
        "--top-band",
        type=float,
        default=0.005,
        dest="top_band",
        help=(
            "Slot-2 must have risk_score >= slot1 - top_band (default 0.005)."
        ),
    )
    parser.add_argument(
        "--require-min-hedge",
        type=int,
        default=1,
        dest="require_min_hedge",
        help=(
            "Minimum hedge score (wins - losses) required for slot-2."
            "  Below this threshold the selector falls back to the"
            " second-highest risk_score and flags fallback_hedge (default 1)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Skip disk I/O; exercise the selector on two synthetic"
            " candidate pools (fallback path + hedge-wins path)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.dry_run:
        print("[final_submit_selector] dry-run scenario A: no hedge wins",
              flush=True)
        report_a = select_two_submissions(
            _dry_run_candidates(),
            top_band=args.top_band,
            require_min_hedge=args.require_min_hedge,
        )
        assert report_a["slot1"]["name"] == "adapter_stage3_bestproxy", report_a
        assert report_a["fallback_hedge"] is True, report_a
        print(f"  slot1 = {report_a['slot1']['name']} "
              f"(risk={report_a['slot1']['risk_score']:.4f})")
        print(f"  slot2 = {report_a['slot2']['name']} (fallback)")

        print("[final_submit_selector] dry-run scenario B: hedge wins",
              flush=True)
        report_b = select_two_submissions(
            _dry_run_candidates_hedge_wins(),
            top_band=0.200,  # relax band so both candidates qualify
            require_min_hedge=1,
        )
        assert report_b["slot1"]["name"] == "adapter_stage3_bestproxy", report_b
        assert report_b["slot2"]["name"] == "adapter_stage2_bestproxy", report_b
        assert report_b["fallback_hedge"] is False, report_b
        assert report_b["slot2"]["hedge_vs_slot1"]["score"] >= 1, report_b
        print(f"  slot1 = {report_b['slot1']['name']}")
        print(f"  slot2 = {report_b['slot2']['name']} "
              f"(hedge score={report_b['slot2']['hedge_vs_slot1']['score']})")

        print("[final_submit_selector] dry-run OK", flush=True)
        return

    if not args.candidates_root or not args.candidates:
        parser.error("--candidates-root and at least one --candidate are required")

    root = Path(args.candidates_root)
    candidates: list[dict[str, Any]] = []
    for name in args.candidates:
        print(f"[final_submit_selector] loading {name} ...", flush=True)
        candidates.append(_load_candidate(root, name))

    report = select_two_submissions(
        candidates,
        top_band=args.top_band,
        require_min_hedge=args.require_min_hedge,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[final_submit_selector] wrote {output_path}", flush=True)
    print(
        f"[final_submit_selector] slot1={report['slot1']['name']} "
        f"slot2={report['slot2']['name'] if report['slot2'] else 'NONE'} "
        f"fallback_hedge={report['fallback_hedge']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
