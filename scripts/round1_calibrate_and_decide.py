"""Calibrate Stage2 proxy correct_rate against the Kaggle LB, then emit the
Round 1 go / no-go decision.

Motivation
----------
The short-circuit watchdog will kill the recover pipeline as soon as
``stage2_proxy_valid_eval.json`` (hard-triad only; 709 rows, cipher + bit +
equation) is on disk and submit the Stage2 selected adapter to Kaggle. We do
**not** get the Step 2 all-family correct-rate before submission. The LB score
that comes back ~5–30 min later is the only cross-check we have for the full
distribution.

This script fuses those two numbers:

* ``correct_rate_hard`` — from ``stage2_proxy_valid_eval.json``.
* ``public_lb`` — from the Kaggle CLI for the most recent successful submission.

It derives an implied ``correct_rate_easy`` under a fixed-prior model and
prints a Round 1 track recommendation. Output is written as JSON so the
tracking cron can pick it up.

Usage
-----

    python scripts/round1_calibrate_and_decide.py \\
        --hard-eval data/processed/stage2_proxy_valid_eval.json \\
        --all-eval data/processed/stage2_proxy_all_valid_eval.json  # optional \\
        --lb-score 0.72                                              # or --kaggle-lookup \\
        --output logs/round1_decision.json

Invariants
----------
* ``correct_rate_hard`` and ``correct_rate_easy`` are always in ``[0, 1]``.
* ``hard_family_fraction_on_lb`` defaults to 0.45 (derived from training-set
  hard triad ratio + synth_hard_triads.yaml family weights). Override via
  ``--hard-fraction``.
* If the implied ``correct_rate_easy`` falls outside ``[0, 1]`` the script
  reports the prior as unreliable and downgrades to an LB-only decision.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants (calibrated from repo metadata — not blind guesses)
# ---------------------------------------------------------------------------

#: Hard-family (cipher + bit + equation) share of the LB test set.
#:
#: Rationale: the official training set has roughly this fraction under the
#: family_weights in configs/synth_hard_triads.yaml (bit=1.4, cipher=1.4,
#: equation=1.4 vs. numeral/unit/gravity at 0.4 each), with per-record
#: balancing in split_builder.py. See docs/ for full derivation.
DEFAULT_HARD_FRACTION: float = 0.45

#: LB bands that map directly onto a Round 1 track.
LB_TARGET: float = 0.86
LB_DELTA_SVD_FLOOR: float = 0.80
LB_LIGHT_ROUND1_FLOOR: float = 0.75


@dataclass
class CalibrationResult:
    correct_rate_hard: float
    correct_rate_all: float | None
    public_lb: float
    hard_fraction: float
    implied_correct_rate_easy: float | None
    prior_is_reliable: bool
    delta_lb_minus_hard: float
    bottleneck: str
    round1_track: str
    round1_rationale: str
    notes: list[str]


# ---------------------------------------------------------------------------
# Load utilities
# ---------------------------------------------------------------------------


def _load_correct_rate(eval_path: Path) -> float:
    with eval_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rate = payload.get("competition_correct_rate")
    if rate is None:
        raise ValueError(
            f"{eval_path} has no 'competition_correct_rate' key "
            f"(found keys: {list(payload.keys())})"
        )
    rate = float(rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(
            f"competition_correct_rate out of range in {eval_path}: {rate}"
        )
    return rate


def _lookup_latest_public_lb(competition: str) -> tuple[float | None, str | None]:
    """Return the highest ``publicScore`` across COMPLETE submissions, with
    the matching submission date. Returns ``(None, None)`` if the CLI call
    fails or no completed submission exists.
    """
    try:
        output = subprocess.check_output(
            [
                "kaggle",
                "competitions",
                "submissions",
                "-c",
                competition,
                "--csv",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None, None

    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) < 2:
        return None, None

    header = [h.strip() for h in lines[0].split(",")]
    try:
        date_idx = header.index("date")
        status_idx = header.index("status")
        score_idx = header.index("publicScore")
    except ValueError:
        return None, None

    best_score: float | None = None
    best_date: str | None = None
    for row in lines[1:]:
        cols = [c.strip() for c in row.split(",")]
        if len(cols) <= max(date_idx, status_idx, score_idx):
            continue
        if "COMPLETE" not in cols[status_idx]:
            continue
        try:
            score = float(cols[score_idx])
        except ValueError:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_date = cols[date_idx]
    return best_score, best_date


# ---------------------------------------------------------------------------
# Core calibration + decision
# ---------------------------------------------------------------------------


def calibrate(
    *,
    correct_rate_hard: float,
    public_lb: float,
    correct_rate_all: float | None = None,
    hard_fraction: float = DEFAULT_HARD_FRACTION,
) -> CalibrationResult:
    """Derive ``correct_rate_easy`` under the fixed-prior model and assign
    the Round 1 track.

    The implied easy-family rate is

        correct_rate_easy = (LB - p_hard * correct_hard) / (1 - p_hard)

    If that value is outside ``[0, 1]`` the prior is incompatible with the
    observed scores, and the decision falls back to LB bands only.
    """
    notes: list[str] = []
    implied_easy: float | None = None
    prior_is_reliable = True

    # Prefer the measured all-family rate if available.
    if correct_rate_all is not None:
        implied_easy = (
            correct_rate_all - hard_fraction * correct_rate_hard
        ) / max(1.0 - hard_fraction, 1e-6)
        notes.append(
            "Using measured all-family correct rate to derive easy-family rate."
        )
    else:
        implied_easy = (
            public_lb - hard_fraction * correct_rate_hard
        ) / max(1.0 - hard_fraction, 1e-6)
        notes.append(
            "No Step 2 all-family eval on disk; deriving easy-family rate "
            "from LB score under the hard_fraction prior."
        )

    if implied_easy is None or not 0.0 <= implied_easy <= 1.0:
        prior_is_reliable = False
        notes.append(
            f"Implied correct_rate_easy={implied_easy} is out of [0, 1]; "
            "prior hard_fraction may be wrong. Falling back to LB-only "
            "decision, treating bottleneck as 'unknown'."
        )
        implied_easy = None

    # Bottleneck analysis.
    if not prior_is_reliable or implied_easy is None:
        bottleneck = "unknown"
    elif correct_rate_hard < implied_easy - 0.08:
        bottleneck = "hard"
    elif implied_easy < correct_rate_hard - 0.08:
        bottleneck = "easy"
    else:
        bottleneck = "balanced"

    # Round 1 track decision.
    if public_lb >= LB_TARGET:
        track = "DONE"
        rationale = (
            f"Public LB {public_lb:.3f} ≥ target {LB_TARGET}. Delete both "
            "monitoring crons and finalize."
        )
    elif public_lb >= LB_DELTA_SVD_FLOOR:
        track = "delta_svd_only"
        rationale = (
            f"Public LB {public_lb:.3f} in [{LB_DELTA_SVD_FLOOR}, {LB_TARGET}). "
            "Low-effort merge-only (delta-SVD) is the right risk/reward."
        )
    elif public_lb >= LB_LIGHT_ROUND1_FLOOR:
        track = "distill_plus_delta_svd"
        rationale = (
            f"Public LB {public_lb:.3f} in [{LB_LIGHT_ROUND1_FLOOR}, "
            f"{LB_DELTA_SVD_FLOOR}). Skip Stage 3 longtrace; run GCD teacher "
            "distill + delta-SVD merge."
        )
    else:
        if bottleneck == "hard":
            track = "full_round1_hard_focus"
            rationale = (
                f"Public LB {public_lb:.3f} < {LB_LIGHT_ROUND1_FLOOR} and hard "
                "families underperform easy by ≥8pp. Run full Round 1: "
                "distill + GCD teacher gate + MoE expert profiler + Stage 3 "
                "longtrace, oversampling cipher/bit/equation."
            )
        elif bottleneck == "easy":
            track = "synth_easy_families_plus_delta_svd"
            rationale = (
                f"Public LB {public_lb:.3f} < {LB_LIGHT_ROUND1_FLOOR} but easy "
                "families are the bottleneck. Synthesize more "
                "numeral/unit/gravity data and run delta-SVD merge; skip Stage "
                "3 longtrace which targets hard traces."
            )
        else:
            track = "full_round1_balanced"
            rationale = (
                f"Public LB {public_lb:.3f} < {LB_LIGHT_ROUND1_FLOOR}; hard/easy "
                "gap inconclusive. Run full Round 1 without family-specific "
                "reweighting."
            )

    return CalibrationResult(
        correct_rate_hard=correct_rate_hard,
        correct_rate_all=correct_rate_all,
        public_lb=public_lb,
        hard_fraction=hard_fraction,
        implied_correct_rate_easy=implied_easy,
        prior_is_reliable=prior_is_reliable,
        delta_lb_minus_hard=public_lb - correct_rate_hard,
        bottleneck=bottleneck,
        round1_track=track,
        round1_rationale=rationale,
        notes=notes,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate Stage 2 proxy correct rate against the Kaggle public "
            "LB and emit the Round 1 track decision as JSON."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--hard-eval",
        type=Path,
        required=True,
        help="Path to stage2_proxy_valid_eval.json (hard-triad).",
    )
    parser.add_argument(
        "--all-eval",
        type=Path,
        default=None,
        help=(
            "Optional path to stage2_proxy_all_valid_eval.json; when provided "
            "the calibration uses the measured all-family rate instead of "
            "inferring from LB under a prior."
        ),
    )
    parser.add_argument(
        "--lb-score",
        type=float,
        default=None,
        help=(
            "Public LB score to calibrate against. If omitted, "
            "--kaggle-lookup must be set to fetch from the Kaggle CLI."
        ),
    )
    parser.add_argument(
        "--kaggle-lookup",
        action="store_true",
        help=(
            "Resolve the most recent COMPLETE public score via the kaggle "
            "CLI for --competition."
        ),
    )
    parser.add_argument(
        "--competition",
        default="nvidia-nemotron-model-reasoning-challenge",
        help="Kaggle competition slug for --kaggle-lookup.",
    )
    parser.add_argument(
        "--hard-fraction",
        type=float,
        default=DEFAULT_HARD_FRACTION,
        help=(
            "Hard-family share of the LB test distribution. Override only "
            "if you have a better prior."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/round1_decision.json"),
        help="Where to write the decision JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    correct_rate_hard = _load_correct_rate(args.hard_eval)
    correct_rate_all: float | None = None
    if args.all_eval is not None and args.all_eval.exists():
        correct_rate_all = _load_correct_rate(args.all_eval)

    lb_score: float | None = args.lb_score
    lb_date: str | None = None
    if lb_score is None:
        if not args.kaggle_lookup:
            parser.error(
                "Either --lb-score or --kaggle-lookup must be provided."
            )
        lb_score, lb_date = _lookup_latest_public_lb(args.competition)
        if lb_score is None:
            parser.error(
                "Kaggle CLI lookup failed or returned no COMPLETE submission."
            )

    if not 0.0 <= lb_score <= 1.0:
        parser.error(f"--lb-score out of range: {lb_score}")
    if not 0.0 <= args.hard_fraction <= 1.0:
        parser.error(f"--hard-fraction out of range: {args.hard_fraction}")

    result = calibrate(
        correct_rate_hard=correct_rate_hard,
        public_lb=lb_score,
        correct_rate_all=correct_rate_all,
        hard_fraction=args.hard_fraction,
    )

    output_payload: dict[str, Any] = asdict(result)
    output_payload["public_lb_submission_date"] = lb_date
    output_payload["inputs"] = {
        "hard_eval_path": str(args.hard_eval),
        "all_eval_path": str(args.all_eval) if args.all_eval else None,
        "competition": args.competition,
        "hard_fraction_source": (
            "cli-override"
            if args.hard_fraction != DEFAULT_HARD_FRACTION
            else "default-from-synth-weights"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, sort_keys=True)

    # Human-readable stdout summary (cron-friendly).
    print("=== Round 1 calibration ===")
    print(f"  correct_rate_hard   : {result.correct_rate_hard:.4f}")
    if result.correct_rate_all is not None:
        print(f"  correct_rate_all    : {result.correct_rate_all:.4f}")
    print(f"  public_lb           : {result.public_lb:.4f}")
    print(f"  delta (LB - hard)   : {result.delta_lb_minus_hard:+.4f}")
    if result.implied_correct_rate_easy is not None:
        print(
            f"  implied_easy        : {result.implied_correct_rate_easy:.4f} "
            f"(prior reliable: {result.prior_is_reliable})"
        )
    print(f"  bottleneck          : {result.bottleneck}")
    print(f"  round1_track        : {result.round1_track}")
    print(f"  rationale           : {result.round1_rationale}")
    for note in result.notes:
        print(f"  note                : {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
