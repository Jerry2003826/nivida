from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.solver_finalizer import apply_solver_assisted_finalizer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply high-confidence CPU solver overrides to raw model predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("data/processed/eval/solver_assisted_finalizer.json"))
    parser.add_argument("--prediction-key", default="generation")
    parser.add_argument("--min-confidence", type=float, default=0.88)
    parser.add_argument("--min-support-coverage", type=float, default=1.0)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--families", default="equation,bit")
    args = parser.parse_args(argv)
    families = {token.strip() for token in args.families.split(",") if token.strip()}
    report = apply_solver_assisted_finalizer(
        predictions_path=args.predictions,
        labels_path=args.labels,
        output_path=args.output,
        report_path=args.report,
        prediction_key=args.prediction_key,
        min_confidence=float(args.min_confidence),
        min_support_coverage=float(args.min_support_coverage),
        beam_width=int(args.beam_width),
        max_depth=int(args.max_depth),
        top_k=int(args.top_k),
        override_families=families,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

