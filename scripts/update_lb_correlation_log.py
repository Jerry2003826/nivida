from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.lb_correlation import append_correlation_entry, parse_merge_weights  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append a Kaggle public/local exact correlation entry.")
    parser.add_argument("--log", type=Path, default=Path("data/processed/eval/lb_correlation_log.json"))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--public-score", type=float)
    parser.add_argument("--exact-report", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--training-recipe", default="")
    parser.add_argument("--merge-weights", default="", help="Comma-separated source=weight pairs.")
    parser.add_argument("--submission-id", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    log = args.log if args.log.is_absolute() else REPO_ROOT / args.log
    exact_report = None if args.exact_report is None else (
        args.exact_report if args.exact_report.is_absolute() else REPO_ROOT / args.exact_report
    )
    adapter_path = None if args.adapter_path is None else (
        args.adapter_path if args.adapter_path.is_absolute() else REPO_ROOT / args.adapter_path
    )
    payload = append_correlation_entry(
        log_path=log,
        candidate=args.candidate,
        public_score=args.public_score,
        exact_report=exact_report,
        adapter_path=adapter_path,
        training_recipe=args.training_recipe,
        merge_weights=parse_merge_weights(args.merge_weights),
        submission_id=args.submission_id,
        notes=args.notes,
    )
    print(json.dumps({"log": str(args.log), "entries": len(payload["entries"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
