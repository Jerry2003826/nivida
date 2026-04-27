from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.weak_family_data import build_research_rescue_data  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build weak-family research SFT recipes from answer-only/short-trace data.")
    parser.add_argument("--answer-train", type=Path, default=Path("data/processed/stage2_answer_only_train.jsonl"))
    parser.add_argument("--answer-valid", type=Path, default=Path("data/processed/stage2_answer_only_valid.jsonl"))
    parser.add_argument("--short-train", type=Path, default=Path("data/processed/stage2_short_trace_train.jsonl"))
    parser.add_argument("--short-valid", type=Path, default=Path("data/processed/stage2_short_trace_valid.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/research_breakout"))
    parser.add_argument("--recipe", action="append", help="Optional recipe filter. Repeatable.")
    args = parser.parse_args(argv)

    report = build_research_rescue_data(
        answer_train=args.answer_train,
        answer_valid=args.answer_valid,
        short_train=args.short_train,
        short_valid=args.short_valid,
        out_dir=args.out_dir,
        recipes=args.recipe,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

