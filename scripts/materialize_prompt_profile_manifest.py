from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import load_jsonl, write_json, write_jsonl  # noqa: E402
from src.research.prompt_profiles import materialize_prompt_profile_row, profile_summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize a local eval manifest with a research prompt profile.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)

    rows = [
        materialize_prompt_profile_row(row, args.profile)
        for row in load_jsonl(args.input)
    ]
    write_jsonl(args.output, rows)
    summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")
    write_json(
        summary_path,
        {
            "input": str(args.input),
            "output": str(args.output),
            "profile": args.profile,
            "rows": len(rows),
            "profiles": profile_summary(),
        },
    )
    print(json.dumps({"output": str(args.output), "rows": len(rows), "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

