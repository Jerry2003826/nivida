from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.candidate_registry import (  # noqa: E402
    build_default_registry,
    canonical_registry_matches,
    validate_registry,
    write_default_registry,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write or validate the research-breakout candidate registry.")
    parser.add_argument("--output", type=Path, default=Path("configs/research_breakout_candidates.json"))
    parser.add_argument("--check", action="store_true", help="Validate that --output matches the canonical registry.")
    parser.add_argument("--check-paths", action="store_true", help="Also require referenced adapter/config paths to exist.")
    args = parser.parse_args(argv)

    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    if args.check:
        if not canonical_registry_matches(output):
            print(json.dumps({"status": "fail", "reason": "registry drift", "path": str(args.output)}, indent=2))
            return 1
        payload = build_default_registry()
    else:
        payload = write_default_registry(output)

    errors = validate_registry(payload, repo_root=REPO_ROOT, check_paths=bool(args.check_paths))
    status = "pass" if not errors else "fail"
    print(json.dumps({"status": status, "path": str(args.output), "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

