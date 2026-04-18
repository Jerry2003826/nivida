from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import read_json, write_json  # noqa: E402


def check_subtype_branch_promotion(
    *,
    promotion_json: str | Path,
    allow_unpromoted: bool = False,
) -> dict[str, Any]:
    source = Path(promotion_json)
    if not source.is_file():
        raise SystemExit(f"promotion json not found: {source}")

    payload = read_json(source)
    if not isinstance(payload, dict):
        raise SystemExit(f"promotion json must be a JSON object: {source}")

    promote = bool(payload.get("promote", False))
    if not promote and not allow_unpromoted:
        raise SystemExit(
            "Subtype-rescue branch did not qualify for stage3 promotion. "
            "Set ALLOW_UNPROMOTED_SUBTYPE_STAGE3=1 only for a deliberate manual experiment."
        )

    result = dict(payload)
    result["promotion_json"] = str(source)
    result["promotion_json_found"] = True
    result["override_used"] = bool(allow_unpromoted and not promote)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether the subtype-rescue branch is allowed to enter stage3."
    )
    parser.add_argument(
        "--promotion-json",
        default="data/processed/stage2_subtype_rescue_promotion.json",
    )
    parser.add_argument("--allow-unpromoted", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    payload = check_subtype_branch_promotion(
        promotion_json=args.promotion_json,
        allow_unpromoted=args.allow_unpromoted,
    )
    if args.output:
        write_json(args.output, payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
