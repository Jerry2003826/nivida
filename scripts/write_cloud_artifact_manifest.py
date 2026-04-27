from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import write_json  # noqa: E402
from src.research.artifact_manifest import build_cloud_artifact_manifest  # noqa: E402


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a cloud eval/training artifact manifest.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-inputs", required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = build_cloud_artifact_manifest(
        repo_root=REPO_ROOT,
        out_dir=args.out_dir,
        eval_inputs=_split_csv(args.eval_inputs),
        candidates=list(args.candidate),
        preflight_path=args.preflight,
    )
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "prediction_line_counts": payload["prediction_line_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

