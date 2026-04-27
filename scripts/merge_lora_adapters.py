from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.adapter_merge import (  # noqa: E402
    DEFAULT_MAX_SUBMIT_RANK,
    clean_output_dir,
    merge_lora_adapters,
    parse_adapter_spec,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge PEFT LoRA adapters into a submit-safe adapter directory.")
    parser.add_argument(
        "--adapter",
        action="append",
        required=True,
        help="Adapter source as name=path:weight. Repeat for each source.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output merged adapter directory.")
    parser.add_argument("--method", choices=["linear", "svd-rank32"], default="linear")
    parser.add_argument("--target-rank", type=int, default=DEFAULT_MAX_SUBMIT_RANK)
    parser.add_argument("--max-submit-rank", type=int, default=DEFAULT_MAX_SUBMIT_RANK)
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before writing.")
    args = parser.parse_args(argv)

    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    adapters = []
    for raw in args.adapter:
        spec = parse_adapter_spec(raw)
        path = spec.path if spec.path.is_absolute() else REPO_ROOT / spec.path
        adapters.append(type(spec)(name=spec.name, path=path, weight=spec.weight))
    if args.clean:
        clean_output_dir(output)
    manifest = merge_lora_adapters(
        adapters=adapters,
        output_dir=output,
        method=args.method,
        target_rank=int(args.target_rank),
        max_submit_rank=int(args.max_submit_rank),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
