from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import read_yaml, write_json
from src.student.audit_target_modules import audit_modules


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect model modules matched by the LoRA target pattern.")
    parser.add_argument("--config", default="configs/train_stage2_selected_trace.yaml")
    parser.add_argument("--output", default="artifacts/target_module_audit.json")
    args = parser.parse_args()

    payload = audit_modules(read_yaml(args.config))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, payload)


if __name__ == "__main__":
    main()
