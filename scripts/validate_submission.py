from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import write_json
from src.experiments.eval_competition_replica import evaluate_replica
from src.student.inference import run_inference
from src.student.package_submission import build_submission_zip, read_adapter_rank, validate_adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an adapter before packaging a submission zip.")
    parser.add_argument("--config", default="configs/train_stage2_selected_trace.yaml")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", default="artifacts/submission_validation.json")
    parser.add_argument("--smoke-input")
    parser.add_argument("--labels")
    parser.add_argument("--splits")
    parser.add_argument("--package-output")
    args = parser.parse_args()

    from src.common.io import read_yaml

    config = read_yaml(args.config)
    adapter_files = validate_adapter_dir(args.adapter_dir)
    adapter_rank = read_adapter_rank(args.adapter_dir)
    payload: dict[str, object] = {
        "adapter_files": adapter_files,
        "adapter_rank": adapter_rank,
        "rank_ok": adapter_rank is None or adapter_rank <= 32,
    }

    predictions_path = None
    if args.smoke_input:
        predictions_path = Path(args.output).with_name("validation_smoke_predictions.jsonl")
        run_inference(
            config,
            input_path=args.smoke_input,
            adapter_dir=args.adapter_dir,
            output_path=predictions_path,
            max_new_tokens=32,
        )
        payload["smoke_predictions_path"] = str(predictions_path)

    if predictions_path and args.labels:
        payload["local_eval"] = evaluate_replica(
            prediction_path=predictions_path,
            label_path=args.labels,
            split_path=args.splits,
        )

    if args.package_output and payload["rank_ok"]:
        packaged = build_submission_zip(args.adapter_dir, args.package_output)
        payload["package_output"] = str(packaged)

    write_json(args.output, payload)


if __name__ == "__main__":
    main()
