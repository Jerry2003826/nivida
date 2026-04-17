from __future__ import annotations
# ruff: noqa: E402

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import read_yaml, write_json
from src.experiments.eval_competition_replica import evaluate_replica
from src.student.inference import run_inference
from src.student.package_submission import build_submission_zip, read_adapter_rank, validate_adapter_dir


class SubmissionValidationError(ValueError):
    """Raised when packaging is requested before a safe local validation gate."""


def validate_submission(
    *,
    config_path: str | Path,
    adapter_dir: str | Path,
    output_path: str | Path,
    smoke_input: str | Path | None = None,
    labels: str | Path | None = None,
    splits: str | Path | None = None,
    package_output: str | Path | None = None,
) -> dict[str, object]:
    config = read_yaml(config_path)
    adapter_files = validate_adapter_dir(adapter_dir)
    adapter_rank = read_adapter_rank(adapter_dir)
    if adapter_rank is not None and adapter_rank > 32:
        raise SubmissionValidationError(f"Adapter rank must be <= 32, got {adapter_rank}")
    if labels and not smoke_input:
        raise SubmissionValidationError("--labels requires --smoke-input so local_eval can be computed")

    payload: dict[str, object] = {
        "adapter_files": adapter_files,
        "adapter_rank": adapter_rank,
        "rank_ok": True,
    }

    predictions_path = None
    if smoke_input:
        predictions_path = Path(output_path).with_name("validation_smoke_predictions.jsonl")
        run_inference(
            config,
            input_path=smoke_input,
            adapter_dir=adapter_dir,
            output_path=predictions_path,
        )
        payload["smoke_predictions_path"] = str(predictions_path)

    if predictions_path and labels:
        payload["local_eval"] = evaluate_replica(
            prediction_path=predictions_path,
            label_path=labels,
            split_path=splits,
        )

    if package_output:
        if "local_eval" not in payload:
            raise SubmissionValidationError(
                "--package-output requires a successful smoke inference + local_eval first"
            )
        packaged = build_submission_zip(adapter_dir, package_output)
        payload["package_output"] = str(packaged)

    write_json(output_path, payload)
    return payload


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

    validate_submission(
        config_path=args.config,
        adapter_dir=args.adapter_dir,
        output_path=args.output,
        smoke_input=args.smoke_input,
        labels=args.labels,
        splits=args.splits,
        package_output=args.package_output,
    )


if __name__ == "__main__":
    main()
