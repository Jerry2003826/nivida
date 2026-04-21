from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import read_yaml, write_json
from src.experiments.eval_competition_replica import evaluate_replica
from src.student.adapter_submission_budget import (
    ensure_submission_budget_safe,
    estimate_submission_budget,
)
from src.student.inference import run_inference
from src.student.package_submission import (
    build_submission_zip,
    read_adapter_rank,
    read_adapter_target_modules,
    validate_adapter_dir,
)


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
    max_new_tokens: int | None = None,
) -> dict[str, object]:
    config = read_yaml(config_path)
    adapter_files = validate_adapter_dir(adapter_dir)
    adapter_rank = read_adapter_rank(adapter_dir)
    if adapter_rank is None:
        raise SubmissionValidationError(
            "Adapter rank could not be read from adapter_config.json; "
            "ensure the adapter was saved via peft.PeftModel.save_pretrained"
        )
    if adapter_rank > 32:
        raise SubmissionValidationError(f"Adapter rank must be <= 32, got {adapter_rank}")
    if labels and not smoke_input:
        raise SubmissionValidationError("--labels requires --smoke-input so local_eval can be computed")
    if package_output and not smoke_input:
        raise SubmissionValidationError("--package-output requires --smoke-input")
    if package_output and not labels:
        raise SubmissionValidationError("--package-output requires --labels")

    payload: dict[str, object] = {
        "adapter_files": adapter_files,
        "adapter_rank": adapter_rank,
        "rank_ok": True,
    }
    adapter_target_modules = read_adapter_target_modules(adapter_dir)
    submission_budget = estimate_submission_budget(
        config,
        target_modules=adapter_target_modules,
        rank=adapter_rank,
    )
    payload["submission_budget"] = submission_budget
    if package_output:
        try:
            ensure_submission_budget_safe(
                config,
                target_modules=adapter_target_modules,
                rank=adapter_rank,
            )
        except ValueError as exc:
            if submission_budget.get("status") == "over_limit":
                raise SubmissionValidationError(
                    "Projected submission zip would exceed Kaggle's 1 GB limit: "
                    f"{submission_budget.get('projected_submission_zip_bytes')}"
                ) from exc
            if submission_budget.get("status") == "unknown":
                raise SubmissionValidationError(
                    "Submission budget cannot be estimated for this model: "
                    f"{submission_budget.get('reason', 'unknown model')}. "
                    "Refusing to package submission.zip without a budget guard."
                ) from exc
            raise SubmissionValidationError(str(exc)) from exc

    predictions_path = None
    if smoke_input:
        predictions_path = Path(output_path).with_name("validation_smoke_predictions.jsonl")
        run_inference(
            config,
            input_path=smoke_input,
            adapter_dir=adapter_dir,
            output_path=predictions_path,
            max_new_tokens=max_new_tokens,
        )
        payload["smoke_predictions_path"] = str(predictions_path)

    if predictions_path and labels:
        payload["local_eval"] = evaluate_replica(
            prediction_path=predictions_path,
            label_path=labels,
            split_path=splits,
            require_complete_coverage=True,
        )

    if package_output:
        if "local_eval" not in payload:
            raise SubmissionValidationError(
                "--package-output requires a successful smoke inference + local_eval first"
            )
        packaged = build_submission_zip(adapter_dir, package_output)
        payload["package_output"] = str(packaged)
        if packaged.exists():
            payload["package_size_bytes"] = packaged.stat().st_size

        # Roundtrip verification: unzip into a tempdir and make sure the zip
        # actually contains exactly the allowlisted files and the adapter
        # config is valid JSON with a sane target_modules / r.  We deliberately
        # avoid loading the base model here (30B base takes ~3min and a lot of
        # VRAM); the upstream validate already exercised the adapter via
        # run_inference earlier, so a fresh zip with correct config is enough.
        roundtrip = _verify_submission_zip_roundtrip(packaged)
        payload["submission_zip_roundtrip"] = roundtrip

    write_json(output_path, payload)
    return payload


def _verify_submission_zip_roundtrip(zip_path: Path) -> dict[str, object]:
    """Unzip submission.zip and verify Kaggle-side loader would succeed.

    Checks:
      * zip is flat (no nested directories)
      * exactly one adapter_config.json at root
      * exactly one adapter_model.safetensors or adapter_model.bin at root
      * no stray files (optimizer.pt, README, tokenizer, etc.)
      * adapter_config.json parses and has positive `r`
      * target_modules present (PEFT loader respects this, not a hardcoded
        LoraConfig)
    """
    with zipfile.ZipFile(zip_path, "r") as archive:
        namelist = archive.namelist()

    nested = [n for n in namelist if "/" in n]
    if nested:
        raise SubmissionValidationError(
            f"submission.zip must be flat; nested entries found: {nested}"
        )

    allowed_names = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
    }
    stray = [n for n in namelist if n not in allowed_names]
    if stray:
        raise SubmissionValidationError(
            f"submission.zip contains disallowed files: {stray}. "
            "Only adapter_config.json + adapter_model.safetensors (or .bin) are permitted."
        )

    if "adapter_config.json" not in namelist:
        raise SubmissionValidationError(
            "submission.zip missing adapter_config.json"
        )

    has_safetensors = "adapter_model.safetensors" in namelist
    has_bin = "adapter_model.bin" in namelist
    if not (has_safetensors or has_bin):
        raise SubmissionValidationError(
            "submission.zip missing adapter_model.safetensors / adapter_model.bin"
        )
    if has_safetensors and has_bin:
        raise SubmissionValidationError(
            "submission.zip contains both safetensors and bin weight files; "
            "pick one"
        )

    # Parse adapter_config.json from the zip to confirm Kaggle loader will
    # respect our target_modules / r.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(tmp_root)
        config_payload = json.loads((tmp_root / "adapter_config.json").read_text(encoding="utf-8"))

    rank = config_payload.get("r")
    if not isinstance(rank, int) or rank <= 0 or rank > 32:
        raise SubmissionValidationError(
            f"Invalid LoRA rank in adapter_config.json: {rank!r} (must be 1..32)"
        )
    target_modules = config_payload.get("target_modules")
    if not target_modules:
        raise SubmissionValidationError(
            "adapter_config.json has no target_modules; Kaggle loader would "
            "reinitialize from defaults."
        )

    return {
        "zip_path": str(zip_path),
        "namelist": sorted(namelist),
        "zip_bytes": int(zip_path.stat().st_size),
        "adapter_config_r": rank,
        "adapter_config_target_modules": target_modules,
        "adapter_config_lora_alpha": config_payload.get("lora_alpha"),
        "adapter_config_peft_type": config_payload.get("peft_type"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an adapter before packaging a submission zip.")
    parser.add_argument("--config", default="configs/train_stage3_repair.yaml")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", default="artifacts/submission_validation.json")
    parser.add_argument("--smoke-input")
    parser.add_argument("--labels")
    parser.add_argument("--splits")
    parser.add_argument("--package-output")
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()

    validate_submission(
        config_path=args.config,
        adapter_dir=args.adapter_dir,
        output_path=args.output,
        smoke_input=args.smoke_input,
        labels=args.labels,
        splits=args.splits,
        package_output=args.package_output,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
