from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import read_json, read_yaml, write_json
from src.student.package_submission import validate_adapter_dir
from scripts.probe_adapter_submission_size import probe_adapter_submission_size
from scripts.select_final_adapter import choose_adapter, copy_adapter, load_eval
from scripts.validate_submission import validate_submission


DEFAULT_STAGE2_ADAPTER_DIR = Path("artifacts/adapter_stage2_bestproxy")
DEFAULT_STAGE3_ADAPTER_DIR = Path("artifacts/adapter_stage3_bestproxy")
DEFAULT_STAGE2_HARD_EVAL = Path("data/processed/stage2_bestproxy_hard_eval.json")
DEFAULT_STAGE2_ALL_EVAL = Path("data/processed/stage2_bestproxy_all_eval.json")
DEFAULT_STAGE3_HARD_EVAL = Path("data/processed/stage3_bestproxy_hard_eval.json")
DEFAULT_STAGE3_ALL_EVAL = Path("data/processed/stage3_bestproxy_all_eval.json")
DEFAULT_OUTPUT_ADAPTER_DIR = Path("artifacts/adapter_final_selected")
DEFAULT_SELECTION_JSON = Path("data/processed/final_adapter_selection.json")
DEFAULT_PROBE_JSON = Path("artifacts/adapter_submission_probe.json")
DEFAULT_VALIDATION_JSON = Path("artifacts/submission_validation.json")
DEFAULT_SUMMARY_JSON = Path("artifacts/final_acceptance_report.json")
DEFAULT_SUBMISSION_ZIP = Path("submission.zip")
DEFAULT_CONFIG = Path("configs/train_stage3_repair.yaml")
DEFAULT_SMOKE_INPUT = Path("data/processed/official_train_tagged.jsonl")
DEFAULT_LABELS = Path("data/processed/official_train_tagged.jsonl")
DEFAULT_SPLITS = Path("data/splits/official/splits.json")
DEFAULT_MAX_NEW_TOKENS = 2048


def _required_file(path: str | Path) -> Path:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Required file not found: {target}")
    return target


def _load_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.is_file():
        return None
    payload = read_json(target)
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Expected JSON object at {target}, got {type(payload).__name__}")


def _nested_get(mapping: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def validate_acceptance_inputs(
    *,
    stage2_adapter_dir: str | Path,
    stage3_adapter_dir: str | Path,
    stage2_hard_eval: str | Path,
    stage2_all_eval: str | Path,
    stage3_hard_eval: str | Path,
    stage3_all_eval: str | Path,
) -> None:
    validate_adapter_dir(stage2_adapter_dir)
    validate_adapter_dir(stage3_adapter_dir)
    _required_file(stage2_hard_eval)
    _required_file(stage2_all_eval)
    _required_file(stage3_hard_eval)
    _required_file(stage3_all_eval)


def run_final_adapter_selection(
    *,
    stage2_hard_eval: str | Path,
    stage2_all_eval: str | Path,
    stage2_adapter_dir: str | Path,
    stage3_hard_eval: str | Path,
    stage3_all_eval: str | Path,
    stage3_adapter_dir: str | Path,
    output_adapter_dir: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    stage2_hard = load_eval(str(stage2_hard_eval))
    stage2_all = load_eval(str(stage2_all_eval))
    stage3_hard = load_eval(str(stage3_hard_eval))
    stage3_all = load_eval(str(stage3_all_eval))

    decision = choose_adapter(stage2_all, stage3_all, stage2_hard, stage3_hard)
    selected_source_dir = (
        str(stage3_adapter_dir)
        if decision["selected_stage"] == "stage3"
        else str(stage2_adapter_dir)
    )
    copy_adapter(selected_source_dir, str(output_adapter_dir))

    payload = {
        "decision": decision,
        "selected_adapter_dir": str(output_adapter_dir),
        "selected_source_dir": selected_source_dir,
        "stage2": {
            "all_family_proxy": stage2_all,
            "hard_triad_proxy": stage2_hard,
        },
        "stage3": {
            "all_family_proxy": stage3_all,
            "hard_triad_proxy": stage3_hard,
        },
    }
    write_json(output_json, payload)
    return payload


def run_trained_probe(
    *,
    config_path: str | Path,
    adapter_dir: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    config = read_yaml(config_path)
    return probe_adapter_submission_size(
        config=config,
        config_path=config_path,
        output_path=output_json,
        existing_adapter_dir=adapter_dir,
    )


def run_submission_validation(
    *,
    config_path: str | Path,
    adapter_dir: str | Path,
    output_json: str | Path,
    smoke_input: str | Path,
    labels: str | Path,
    splits: str | Path,
    package_output: str | Path,
    max_new_tokens: int,
) -> dict[str, Any]:
    return validate_submission(
        config_path=config_path,
        adapter_dir=adapter_dir,
        output_path=output_json,
        smoke_input=smoke_input,
        labels=labels,
        splits=splits,
        package_output=package_output,
        max_new_tokens=max_new_tokens,
    )


def build_acceptance_summary(
    *,
    selection_json: str | Path,
    probe_json: str | Path,
    validation_json: str | Path,
    summary_json: str | Path,
    submission_zip: str | Path,
    status: str,
    failure_stage: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    selection_payload = _load_json_if_exists(selection_json)
    probe_payload = _load_json_if_exists(probe_json)
    validation_payload = _load_json_if_exists(validation_json)

    selected_stage = _nested_get(selection_payload, "decision", "selected_stage")
    selected_source_dir = selection_payload.get("selected_source_dir") if selection_payload else None
    selected_adapter_dir = selection_payload.get("selected_adapter_dir") if selection_payload else None

    projected_zip_bytes = _nested_get(validation_payload, "submission_budget", "projected_submission_zip_bytes")
    if projected_zip_bytes is None and probe_payload is not None:
        projected_zip_bytes = probe_payload.get("formula_predicted_zip_bytes")

    probe_observed_suffixes = None
    if probe_payload is not None:
        probe_observed_suffixes = probe_payload.get("artifact_selected_suffixes")
        if probe_observed_suffixes is None:
            probe_observed_suffixes = probe_payload.get("selected_suffixes")

    report = {
        "status": status,
        "failure_stage": failure_stage,
        "error": error,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selected_stage": selected_stage,
        "selected_source_dir": selected_source_dir,
        "selected_adapter_dir": selected_adapter_dir,
        "paths": {
            "selection_json": str(selection_json),
            "probe_json": str(probe_json),
            "validation_json": str(validation_json),
            "summary_json": str(summary_json),
            "submission_zip": str(submission_zip),
        },
        "selection": {
            "decision": _nested_get(selection_payload, "decision"),
            "stage2_hard_triad_correct_rate": _nested_get(
                selection_payload, "stage2", "hard_triad_proxy", "competition_correct_rate"
            ),
            "stage2_all_family_correct_rate": _nested_get(
                selection_payload, "stage2", "all_family_proxy", "competition_correct_rate"
            ),
            "stage3_hard_triad_correct_rate": _nested_get(
                selection_payload, "stage3", "hard_triad_proxy", "competition_correct_rate"
            ),
            "stage3_all_family_correct_rate": _nested_get(
                selection_payload, "stage3", "all_family_proxy", "competition_correct_rate"
            ),
        },
        "probe": {
            "probe_mode": None if probe_payload is None else probe_payload.get("probe_mode"),
            "requested_rank": None if probe_payload is None else probe_payload.get("requested_rank"),
            "observed_rank": None if probe_payload is None else probe_payload.get("rank"),
            "requested_selected_suffixes": None
            if probe_payload is None
            else probe_payload.get("requested_selected_suffixes"),
            "observed_selected_suffixes": probe_observed_suffixes,
            "artifact_matches_requested_config": None
            if probe_payload is None
            else probe_payload.get("artifact_matches_requested_config"),
            "artifact_shape_matches_formula": None
            if probe_payload is None
            else probe_payload.get("artifact_shape_matches_formula"),
            "lora_b_likely_untrained": None
            if probe_payload is None
            else probe_payload.get("lora_b_likely_untrained"),
            "real_trained_adapter_archive_ratio": None
            if probe_payload is None
            else probe_payload.get("real_trained_adapter_archive_ratio"),
            "real_trained_adapter_weight_compression_ratio": None
            if probe_payload is None
            else probe_payload.get("real_trained_adapter_weight_compression_ratio"),
            "projected_zip_bytes": projected_zip_bytes,
            "physical_probe_zip_bytes": None if probe_payload is None else probe_payload.get("zip_size_bytes"),
        },
        "validation": {
            "submission_budget_status": _nested_get(validation_payload, "submission_budget", "status"),
            "projected_zip_bytes": projected_zip_bytes,
            "physical_packaged_zip_bytes": None
            if validation_payload is None
            else validation_payload.get("package_size_bytes"),
            "package_output": None if validation_payload is None else validation_payload.get("package_output"),
            "local_eval": None if validation_payload is None else validation_payload.get("local_eval"),
        },
    }
    write_json(summary_json, report)
    return report


def run_local_final_acceptance(
    *,
    stage2_adapter_dir: str | Path = DEFAULT_STAGE2_ADAPTER_DIR,
    stage3_adapter_dir: str | Path = DEFAULT_STAGE3_ADAPTER_DIR,
    stage2_hard_eval: str | Path = DEFAULT_STAGE2_HARD_EVAL,
    stage2_all_eval: str | Path = DEFAULT_STAGE2_ALL_EVAL,
    stage3_hard_eval: str | Path = DEFAULT_STAGE3_HARD_EVAL,
    stage3_all_eval: str | Path = DEFAULT_STAGE3_ALL_EVAL,
    output_adapter_dir: str | Path = DEFAULT_OUTPUT_ADAPTER_DIR,
    selection_json: str | Path = DEFAULT_SELECTION_JSON,
    probe_json: str | Path = DEFAULT_PROBE_JSON,
    validation_json: str | Path = DEFAULT_VALIDATION_JSON,
    summary_json: str | Path = DEFAULT_SUMMARY_JSON,
    submission_zip: str | Path = DEFAULT_SUBMISSION_ZIP,
    config_path: str | Path = DEFAULT_CONFIG,
    smoke_input: str | Path = DEFAULT_SMOKE_INPUT,
    labels: str | Path = DEFAULT_LABELS,
    splits: str | Path = DEFAULT_SPLITS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    current_stage = "input_validation"
    try:
        validate_acceptance_inputs(
            stage2_adapter_dir=stage2_adapter_dir,
            stage3_adapter_dir=stage3_adapter_dir,
            stage2_hard_eval=stage2_hard_eval,
            stage2_all_eval=stage2_all_eval,
            stage3_hard_eval=stage3_hard_eval,
            stage3_all_eval=stage3_all_eval,
        )

        current_stage = "select_final_adapter"
        run_final_adapter_selection(
            stage2_hard_eval=stage2_hard_eval,
            stage2_all_eval=stage2_all_eval,
            stage2_adapter_dir=stage2_adapter_dir,
            stage3_hard_eval=stage3_hard_eval,
            stage3_all_eval=stage3_all_eval,
            stage3_adapter_dir=stage3_adapter_dir,
            output_adapter_dir=output_adapter_dir,
            output_json=selection_json,
        )

        current_stage = "trained_probe"
        run_trained_probe(
            config_path=config_path,
            adapter_dir=output_adapter_dir,
            output_json=probe_json,
        )

        current_stage = "validate_submission"
        run_submission_validation(
            config_path=config_path,
            adapter_dir=output_adapter_dir,
            output_json=validation_json,
            smoke_input=smoke_input,
            labels=labels,
            splits=splits,
            package_output=submission_zip,
            max_new_tokens=max_new_tokens,
        )
    except BaseException as exc:
        build_acceptance_summary(
            selection_json=selection_json,
            probe_json=probe_json,
            validation_json=validation_json,
            summary_json=summary_json,
            submission_zip=submission_zip,
            status="fail",
            failure_stage=current_stage,
            error=str(exc),
        )
        raise

    return build_acceptance_summary(
        selection_json=selection_json,
        probe_json=probe_json,
        validation_json=validation_json,
        summary_json=summary_json,
        submission_zip=submission_zip,
        status="pass",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the local final acceptance chain for already-downloaded canonical stage artifacts."
    )
    parser.add_argument("--stage2-adapter-dir", default=str(DEFAULT_STAGE2_ADAPTER_DIR))
    parser.add_argument("--stage3-adapter-dir", default=str(DEFAULT_STAGE3_ADAPTER_DIR))
    parser.add_argument("--stage2-hard-eval", default=str(DEFAULT_STAGE2_HARD_EVAL))
    parser.add_argument("--stage2-all-eval", default=str(DEFAULT_STAGE2_ALL_EVAL))
    parser.add_argument("--stage3-hard-eval", default=str(DEFAULT_STAGE3_HARD_EVAL))
    parser.add_argument("--stage3-all-eval", default=str(DEFAULT_STAGE3_ALL_EVAL))
    parser.add_argument("--output-adapter-dir", default=str(DEFAULT_OUTPUT_ADAPTER_DIR))
    parser.add_argument("--selection-json", default=str(DEFAULT_SELECTION_JSON))
    parser.add_argument("--probe-json", default=str(DEFAULT_PROBE_JSON))
    parser.add_argument("--validation-json", default=str(DEFAULT_VALIDATION_JSON))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--submission-zip", default=str(DEFAULT_SUBMISSION_ZIP))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke-input", default=str(DEFAULT_SMOKE_INPUT))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    run_local_final_acceptance(
        stage2_adapter_dir=args.stage2_adapter_dir,
        stage3_adapter_dir=args.stage3_adapter_dir,
        stage2_hard_eval=args.stage2_hard_eval,
        stage2_all_eval=args.stage2_all_eval,
        stage3_hard_eval=args.stage3_hard_eval,
        stage3_all_eval=args.stage3_all_eval,
        output_adapter_dir=args.output_adapter_dir,
        selection_json=args.selection_json,
        probe_json=args.probe_json,
        validation_json=args.validation_json,
        summary_json=args.summary_json,
        submission_zip=args.submission_zip,
        config_path=args.config,
        smoke_input=args.smoke_input,
        labels=args.labels,
        splits=args.splits,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
