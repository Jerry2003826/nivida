from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import warnings

from src.common.io import load_jsonl, read_json, read_yaml, write_json, write_jsonl
from src.competition.harness_prompt import (
    build_chat_thinking_prompt,
    wrap_as_thinking,
)
from src.competition.prompt_templates import (
    PROMPT_MODE_CHAT_THINKING,
    PROMPT_MODE_GENERIC,
    PROMPT_MODE_RAW_WITH_GUARD,
    build_competition_prompt,
)
from src.competition.schema import PuzzleExample
from src.teacher.chain_search import ChainSearchEngine
from src.teacher.error_taxonomy import classify_error
from src.teacher.family_tagger import apply_family_tags
from src.teacher.program_signature import annotate_example_from_candidates
from src.teacher.trace_compiler import compile_completion, compile_completion_body


TRACE_STYLES = {"answer_only", "short_trace", "token_trace"}
PROMPT_MODES = {
    PROMPT_MODE_RAW_WITH_GUARD,
    PROMPT_MODE_GENERIC,
    PROMPT_MODE_CHAT_THINKING,
}
HARD_TRIAD_FAMILIES = ("cipher", "bit", "equation")
BALANCED_FAMILY_ORDER = ("cipher", "bit", "equation", "numeral", "unit", "gravity")

REPAIR_ARTIFACT_REMEDIATION = (
    "Regenerate the artifact with a competition-correct evaluator "
    "(for example run_baseline or eval_competition_replica) so each record "
    "includes at least 'id' and 'competition_correct'."
)


class RepairArtifactSchemaError(ValueError):
    """Raised when a repair artifact does not satisfy the current baseline schema.

    The most common trigger is a legacy ``baseline_eval.json`` produced before the
    ``competition_correct`` field existed. Silently treating such artifacts as
    "all-failures" would contaminate stage3 training data, so this surfaces as a
    hard error instead.
    """


_REQUIRED_FAILURE_FIELDS: tuple[str, ...] = ("id", "competition_correct")


def _parse_input_paths(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _load_examples(input_paths: list[str]) -> list[PuzzleExample]:
    examples: list[PuzzleExample] = []
    for input_path in input_paths:
        examples.extend(PuzzleExample.from_dict(row) for row in load_jsonl(input_path))
    return examples


def _annotate_examples(
    examples: list[PuzzleExample],
    *,
    beam_width: int = 8,
    max_depth: int = 2,
    top_k: int = 2,
) -> list[PuzzleExample]:
    apply_family_tags(examples)
    engine = ChainSearchEngine(beam_width=beam_width, max_depth=max_depth)
    for example in examples:
        if example.metadata.program_signature and example.metadata.teacher_confidence is not None:
            continue
        candidates = engine.solve_example(example, top_k=top_k)
        annotate_example_from_candidates(example, candidates)
    return examples


def _metadata_value(example: PuzzleExample, key: str, default: Any = None) -> Any:
    if key in example.metadata.extras and example.metadata.extras[key] not in (None, ""):
        return example.metadata.extras[key]
    if hasattr(example.metadata, key):
        value = getattr(example.metadata, key)
        if value not in (None, ""):
            return value
    return default


def _source_prompt_type(example: PuzzleExample, prompt_mode: str) -> str:
    if prompt_mode == PROMPT_MODE_GENERIC:
        return "generic"
    if prompt_mode == PROMPT_MODE_CHAT_THINKING:
        return "chat_thinking"
    if example.raw_prompt.strip():
        return "raw"
    return "generic_fallback"


def _render_prompt_and_completion(
    example: PuzzleExample,
    *,
    prompt_mode: str,
    trace_style: str,
    tokenizer: Any | None,
) -> tuple[str, str]:
    """Return ``(prompt, completion)`` under the requested prompt_mode.

    ``chat_thinking`` requires a tokenizer with ``apply_chat_template`` so the
    training input is byte-identical to what the evaluator feeds to vLLM. Other
    modes keep the legacy text path.
    """
    if prompt_mode == PROMPT_MODE_CHAT_THINKING:
        if tokenizer is None:
            raise ValueError(
                "prompt_mode='chat_thinking' requires a tokenizer with "
                "apply_chat_template(). Pass tokenizer=... to build_sft_record "
                "(or to the profile builder) or switch to raw_with_guard."
            )
        prompt = build_chat_thinking_prompt(example, tokenizer)
        body = compile_completion_body(example, style=trace_style)
        completion = wrap_as_thinking(body, example.target_answer or "")
        return prompt, completion
    prompt = build_competition_prompt(example, mode=prompt_mode)
    completion = compile_completion(example, style=trace_style)
    return prompt, completion


def build_sft_record(
    example: PuzzleExample,
    *,
    stage: str,
    prompt_mode: str = PROMPT_MODE_RAW_WITH_GUARD,
    trace_style: str = "answer_only",
    default_official_family: str = "equation",
    include_metadata: bool = True,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")
    if trace_style not in TRACE_STYLES:
        raise ValueError(f"Unsupported trace style: {trace_style}")

    official_family = str(_metadata_value(example, "official_family", default_official_family))
    subtype = _metadata_value(example, "subtype")
    teacher_confidence = _metadata_value(example, "teacher_confidence")
    program_signature = _metadata_value(example, "program_signature")
    source_dataset = str(_metadata_value(example, "source_dataset", _metadata_value(example, "source", "official")))
    difficulty = _metadata_value(example, "difficulty")
    prompt, completion = _render_prompt_and_completion(
        example,
        prompt_mode=prompt_mode,
        trace_style=trace_style,
        tokenizer=tokenizer,
    )

    record = {
        "id": example.id,
        "prompt": prompt,
        "completion": completion,
        "stage": stage,
        "prompt_mode": prompt_mode,
        "trace_style": trace_style,
        "source_prompt_type": _source_prompt_type(example, prompt_mode),
        "target_answer": example.target_answer,
        "official_family": official_family,
        "subtype": subtype,
        "program_signature": program_signature,
        "teacher_confidence": teacher_confidence,
        "source_dataset": source_dataset,
        "difficulty": difficulty,
        "split": example.metadata.split,
    }
    if include_metadata:
        record["metadata"] = {
            "official_family": official_family,
            "subtype": subtype,
            "program_signature": program_signature,
            "teacher_confidence": teacher_confidence,
            "difficulty": difficulty,
            "source_dataset": source_dataset,
            "extras": dict(example.metadata.extras),
        }
    return record


def filter_examples_by_split(
    examples: list[PuzzleExample],
    *,
    split_file: str | Path,
    split_name: str,
    split_role: str,
) -> list[PuzzleExample]:
    payload = read_json(split_file)
    if split_name not in payload:
        raise KeyError(f"Split '{split_name}' not found in {split_file}")
    split_payload = payload[split_name]
    key = f"{split_role}_ids"
    if key not in split_payload:
        raise KeyError(f"Split role '{split_role}' not found in split '{split_name}'")
    keep_ids = set(split_payload[key])
    filtered: list[PuzzleExample] = []
    for example in examples:
        source_kind = _source_kind(example)
        if example.id in keep_ids:
            filtered.append(example)
            continue
        if split_role == "train" and source_kind in {"synth", "repair"}:
            filtered.append(example)
    return filtered


def export_split_subset(
    examples: list[PuzzleExample],
    *,
    split_file: str | Path,
    split_name: str,
    split_role: str,
) -> list[dict[str, Any]]:
    return [
        example.to_dict()
        for example in filter_examples_by_split(
            examples,
            split_file=split_file,
            split_name=split_name,
            split_role=split_role,
        )
    ]


def _source_kind(example: PuzzleExample) -> str:
    source = str(_metadata_value(example, "source", "official"))
    if source == "synthetic":
        return "synth"
    if source == "repair":
        return "repair"
    return "official"


def _filter_by_source(examples: list[PuzzleExample], source_filter: str | None) -> list[PuzzleExample]:
    if not source_filter:
        return examples
    wanted = source_filter.strip()
    return [example for example in examples if _source_kind(example) == wanted]


def build_stage1_sft(
    examples: list[PuzzleExample],
    *,
    prompt_mode: str = PROMPT_MODE_RAW_WITH_GUARD,
    include_metadata: bool = True,
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    records = []
    for example in examples:
        if _source_kind(example) != "official":
            continue
        if example.target_answer is None or not example.query:
            continue
        records.append(
            build_sft_record(
                example,
                stage="stage1",
                prompt_mode=prompt_mode,
                trace_style="answer_only",
                include_metadata=include_metadata,
                tokenizer=tokenizer,
            )
        )
    return records


def _select_official_stage2(example: PuzzleExample) -> bool:
    if _source_kind(example) != "official":
        return False
    if example.target_answer is None or not example.query:
        return False
    if float(_metadata_value(example, "teacher_confidence", 0.0) or 0.0) < 0.80:
        return False
    if not bool(example.metadata.extras.get("solver_verifiable")):
        return False
    if float(example.metadata.extras.get("support_coverage", 0.0) or 0.0) < 1.0:
        return False
    return bool(_metadata_value(example, "program_signature"))


def _select_synth_stage2(example: PuzzleExample, *, official_signatures: set[str]) -> bool:
    if _source_kind(example) != "synth":
        return False
    if example.target_answer is None or not example.query:
        return False
    if not bool(example.metadata.extras.get("solver_verifiable")):
        return False
    signature = _metadata_value(example, "program_signature")
    if not signature:
        return False
    return signature not in official_signatures


def _family_for_record(record: dict[str, Any]) -> str:
    family = record.get("official_family")
    return str(family) if family else "unknown"


def _signature_bucket_key(record: dict[str, Any]) -> str:
    signature = record.get("program_signature")
    if signature:
        return str(signature)
    return f"id:{record.get('id', 'unknown')}"


def _ordered_families(records: list[dict[str, Any]]) -> list[str]:
    seen = {_family_for_record(record) for record in records}
    ordered = [family for family in BALANCED_FAMILY_ORDER if family in seen]
    ordered.extend(sorted(seen - set(BALANCED_FAMILY_ORDER)))
    return ordered


def _truncate_by_signature_bucket(
    records: list[dict[str, Any]],
    *,
    max_per_signature_bucket: int,
) -> list[dict[str, Any]]:
    if max_per_signature_bucket <= 0:
        return list(records)
    signature_counts: dict[str, int] = {}
    kept: list[dict[str, Any]] = []
    for record in records:
        key = _signature_bucket_key(record)
        if signature_counts.get(key, 0) >= max_per_signature_bucket:
            continue
        signature_counts[key] = signature_counts.get(key, 0) + 1
        kept.append(record)
    return kept


def _balance_records_by_family(
    records: list[dict[str, Any]],
    *,
    hard_triad_repeat_factor: int,
    max_per_signature_bucket: int,
) -> list[dict[str, Any]]:
    if not records:
        return []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(_family_for_record(record), []).append(record)

    family_schedule: list[str] = []
    repeat_factor = max(1, int(hard_triad_repeat_factor))
    for family in _ordered_families(records):
        visits = repeat_factor if family in HARD_TRIAD_FAMILIES else 1
        family_schedule.extend([family] * visits)

    indices = {family: 0 for family in buckets}
    signature_counts: dict[str, int] = {}
    balanced: list[dict[str, Any]] = []
    while True:
        made_progress = False
        for family in family_schedule:
            bucket = buckets.get(family, [])
            while indices.get(family, 0) < len(bucket):
                record = bucket[indices[family]]
                indices[family] += 1
                signature_key = _signature_bucket_key(record)
                if max_per_signature_bucket > 0 and signature_counts.get(signature_key, 0) >= max_per_signature_bucket:
                    continue
                signature_counts[signature_key] = signature_counts.get(signature_key, 0) + 1
                balanced.append(record)
                made_progress = True
                break
        if not made_progress:
            break
    return balanced


def _record_source_dataset(record: dict[str, Any]) -> str:
    source_dataset = record.get("source_dataset")
    return str(source_dataset) if source_dataset else "unknown"


def summarise_selected_sft(records: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(_family_for_record(record) for record in records)
    subtype_counts = Counter(
        f"{_family_for_record(record)}:{record.get('subtype')}"
        for record in records
        if record.get("subtype")
    )
    source_dataset_counts = Counter(_record_source_dataset(record) for record in records)
    hard_triad_records = sum(_family_for_record(record) in HARD_TRIAD_FAMILIES for record in records)
    return {
        "num_records": len(records),
        "family_counts": dict(sorted(family_counts.items())),
        "subtype_counts": dict(sorted(subtype_counts.items())),
        "source_dataset_counts": dict(sorted(source_dataset_counts.items())),
        "hard_triad_ratio": 0.0 if not records else hard_triad_records / len(records),
    }


def _mark_stage3_record(
    record: dict[str, Any],
    *,
    source_dataset: str,
    repair_source: str,
) -> dict[str, Any]:
    record["source_dataset"] = source_dataset
    record["repair_source"] = repair_source
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        metadata["source_dataset"] = source_dataset
        extras = dict(metadata.get("extras", {}))
        extras["repair_source"] = repair_source
        metadata["extras"] = extras
    return record


def summarise_repair_sft(records: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(_family_for_record(record) for record in records)
    source_dataset_counts = Counter(_record_source_dataset(record) for record in records)
    repair_sources = Counter(str(record.get("repair_source", "repair")) for record in records)
    return {
        "num_records": len(records),
        "repair_count": repair_sources.get("repair", 0),
        "replay_count": repair_sources.get("replay", 0),
        "family_counts": dict(sorted(family_counts.items())),
        "source_dataset_counts": dict(sorted(source_dataset_counts.items())),
    }


def build_selected_sft(
    examples: list[PuzzleExample],
    *,
    prompt_mode: str = PROMPT_MODE_RAW_WITH_GUARD,
    trace_style: str = "token_trace",
    include_metadata: bool = True,
    beam_width: int = 8,
    max_depth: int = 2,
    top_k: int = 2,
    balance_by_family: bool = True,
    hard_triad_repeat_factor: int = 2,
    max_per_signature_bucket: int = 64,
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    annotated = _annotate_examples(
        examples,
        beam_width=beam_width,
        max_depth=max_depth,
        top_k=top_k,
    )
    selected_official = [example for example in annotated if _select_official_stage2(example)]
    official_signatures = {
        str(_metadata_value(example, "program_signature"))
        for example in selected_official
        if _metadata_value(example, "program_signature")
    }
    selected_synth = [example for example in annotated if _select_synth_stage2(example, official_signatures=official_signatures)]
    dedupe_keys: set[tuple[str, Any, Any]] = set()
    records: list[dict[str, Any]] = []
    for example in selected_official + selected_synth:
        key = (example.query, example.target_answer, _metadata_value(example, "program_signature"))
        if key in dedupe_keys:
            continue
        dedupe_keys.add(key)
        records.append(
            build_sft_record(
                example,
                stage="stage2",
                prompt_mode=prompt_mode,
                trace_style=trace_style,
                include_metadata=include_metadata,
                tokenizer=tokenizer,
            )
        )
    if balance_by_family:
        return _balance_records_by_family(
            records,
            hard_triad_repeat_factor=hard_triad_repeat_factor,
            max_per_signature_bucket=max_per_signature_bucket,
        )
    return _truncate_by_signature_bucket(records, max_per_signature_bucket=max_per_signature_bucket)


def _repair_bucket(example: PuzzleExample, failure_row: dict[str, Any]) -> str:
    enriched = {
        **failure_row,
        "official_family": _metadata_value(example, "official_family"),
        "subtype": _metadata_value(example, "subtype"),
        "program_signature": _metadata_value(example, "program_signature"),
        "teacher_confidence": _metadata_value(example, "teacher_confidence"),
    }
    return classify_error(enriched)


def _validate_repair_artifact(
    payload: Any,
    *,
    artifact_path: str | Path,
) -> list[dict[str, Any]]:
    """Validate a repair artifact (typically ``baseline_eval.json``) and return its records.

    Ensures that every downstream assumption in ``build_repair_set`` is met:

    - the payload is a JSON object
    - it carries a non-empty ``records`` (or legacy ``rows``) list
    - every record contains at least ``id`` and ``competition_correct``

    The previous implementation relied on ``row.get("competition_correct", False)``
    which silently treats any legacy record (no such field) as a failure, producing
    a stage3 dataset that spans the entire training set.
    """
    if not isinstance(payload, dict):
        raise RepairArtifactSchemaError(
            f"Repair artifact {artifact_path} is not a JSON object "
            f"(got {type(payload).__name__}). {REPAIR_ARTIFACT_REMEDIATION}"
        )

    records = payload.get("records")
    if records is None:
        records = payload.get("rows")
        if records is None:
            raise RepairArtifactSchemaError(
                f"Repair artifact {artifact_path} is missing both 'records' and 'rows'. "
                f"{REPAIR_ARTIFACT_REMEDIATION}"
            )
        warnings.warn(
            f"Repair artifact {artifact_path} uses the legacy 'rows' field; "
            "the canonical schema uses 'records'. Regenerate via run_baseline.",
            DeprecationWarning,
            stacklevel=3,
        )

    if not isinstance(records, list):
        raise RepairArtifactSchemaError(
            f"Repair artifact {artifact_path} has non-list records "
            f"(got {type(records).__name__}). {REPAIR_ARTIFACT_REMEDIATION}"
        )
    if not records:
        raise RepairArtifactSchemaError(
            f"Repair artifact {artifact_path} has an empty records list. "
            "The baseline run produced no evaluation rows. "
            f"{REPAIR_ARTIFACT_REMEDIATION}"
        )

    missing_counts: dict[str, int] = {}
    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            raise RepairArtifactSchemaError(
                f"Repair artifact {artifact_path} record at index {idx} is "
                f"{type(row).__name__}, expected dict."
            )
        for field in _REQUIRED_FAILURE_FIELDS:
            if field not in row:
                missing_counts[field] = missing_counts.get(field, 0) + 1

    if missing_counts:
        summary = ", ".join(
            f"'{field}' (missing in {count}/{len(records)} records)"
            for field, count in sorted(missing_counts.items())
        )
        raise RepairArtifactSchemaError(
            f"Repair artifact {artifact_path} is missing required fields: {summary}. "
            f"This looks like a legacy baseline schema. {REPAIR_ARTIFACT_REMEDIATION}"
        )

    return records


def build_repair_set(
    examples: list[PuzzleExample],
    *,
    repair_artifact: str | Path,
    prompt_mode: str = PROMPT_MODE_RAW_WITH_GUARD,
    trace_style: str = "short_trace",
    include_metadata: bool = True,
    beam_width: int = 8,
    max_depth: int = 2,
    top_k: int = 2,
    replay_input: str | Path | None = None,
    replay_ratio: float = 0.0,
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    payload = read_json(repair_artifact)
    repair_rows = _validate_repair_artifact(payload, artifact_path=repair_artifact)
    failure_index = {
        str(row["id"]): row
        for row in repair_rows
        if not row["competition_correct"]
    }
    annotated = _annotate_examples(
        examples,
        beam_width=beam_width,
        max_depth=max_depth,
        top_k=top_k,
    )
    repair_records: list[dict[str, Any]] = []
    for example in annotated:
        failure_row = failure_index.get(example.id)
        if failure_row is None:
            continue
        bucket = _repair_bucket(example, failure_row)
        record = build_sft_record(
            example,
            stage="stage3",
            prompt_mode=prompt_mode,
            trace_style=trace_style,
            include_metadata=include_metadata,
            tokenizer=tokenizer,
        )
        record["error_type"] = bucket
        record["target_signature"] = _metadata_value(example, "program_signature")
        record["predicted_signature"] = failure_row.get("predicted_signature")
        repair_records.append(_mark_stage3_record(record, source_dataset="repair", repair_source="repair"))

    replay_records: list[dict[str, Any]] = []
    if replay_input and replay_ratio > 0.0:
        replay_payload = read_json(replay_input)
        replay_records_payload = []
        if isinstance(replay_payload, dict):
            replay_records_payload = replay_payload.get("records", replay_payload.get("rows", []))
        if replay_records_payload:
            replay_rows = _validate_repair_artifact(replay_payload, artifact_path=replay_input)
        else:
            replay_rows = []
        success_ids = {
            str(row["id"])
            for row in replay_rows
            if bool(row.get("competition_correct", False))
        }
        replay_candidates: list[dict[str, Any]] = []
        for example in annotated:
            if example.id not in success_ids:
                continue
            record = build_sft_record(
                example,
                stage="stage3",
                prompt_mode=prompt_mode,
                trace_style=trace_style,
                include_metadata=include_metadata,
                tokenizer=tokenizer,
            )
            replay_candidates.append(_mark_stage3_record(record, source_dataset="replay", repair_source="replay"))
        replay_target = int(len(repair_records) * replay_ratio)
        if replay_target > 0:
            replay_candidates = _balance_records_by_family(
                replay_candidates,
                hard_triad_repeat_factor=1,
                max_per_signature_bucket=64,
            )
            replay_records = replay_candidates[:replay_target]

    return repair_records + replay_records


def _resolve_input_paths(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    if getattr(args, "inputs", None):
        return [str(path) for path in args.inputs]
    return _parse_input_paths(
        args.input
        or config.get("input")
        or config.get("training", {}).get("dataset_input_paths")
        or config.get("training", {}).get("dataset_path")
    )


def _resolve_cli_tokenizer(
    prompt_mode: str,
    tokenizer_path: str | None,
    config: dict[str, Any],
) -> Any | None:
    """Load a tokenizer only when prompt_mode requires one.

    This is a helper for the CLI entry point; library callers should pass
    ``tokenizer`` explicitly to the builder functions. ``transformers`` is
    imported lazily so that non-chat-thinking invocations don't depend on it.
    """
    if prompt_mode != PROMPT_MODE_CHAT_THINKING:
        return None
    path = tokenizer_path or config.get("tokenizer_path")
    if not path:
        raise ValueError(
            "prompt_mode='chat_thinking' requires --tokenizer-path (or "
            "'tokenizer_path' in the config). Populate the directory via "
            "scripts/probe_chat_template.py."
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for prompt_mode='chat_thinking'. "
            "Install with: pip install transformers"
        ) from exc
    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage1/stage2/stage3 SFT datasets.")
    parser.add_argument("--config")
    parser.add_argument("--input")
    parser.add_argument("--inputs", nargs="*")
    parser.add_argument("--output")
    parser.add_argument("--selection-profile", choices=["stage1", "stage2", "stage3"], default="stage2")
    parser.add_argument("--prompt-mode", choices=sorted(PROMPT_MODES), default=PROMPT_MODE_RAW_WITH_GUARD)
    parser.add_argument("--completion-style", choices=sorted(TRACE_STYLES), default="token_trace")
    parser.add_argument("--source-filter", choices=["official", "synth", "repair"])
    parser.add_argument("--split-file")
    parser.add_argument("--split-name")
    parser.add_argument("--split-role", choices=["train", "valid"])
    parser.add_argument("--repair-artifact")
    parser.add_argument("--replay-input")
    parser.add_argument("--replay-ratio", type=float, default=0.0)
    parser.add_argument("--default-official-family", default="equation")
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--balance-by-family", dest="balance_by_family", action="store_true")
    parser.add_argument("--no-balance-by-family", dest="balance_by_family", action="store_false")
    parser.set_defaults(balance_by_family=True)
    parser.add_argument("--hard-triad-repeat-factor", type=int, default=2)
    parser.add_argument("--max-per-signature-bucket", type=int, default=64)
    parser.add_argument("--report-output")
    parser.add_argument(
        "--tokenizer-path",
        help=(
            "Local directory for the tokenizer used by prompt_mode=chat_thinking. "
            "Populate via scripts/probe_chat_template.py (tokenizer-only mode)."
        ),
    )
    args = parser.parse_args()

    config = read_yaml(args.config) if args.config else {}
    input_paths = _resolve_input_paths(args, config)
    if not input_paths:
        raise ValueError("At least one input JSONL path is required.")

    examples = _load_examples(input_paths)
    if args.split_file and args.split_name and args.split_role:
        examples = filter_examples_by_split(
            examples,
            split_file=args.split_file,
            split_name=args.split_name,
            split_role=args.split_role,
        )
    examples = _filter_by_source(examples, args.source_filter)

    prompt_mode = args.prompt_mode
    trace_style = args.completion_style
    include_metadata = args.include_metadata
    tokenizer = _resolve_cli_tokenizer(prompt_mode, args.tokenizer_path, config)

    if args.selection_profile == "stage1":
        dataset = build_stage1_sft(
            examples,
            prompt_mode=prompt_mode,
            include_metadata=include_metadata,
            tokenizer=tokenizer,
        )
        report = {"num_records": len(dataset)}
        default_output = "data/processed/stage1_format_align.jsonl"
    elif args.selection_profile == "stage2":
        dataset = build_selected_sft(
            examples,
            prompt_mode=prompt_mode,
            trace_style=trace_style,
            include_metadata=include_metadata,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            top_k=args.top_k,
            balance_by_family=args.balance_by_family,
            hard_triad_repeat_factor=args.hard_triad_repeat_factor,
            max_per_signature_bucket=args.max_per_signature_bucket,
            tokenizer=tokenizer,
        )
        report = summarise_selected_sft(dataset)
        default_output = "data/processed/stage2_distill.jsonl"
    else:
        if not args.repair_artifact:
            raise ValueError("--repair-artifact is required for selection-profile=stage3")
        dataset = build_repair_set(
            examples,
            repair_artifact=args.repair_artifact,
            prompt_mode=prompt_mode,
            trace_style=trace_style,
            include_metadata=include_metadata,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            top_k=args.top_k,
            replay_input=args.replay_input,
            replay_ratio=args.replay_ratio,
            tokenizer=tokenizer,
        )
        report = summarise_repair_sft(dataset)
        default_output = "data/processed/stage3_repair.jsonl"

    output_path = args.output or config.get("output") or default_output
    write_jsonl(output_path, dataset)
    if args.report_output:
        write_json(args.report_output, report)


if __name__ == "__main__":
    main()
