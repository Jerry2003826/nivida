#!/usr/bin/env bash
set -euo pipefail

# Harness-aligned prompt format (chat_thinking) requires the Nemotron tokenizer.
# Populate via `python scripts/probe_chat_template.py` (tokenizer-only mode).
TOKENIZER_PATH="${TOKENIZER_PATH:-artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default}"
STAGE2_ADAPTER_DIR="${STAGE2_ADAPTER_DIR:-artifacts/adapter_stage2_selected_trace}"
STAGE3_CONFIG_TEMPLATE="${STAGE3_CONFIG_TEMPLATE:-configs/train_stage3_repair.yaml}"
TRAIN_SUBSET="${TRAIN_SUBSET:-data/processed/stage3_repair_subset_train.jsonl}"
VALID_SUBSET="${VALID_SUBSET:-data/processed/stage3_repair_subset_valid.jsonl}"
TRAIN_PREDICTIONS="${TRAIN_PREDICTIONS:-data/processed/stage2_train_predictions.jsonl}"
VALID_PREDICTIONS="${VALID_PREDICTIONS:-data/processed/stage2_valid_predictions.jsonl}"
TRAIN_EVAL="${TRAIN_EVAL:-data/processed/stage2_model_eval_train.json}"
VALID_EVAL="${VALID_EVAL:-data/processed/stage2_model_eval_valid.json}"
TRAIN_FAILURES="${TRAIN_FAILURES:-data/processed/stage2_model_failures_train.json}"
VALID_FAILURES="${VALID_FAILURES:-data/processed/stage2_model_failures_valid.json}"
TRAIN_SUCCESSES="${TRAIN_SUCCESSES:-data/processed/stage2_model_successes_train.json}"
VALID_SUCCESSES="${VALID_SUCCESSES:-data/processed/stage2_model_successes_valid.json}"
STAGE3_TRAIN_REPORT="${STAGE3_TRAIN_REPORT:-data/processed/stage3_repair_train_report.json}"
STAGE3_VALID_REPORT="${STAGE3_VALID_REPORT:-data/processed/stage3_repair_valid_report.json}"

if [[ ! -d "$STAGE2_ADAPTER_DIR" ]]; then
  echo "Missing stage2 adapter: $STAGE2_ADAPTER_DIR" >&2
  exit 1
fi

python scripts/prepare_data.py --config configs/data_official.yaml
python scripts/export_split_subset.py \
  --input data/processed/official_train_tagged.jsonl \
  --output "$TRAIN_SUBSET" \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad_rule_novelty \
  --split-role train
python scripts/export_split_subset.py \
  --input data/processed/official_train_tagged.jsonl \
  --output "$VALID_SUBSET" \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad_rule_novelty \
  --split-role valid

python -m src.student.inference \
  --config configs/train_stage2_selected_trace.yaml \
  --input "$TRAIN_SUBSET" \
  --adapter-dir "$STAGE2_ADAPTER_DIR" \
  --output "$TRAIN_PREDICTIONS" \
  --max-new-tokens 2048
python -m src.student.inference \
  --config configs/train_stage2_selected_trace.yaml \
  --input "$VALID_SUBSET" \
  --adapter-dir "$STAGE2_ADAPTER_DIR" \
  --output "$VALID_PREDICTIONS" \
  --max-new-tokens 2048

python -m src.experiments.eval_competition_replica \
  --predictions "$TRAIN_PREDICTIONS" \
  --labels "$TRAIN_SUBSET" \
  --output "$TRAIN_EVAL"
python -m src.experiments.eval_competition_replica \
  --predictions "$VALID_PREDICTIONS" \
  --labels "$VALID_SUBSET" \
  --output "$VALID_EVAL"

python - "$TRAIN_EVAL" "$TRAIN_FAILURES" "$TRAIN_SUCCESSES" <<'PY'
from pathlib import Path
import json
import sys

source_path, failure_path, success_path = sys.argv[1:4]
payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
records = list(payload.get("records", []))
failures = {**payload, "records": [row for row in records if not row.get("competition_correct", False)]}
successes = {**payload, "records": [row for row in records if row.get("competition_correct", False)]}
Path(failure_path).write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
Path(success_path).write_text(json.dumps(successes, ensure_ascii=False, indent=2), encoding="utf-8")
PY
python - "$VALID_EVAL" "$VALID_FAILURES" "$VALID_SUCCESSES" <<'PY'
from pathlib import Path
import json
import sys

source_path, failure_path, success_path = sys.argv[1:4]
payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
records = list(payload.get("records", []))
failures = {**payload, "records": [row for row in records if not row.get("competition_correct", False)]}
successes = {**payload, "records": [row for row in records if row.get("competition_correct", False)]}
Path(failure_path).write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
Path(success_path).write_text(json.dumps(successes, ensure_ascii=False, indent=2), encoding="utf-8")
PY

python -m src.student.sft_dataset_builder \
  --input "$TRAIN_SUBSET" \
  --output data/processed/stage3_repair_train.jsonl \
  --selection-profile stage3 \
  --prompt-mode chat_thinking \
  --tokenizer-path "$TOKENIZER_PATH" \
  --completion-style short_trace \
  --beam-width 8 \
  --max-depth 2 \
  --top-k 2 \
  --repair-artifact "$TRAIN_FAILURES" \
  --replay-input "$TRAIN_SUCCESSES" \
  --replay-ratio 0.25 \
  --report-output "$STAGE3_TRAIN_REPORT"
python -m src.student.sft_dataset_builder \
  --input "$VALID_SUBSET" \
  --output data/processed/stage3_repair_valid.jsonl \
  --selection-profile stage3 \
  --prompt-mode chat_thinking \
  --tokenizer-path "$TOKENIZER_PATH" \
  --completion-style short_trace \
  --beam-width 8 \
  --max-depth 2 \
  --top-k 2 \
  --repair-artifact "$VALID_FAILURES" \
  --report-output "$STAGE3_VALID_REPORT"

STAGE3_RUNTIME_CONFIG="$(mktemp)"
trap 'rm -f "$STAGE3_RUNTIME_CONFIG"' EXIT
python - "$STAGE3_CONFIG_TEMPLATE" "$STAGE2_ADAPTER_DIR" "$STAGE3_RUNTIME_CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml

template_path, init_adapter_dir, output_path = sys.argv[1:4]
payload = yaml.safe_load(Path(template_path).read_text(encoding="utf-8")) or {}
payload.setdefault("training", {})["init_adapter_dir"] = init_adapter_dir
Path(output_path).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY

python -m src.student.lora_train --config "$STAGE3_RUNTIME_CONFIG" --force-train
