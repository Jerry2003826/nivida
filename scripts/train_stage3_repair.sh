#!/usr/bin/env bash
set -euo pipefail

# Harness-aligned prompt format (chat_thinking) requires the Nemotron tokenizer.
# Populate via `python scripts/probe_chat_template.py` (tokenizer-only mode).

TOKENIZER_PATH="${TOKENIZER_PATH:-artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default}"

STAGE2_ADAPTER_DIR="${STAGE2_ADAPTER_DIR:-artifacts/adapter_stage2_selected_trace}"
STAGE3_CONFIG_TEMPLATE="${STAGE3_CONFIG_TEMPLATE:-configs/train_stage3_repair.yaml}"

REPAIR_TRAIN_SUBSET="${REPAIR_TRAIN_SUBSET:-data/processed/stage3_repair_subset_train.jsonl}"
REPLAY_TRAIN_SUBSET="${REPLAY_TRAIN_SUBSET:-data/processed/stage3_replay_pool_train.jsonl}"
VALID_SUBSET="${VALID_SUBSET:-data/processed/stage3_repair_subset_valid.jsonl}"

TRAIN_REPLAY_PREDICTIONS="${TRAIN_REPLAY_PREDICTIONS:-data/processed/stage2_replay_pool_predictions_train.jsonl}"
VALID_PREDICTIONS="${VALID_PREDICTIONS:-data/processed/stage2_valid_predictions.jsonl}"

TRAIN_REPLAY_EVAL="${TRAIN_REPLAY_EVAL:-data/processed/stage2_model_eval_replay_pool_train.json}"
VALID_EVAL="${VALID_EVAL:-data/processed/stage2_model_eval_valid.json}"

TRAIN_FAILURES="${TRAIN_FAILURES:-data/processed/stage2_model_failures_train.json}"
TRAIN_SUCCESSES="${TRAIN_SUCCESSES:-data/processed/stage2_model_successes_all_train.json}"
VALID_FAILURES="${VALID_FAILURES:-data/processed/stage2_model_failures_valid.json}"

STAGE3_TRAIN_REPORT="${STAGE3_TRAIN_REPORT:-data/processed/stage3_repair_train_report.json}"
STAGE3_VALID_REPORT="${STAGE3_VALID_REPORT:-data/processed/stage3_repair_valid_report.json}"

REPLAY_RATIO="${REPLAY_RATIO:-0.25}"

if [[ ! -d "$STAGE2_ADAPTER_DIR" ]]; then
  echo "Missing stage2 adapter: $STAGE2_ADAPTER_DIR" >&2
  exit 1
fi

python scripts/prepare_data.py --config configs/data_official.yaml

# 1) Hard-triad train subset used to restrict repair failures.
python scripts/export_split_subset.py \
  --input data/processed/official_train_tagged.jsonl \
  --output "$REPAIR_TRAIN_SUBSET" \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad_rule_novelty \
  --split-role train

# 2) All-family train subset used as the stage2 replay pool.
python scripts/export_split_subset.py \
  --input data/processed/official_train_tagged.jsonl \
  --output "$REPLAY_TRAIN_SUBSET" \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty_all \
  --split-role train

# 3) Hard-triad valid subset for validation-only repair set.
python scripts/export_split_subset.py \
  --input data/processed/official_train_tagged.jsonl \
  --output "$VALID_SUBSET" \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad_rule_novelty \
  --split-role valid

# 4) Run stage2 adapter on the all-family replay pool (train) and hard-triad valid subset.
python -m src.student.inference \
  --config configs/train_stage2_selected_trace.yaml \
  --input "$REPLAY_TRAIN_SUBSET" \
  --adapter-dir "$STAGE2_ADAPTER_DIR" \
  --output "$TRAIN_REPLAY_PREDICTIONS" \
  --max-new-tokens 2048

python -m src.student.inference \
  --config configs/train_stage2_selected_trace.yaml \
  --input "$VALID_SUBSET" \
  --adapter-dir "$STAGE2_ADAPTER_DIR" \
  --output "$VALID_PREDICTIONS" \
  --max-new-tokens 2048

# 5) Evaluate both prediction files against the competition replica metric.
python -m src.experiments.eval_competition_replica \
  --predictions "$TRAIN_REPLAY_PREDICTIONS" \
  --labels "$REPLAY_TRAIN_SUBSET" \
  --output "$TRAIN_REPLAY_EVAL"

python -m src.experiments.eval_competition_replica \
  --predictions "$VALID_PREDICTIONS" \
  --labels "$VALID_SUBSET" \
  --output "$VALID_EVAL"

# 6) Split eval artifacts:
#    - repair failures: restricted to hard-triad train ids
#    - replay successes: kept across the full all-family train pool
#    - valid failures: stay within the hard-triad valid subset
python scripts/split_eval_artifact.py \
  --input "$TRAIN_REPLAY_EVAL" \
  --restrict-ids "$REPAIR_TRAIN_SUBSET" \
  --output-failures "$TRAIN_FAILURES"

python scripts/split_eval_artifact.py \
  --input "$TRAIN_REPLAY_EVAL" \
  --output-successes "$TRAIN_SUCCESSES"

python scripts/split_eval_artifact.py \
  --input "$VALID_EVAL" \
  --output-failures "$VALID_FAILURES"

# 7) Build stage3 train set over the full official pool so replay can reach
#    easy-triad anchors. build_repair_set() filters records per artifact ids.
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
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
  --replay-ratio "$REPLAY_RATIO" \
  --report-output "$STAGE3_TRAIN_REPORT"

# 8) Build stage3 valid set from the hard-triad valid subset only; no replay.
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

# 9) Continue training from the stage2 adapter (injected into a runtime config).
STAGE3_RUNTIME_CONFIG="$(mktemp)"
trap 'rm -f "$STAGE3_RUNTIME_CONFIG"' EXIT

python - "$STAGE3_CONFIG_TEMPLATE" "$STAGE2_ADAPTER_DIR" "$STAGE3_RUNTIME_CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml

template_path, init_adapter_dir, output_path = sys.argv[1:4]
payload = yaml.safe_load(Path(template_path).read_text(encoding="utf-8")) or {}
payload.setdefault("training", {})["init_adapter_dir"] = init_adapter_dir
Path(output_path).write_text(
    yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY

python -m src.student.lora_train --config "$STAGE3_RUNTIME_CONFIG" --force-train
