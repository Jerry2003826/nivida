#!/usr/bin/env bash
set -euo pipefail

# Harness-aligned prompt format (chat_thinking) requires the Nemotron tokenizer.
# Populate via `python scripts/probe_chat_template.py` (tokenizer-only mode).

TOKENIZER_PATH="${TOKENIZER_PATH:-artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default}"

STAGE2_ADAPTER_DIR="${STAGE2_ADAPTER_DIR:-artifacts/adapter_stage2_selected_trace}"
STAGE3_CONFIG_TEMPLATE="${STAGE3_CONFIG_TEMPLATE:-configs/train_stage3_repair.yaml}"

FULL_TRAIN_INPUT="${FULL_TRAIN_INPUT:-data/processed/official_train_tagged.jsonl}"

REPAIR_TRAIN_SUBSET="${REPAIR_TRAIN_SUBSET:-data/processed/stage3_repair_subset_train.jsonl}"
REPLAY_TRAIN_SUBSET="${REPLAY_TRAIN_SUBSET:-data/processed/stage3_replay_pool_train.jsonl}"
VALID_SUBSET="${VALID_SUBSET:-data/processed/stage3_repair_subset_valid.jsonl}"

FULL_TRAIN_PREDICTIONS="${FULL_TRAIN_PREDICTIONS:-data/processed/stage2_full_train_predictions.jsonl}"
VALID_PREDICTIONS="${VALID_PREDICTIONS:-data/processed/stage2_valid_predictions.jsonl}"

FULL_TRAIN_EVAL="${FULL_TRAIN_EVAL:-data/processed/stage2_full_train_eval.json}"
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

# Hard-triad train subset -> used later to restrict repair failures.
python scripts/export_split_subset.py \
  --input "$FULL_TRAIN_INPUT" \
  --output "$REPAIR_TRAIN_SUBSET" \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad_rule_novelty \
  --split-role train

# All-family replay pool:
#   rule_novelty_all/train minus hard_triad_rule_novelty/valid
# This keeps easy-triad anchors available for replay without leaking any
# hard-triad valid id into stage3 training.
python scripts/export_split_subset.py \
  --input "$FULL_TRAIN_INPUT" \
  --output "$REPLAY_TRAIN_SUBSET" \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty_all \
  --split-role train \
  --exclude-split-file data/splits/official/splits.json \
  --exclude-split-name hard_triad_rule_novelty \
  --exclude-split-role valid

# Hard-triad valid subset.
python scripts/export_split_subset.py \
  --input "$FULL_TRAIN_INPUT" \
  --output "$VALID_SUBSET" \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad_rule_novelty \
  --split-role valid

# Run stage2 adapter on the FULL official train pool, not only the replay pool.
# The two splits are not nested in split_builder, so a hard-triad train id can
# be absent from rule_novelty_all/train; running over the full pool guarantees
# no hard-triad failure is silently dropped from the stage3 repair artifact.
python -m src.student.inference \
  --config configs/train_stage2_selected_trace.yaml \
  --input "$FULL_TRAIN_INPUT" \
  --adapter-dir "$STAGE2_ADAPTER_DIR" \
  --output "$FULL_TRAIN_PREDICTIONS" \
  --max-new-tokens 2048

python -m src.student.inference \
  --config configs/train_stage2_selected_trace.yaml \
  --input "$VALID_SUBSET" \
  --adapter-dir "$STAGE2_ADAPTER_DIR" \
  --output "$VALID_PREDICTIONS" \
  --max-new-tokens 2048

# Strict coverage gate: any missing / unexpected / duplicate prediction id will
# raise and stop the pipeline instead of silently polluting the repair set.
python -m src.experiments.eval_competition_replica \
  --predictions "$FULL_TRAIN_PREDICTIONS" \
  --labels "$FULL_TRAIN_INPUT" \
  --output "$FULL_TRAIN_EVAL" \
  --require-complete-coverage

python -m src.experiments.eval_competition_replica \
  --predictions "$VALID_PREDICTIONS" \
  --labels "$VALID_SUBSET" \
  --output "$VALID_EVAL" \
  --require-complete-coverage

# Repair failures: only hard-triad train ids.
python scripts/split_eval_artifact.py \
  --input "$FULL_TRAIN_EVAL" \
  --restrict-ids "$REPAIR_TRAIN_SUBSET" \
  --output-failures "$TRAIN_FAILURES"

# Replay successes: all-family train ids minus hard-triad valid (already
# leak-free by construction of REPLAY_TRAIN_SUBSET).
python scripts/split_eval_artifact.py \
  --input "$FULL_TRAIN_EVAL" \
  --restrict-ids "$REPLAY_TRAIN_SUBSET" \
  --output-successes "$TRAIN_SUCCESSES"

# Valid failures: hard-triad valid only.
python scripts/split_eval_artifact.py \
  --input "$VALID_EVAL" \
  --restrict-ids "$VALID_SUBSET" \
  --output-failures "$VALID_FAILURES"

python -m src.student.sft_dataset_builder \
  --input "$FULL_TRAIN_INPUT" \
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
Path(output_path).write_text(
    yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY

python -m src.student.lora_train --config "$STAGE3_RUNTIME_CONFIG" --force-train
