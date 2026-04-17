#!/usr/bin/env bash
set -euo pipefail

TOKENIZER_PATH="${TOKENIZER_PATH:-artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default}"
SMOKE_TRAIN_LIMIT="${SMOKE_TRAIN_LIMIT:-64}"
SMOKE_VALID_LIMIT="${SMOKE_VALID_LIMIT:-32}"

if [[ ! -f data/processed/official_train_tagged.jsonl || ! -f data/splits/official/splits.json ]]; then
  python scripts/prepare_data.py --config configs/data_official.yaml
fi

mkdir -p data/processed/smoke artifacts/smoke

python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/smoke/stage1_format_align_train.jsonl \
  --selection-profile stage1 \
  --prompt-mode chat_thinking \
  --tokenizer-path "$TOKENIZER_PATH" \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty_all \
  --split-role train

python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/smoke/stage1_format_align_valid.jsonl \
  --selection-profile stage1 \
  --prompt-mode chat_thinking \
  --tokenizer-path "$TOKENIZER_PATH" \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty_all \
  --split-role valid

python scripts/subsample_jsonl.py \
  --input data/processed/smoke/stage1_format_align_train.jsonl \
  --output data/processed/smoke/stage1_format_align_train.jsonl \
  --limit "$SMOKE_TRAIN_LIMIT"

python scripts/subsample_jsonl.py \
  --input data/processed/smoke/stage1_format_align_valid.jsonl \
  --output data/processed/smoke/stage1_format_align_valid.jsonl \
  --limit "$SMOKE_VALID_LIMIT"

python -m src.student.lora_train --config configs/smoke/train_stage1_smoke.yaml
