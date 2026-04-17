#!/usr/bin/env bash
set -euo pipefail

# Harness-aligned prompt format (chat_thinking) requires the Nemotron tokenizer.
# Populate via `python scripts/probe_chat_template.py` (tokenizer-only mode).
TOKENIZER_PATH="${TOKENIZER_PATH:-artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default}"

python scripts/prepare_data.py --config configs/data_official.yaml
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/stage1_format_align_train.jsonl \
  --selection-profile stage1 \
  --prompt-mode chat_thinking \
  --tokenizer-path "$TOKENIZER_PATH" \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty_all \
  --split-role train
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/stage1_format_align_valid.jsonl \
  --selection-profile stage1 \
  --prompt-mode chat_thinking \
  --tokenizer-path "$TOKENIZER_PATH" \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty_all \
  --split-role valid
python -m src.student.lora_train --config configs/train_stage1_format.yaml --force-train
