#!/usr/bin/env bash
set -euo pipefail

# Harness-aligned prompt format (chat_thinking) requires the Nemotron tokenizer.
# Populate via `python scripts/probe_chat_template.py` (tokenizer-only mode).
TOKENIZER_PATH="${TOKENIZER_PATH:-artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default}"

python scripts/prepare_data.py --config configs/data_official.yaml

# split_builder constructs rule_novelty_all and hard_triad_rule_novelty
# independently (different seeds, different subsets). Stage1 must exclude
# hard_triad_rule_novelty/valid here so later stage2/stage3 hard-triad
# validation is a genuine holdout across the whole staged pipeline (stage1
# -> stage2 init_adapter_dir=stage1 -> stage3 init_adapter_dir=stage2).
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/stage1_format_align_train.jsonl \
  --selection-profile stage1 \
  --prompt-mode chat_thinking \
  --tokenizer-path "$TOKENIZER_PATH" \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty_all \
  --split-role train \
  --exclude-split-file data/splits/official/splits.json \
  --exclude-split-name hard_triad_rule_novelty \
  --exclude-split-role valid
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
