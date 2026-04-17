#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_data.py --config configs/data_official.yaml
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/stage1_format_align_train.jsonl \
  --selection-profile stage1 \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty \
  --split-role train
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/stage1_format_align_valid.jsonl \
  --selection-profile stage1 \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty \
  --split-role valid
python -m src.student.lora_train --config configs/train_lora_official.yaml
