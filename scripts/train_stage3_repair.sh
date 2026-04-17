#!/usr/bin/env bash
set -euo pipefail

python -m src.experiments.run_baseline \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/baseline_eval.json
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/stage3_repair_train.jsonl \
  --selection-profile stage3 \
  --completion-style short_trace \
  --repair-artifact data/processed/baseline_eval.json \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad \
  --split-role train
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/stage3_repair_valid.jsonl \
  --selection-profile stage3 \
  --completion-style short_trace \
  --repair-artifact data/processed/baseline_eval.json \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad \
  --split-role valid
python -m src.student.lora_train --config configs/train_lora_stage3_repair.yaml
