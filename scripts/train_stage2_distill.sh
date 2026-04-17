#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_data.py --config configs/data_official.yaml
python -m src.teacher.synth_generator --config configs/synth.yaml
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl,data/synthetic/stage2_synth.jsonl \
  --output data/processed/stage2_distill_train.jsonl \
  --selection-profile stage2 \
  --completion-style token_trace \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad \
  --split-role train
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl,data/synthetic/stage2_synth.jsonl \
  --output data/processed/stage2_distill_valid.jsonl \
  --selection-profile stage2 \
  --completion-style token_trace \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad \
  --split-role valid
python -m src.student.lora_train --config configs/train_lora.yaml
