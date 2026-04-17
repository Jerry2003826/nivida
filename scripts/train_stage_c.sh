#!/usr/bin/env bash
set -euo pipefail

python -m src.student.sft_dataset_builder --config configs/train_lora.yaml --stage stage_c --output data/processed/stage_c_train.jsonl
python -m src.student.lora_train --config configs/train_lora.yaml --dry-run
