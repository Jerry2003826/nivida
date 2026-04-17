#!/usr/bin/env bash
set -euo pipefail

python -m src.student.sft_dataset_builder --config configs/train_lora.yaml --stage format_only --output data/processed/format_only_train.jsonl
python -m src.student.lora_train --config configs/train_lora.yaml --dry-run
