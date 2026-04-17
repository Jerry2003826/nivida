#!/usr/bin/env bash
set -euo pipefail

python -m src.experiments.run_hardcase_repair --input data/processed/baseline_eval.json --output data/processed/hard_cases.json
python -m src.student.lora_train --config configs/train_lora.yaml --dry-run
