#!/usr/bin/env bash
set -euo pipefail

python -m src.teacher.synth_generator --config configs/synth.yaml
python -m src.student.lora_train --config configs/train_lora.yaml --dry-run
