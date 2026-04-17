#!/usr/bin/env bash
set -euo pipefail

cd /workspace/nivida_remote

pkill -f "src.student.lora_train --config configs/train_lora_official_full_runpod.yaml" || true
rm -f /workspace/nivida_remote/train_official_full_runpod.log

nohup python -m src.student.lora_train --config configs/train_lora_official_full_runpod.yaml \
  > /workspace/nivida_remote/train_official_full_runpod.log 2>&1 < /dev/null &

echo $! > /workspace/nivida_remote/train_official_full_runpod.pid
cat /workspace/nivida_remote/train_official_full_runpod.pid
