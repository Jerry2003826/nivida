#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_data.py --config configs/data_official.yaml
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/official_stage_c_train.jsonl \
  --stage stage_c \
  --split-file data/splits/official/splits.json \
  --split-name iid \
  --split-role train
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/official_stage_c_valid.jsonl \
  --stage stage_c \
  --split-file data/splits/official/splits.json \
  --split-name iid \
  --split-role valid
python -m src.experiments.run_baseline \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/baseline_eval.json
python -m src.experiments.run_teacher_benchmark \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/teacher_benchmark.json \
  --max-per-family 30
python -m src.experiments.run_hardcase_repair \
  --input data/processed/baseline_eval.json \
  --output data/processed/hard_cases.json \
  --max-items 64
python -m src.teacher.synth_generator --config configs/synth.yaml
python -m src.experiments.build_global_rule_graph \
  --input data/processed/teacher_benchmark.json \
  --output data/processed/global_rule_graph.json
python -m src.student.lora_train --config configs/train_lora_smoke.yaml
