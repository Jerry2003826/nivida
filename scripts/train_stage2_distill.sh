#!/usr/bin/env bash
set -euo pipefail

# Harness-aligned prompt format (chat_thinking) requires the Nemotron tokenizer.
# Populate via `python scripts/probe_chat_template.py` (tokenizer-only mode).
TOKENIZER_PATH="${TOKENIZER_PATH:-artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default}"
STAGE1_ADAPTER_DIR="${STAGE1_ADAPTER_DIR:-artifacts/adapter_stage1_format}"
STAGE2_CONFIG_TEMPLATE="${STAGE2_CONFIG_TEMPLATE:-configs/train_stage2_selected_trace.yaml}"
STAGE2_REPORT="${STAGE2_REPORT:-data/processed/stage2_distill_report.json}"
STAGE2_VALID_REPORT="${STAGE2_VALID_REPORT:-data/processed/stage2_distill_valid_report.json}"

if [[ ! -d "$STAGE1_ADAPTER_DIR" ]]; then
  echo "Missing stage1 adapter: $STAGE1_ADAPTER_DIR" >&2
  exit 1
fi

python scripts/prepare_data.py --config configs/data_official.yaml
python -m src.teacher.synth_generator --config configs/synth_hard_triads.yaml
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl,data/synthetic/synth_hard_triads.jsonl \
  --output data/processed/stage2_distill_train.jsonl \
  --selection-profile stage2 \
  --prompt-mode chat_thinking \
  --tokenizer-path "$TOKENIZER_PATH" \
  --completion-style token_trace \
  --beam-width 10 \
  --max-depth 3 \
  --top-k 3 \
  --balance-by-family \
  --hard-triad-repeat-factor 2 \
  --max-per-signature-bucket 64 \
  --report-output "$STAGE2_REPORT" \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad_rule_novelty \
  --split-role train
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl,data/synthetic/synth_hard_triads.jsonl \
  --output data/processed/stage2_distill_valid.jsonl \
  --selection-profile stage2 \
  --prompt-mode chat_thinking \
  --tokenizer-path "$TOKENIZER_PATH" \
  --completion-style token_trace \
  --beam-width 10 \
  --max-depth 3 \
  --top-k 3 \
  --balance-by-family \
  --hard-triad-repeat-factor 2 \
  --max-per-signature-bucket 64 \
  --report-output "$STAGE2_VALID_REPORT" \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad_rule_novelty \
  --split-role valid

STAGE2_RUNTIME_CONFIG="$(mktemp)"
trap 'rm -f "$STAGE2_RUNTIME_CONFIG"' EXIT
python - "$STAGE2_CONFIG_TEMPLATE" "$STAGE1_ADAPTER_DIR" "$STAGE2_RUNTIME_CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml

template_path, init_adapter_dir, output_path = sys.argv[1:4]
payload = yaml.safe_load(Path(template_path).read_text(encoding="utf-8")) or {}
payload.setdefault("training", {})["init_adapter_dir"] = init_adapter_dir
Path(output_path).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY

python -m src.student.lora_train --config "$STAGE2_RUNTIME_CONFIG" --force-train
