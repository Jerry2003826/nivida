#!/usr/bin/env bash
set -euo pipefail

# Build the answer-focused continuation datasets used by:
#   configs/train_stage2_thin_answer_only.yaml
#   configs/train_stage2_thin_short_trace.yaml
#
# This script prepares data only. It intentionally does not start training.

cd "${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

TOKENIZER_PATH="${TOKENIZER_PATH:-artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default}"
TRAIN_INPUT="${TRAIN_INPUT:-data/processed/stage2_official_train_no_hard_valid.jsonl}"
VALID_INPUT="${VALID_INPUT:-data/processed/proxy_all_family_valid.jsonl}"
OUT_DIR="${OUT_DIR:-data/processed}"

build_split() {
  local split_name="$1"
  local input="$2"
  local style="$3"
  local output="$4"
  local report="$5"
  if [[ ! -f "$input" ]]; then
    echo "Missing ${split_name} input: ${input}" >&2
    exit 1
  fi
  python -m src.student.sft_dataset_builder \
    --input "$input" \
    --output "$output" \
    --selection-profile stage2 \
    --prompt-mode chat_thinking \
    --tokenizer-path "$TOKENIZER_PATH" \
    --completion-style "$style" \
    --beam-width 10 \
    --max-depth 3 \
    --top-k 3 \
    --no-balance-by-family \
    --hard-triad-repeat-factor 1 \
    --max-per-signature-bucket 0 \
    --include-metadata \
    --report-output "$report"
}

validate_jsonl() {
  local path="$1"
  python - "$path" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
count = 0
empty = 0
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    count += 1
    if not row.get("prompt") or not row.get("completion"):
        empty += 1
if count <= 0:
    raise SystemExit(f"{path} is empty")
if empty:
    raise SystemExit(f"{path} has {empty} rows without prompt/completion")
print(json.dumps({"path": str(path), "rows": count}, indent=2))
PY
}

mkdir -p "$OUT_DIR"

build_split train "$TRAIN_INPUT" answer_only \
  "$OUT_DIR/stage2_answer_only_train.jsonl" \
  "$OUT_DIR/stage2_answer_only_train_report.json"
build_split valid "$VALID_INPUT" answer_only \
  "$OUT_DIR/stage2_answer_only_valid.jsonl" \
  "$OUT_DIR/stage2_answer_only_valid_report.json"
build_split train "$TRAIN_INPUT" short_trace \
  "$OUT_DIR/stage2_short_trace_train.jsonl" \
  "$OUT_DIR/stage2_short_trace_train_report.json"
build_split valid "$VALID_INPUT" short_trace \
  "$OUT_DIR/stage2_short_trace_valid.jsonl" \
  "$OUT_DIR/stage2_short_trace_valid_report.json"

validate_jsonl "$OUT_DIR/stage2_answer_only_train.jsonl"
validate_jsonl "$OUT_DIR/stage2_answer_only_valid.jsonl"
validate_jsonl "$OUT_DIR/stage2_short_trace_train.jsonl"
validate_jsonl "$OUT_DIR/stage2_short_trace_valid.jsonl"

echo "answer-focused stage2 datasets ready under ${OUT_DIR}"
