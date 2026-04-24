#!/usr/bin/env bash
set -euo pipefail

# Run on the GPU server from /workspace/nivida_h200_run.
# This script only generates prediction JSONL files. Scoring is local.

cd "${REPO_DIR:-/workspace/nivida_h200_run}"

source "${VENV:-/workspace/venvs/nemotron_t241/bin/activate}"
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export KAGGLEHUB_CACHE="${KAGGLEHUB_CACHE:-/workspace/.cache/kagglehub}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

EVAL_INPUT="${EVAL_INPUT:-data/processed/local_eval_manifests/proxy_all_balanced_32pf.jsonl}"
OUT_DIR="${OUT_DIR:-data/processed/local_eval_predictions_v3}"
CONFIG="${CONFIG:-configs/train_stage2_thin.yaml}"
mkdir -p "$OUT_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"
}

run_adapter() {
  local name="$1"
  local adapter="$2"
  log "inference ${name}: adapter=${adapter} input=${EVAL_INPUT}"
  python -m src.student.inference \
    --config "$CONFIG" \
    --input "$EVAL_INPUT" \
    --adapter-dir "$adapter" \
    --output "${OUT_DIR}/${name}_predictions.jsonl" \
    --runtime-eval
}

log "inference-only start"
run_adapter b_thin artifacts/adapter_stage2_thin

if [[ -d artifacts/adapter_stage2_thin_expertmean_shared_top8_s1 ]]; then
  run_adapter norm_shared_s1 artifacts/adapter_stage2_thin_expertmean_shared_top8_s1
fi

if [[ -d artifacts/adapter_stage2_thin_routeweighted_mixed ]]; then
  run_adapter raw_routeweighted_mixed artifacts/adapter_stage2_thin_routeweighted_mixed
fi

if [[ -n "${EXTRA_ADAPTERS:-}" ]]; then
  IFS=',' read -ra ITEMS <<< "$EXTRA_ADAPTERS"
  for item in "${ITEMS[@]}"; do
    name="${item%%=*}"
    path="${item#*=}"
    run_adapter "$name" "$path"
  done
fi

log "inference-only done"
