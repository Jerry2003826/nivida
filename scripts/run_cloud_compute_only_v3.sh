#!/usr/bin/env bash
set -euo pipefail

# Run this on the GPU server from /workspace/nivida_h200_run.
# It performs compute-only work: route probing and optional inference.
# It does not train and does not submit to Kaggle.

cd "${REPO_DIR:-/workspace/nivida_h200_run}"

resolve_activate_path() {
  local candidate="$1"
  if [[ -d "$candidate" && -f "$candidate/bin/activate" ]]; then
    printf '%s\n' "$candidate/bin/activate"
    return 0
  fi
  if [[ -f "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  echo "Missing virtualenv activate script: $candidate" >&2
  echo "Set VENV to either a venv directory or its bin/activate file." >&2
  return 1
}

ACTIVATE_PATH="$(resolve_activate_path "${VENV:-/workspace/venvs/nemotron_t241/bin/activate}")"
VENV_ROOT="$(cd "$(dirname "$ACTIVATE_PATH")/.." && pwd)"
source "$ACTIVATE_PATH"
export LD_LIBRARY_PATH="${VENV_ROOT}/lib/python3.12/site-packages/nvidia/cusparselt/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export KAGGLEHUB_CACHE="${KAGGLEHUB_CACHE:-/workspace/.cache/kagglehub}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

OUT_DIR="${OUT_DIR:-data/processed/route_probe_v3}"
LOG_DIR="${LOG_DIR:-logs/setup_new_machine}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"
}

probe_prompt_examples() {
  local name="$1"
  local input="$2"
  local limit="$3"
  local weight_note="$4"
  log "probe ${name}: input=${input} limit=${limit} ${weight_note}"
  python scripts/probe_nemotron_expert_routes.py \
    --config configs/train_stage2_thin.yaml \
    --input "$input" \
    --adapter-dir artifacts/adapter_stage2_thin \
    --output "${OUT_DIR}/${name}_prompt_examples.json" \
    --limit "$limit" \
    --top-k 8 \
    --count-scope prompt \
    --record-examples
}

log "compute-only v3 start"

if [[ "${USE_BATCH_PROBE:-1}" == "1" ]]; then
  log "batch probe prompt-only jobs with one model load"
  python scripts/probe_nemotron_expert_routes_batch.py \
    --config configs/train_stage2_thin.yaml \
    --adapter-dir artifacts/adapter_stage2_thin \
    --top-k 8 \
    --job "official_hard=data/processed/stage2_official_valid_hard_triad.jsonl:${OUT_DIR}/official_hard_prompt_examples.json:256:prompt:0" \
    --job "official_all=data/processed/proxy_all_family_valid.jsonl:${OUT_DIR}/official_all_prompt_examples.json:256:prompt:0" \
    --job "stage2_train=data/processed/stage2_distill_train.jsonl:${OUT_DIR}/stage2_train_prompt_examples.json:512:prompt:0" \
    --job "public_visible=官方资料/test.csv:${OUT_DIR}/public_visible_prompt_examples.json:35:prompt:0"
else
  probe_prompt_examples official_hard data/processed/stage2_official_valid_hard_triad.jsonl 256 "no-public selection weight 0.40"
  probe_prompt_examples official_all data/processed/proxy_all_family_valid.jsonl 256 "no-public selection weight 0.40"
  probe_prompt_examples stage2_train data/processed/stage2_distill_train.jsonl 512 "no-public selection weight 0.20"
  probe_prompt_examples public_visible 官方资料/test.csv 35 "diagnostic only"
fi

if [[ "${RUN_PUBLIC_GENERATION_DIAG:-0}" == "1" ]]; then
  log "probe public_visible generation_delta diagnostic"
  python scripts/probe_nemotron_expert_routes.py \
    --config configs/train_stage2_thin.yaml \
    --input 官方资料/test.csv \
    --adapter-dir artifacts/adapter_stage2_thin \
    --output "${OUT_DIR}/public_visible_generation_delta_examples.json" \
    --limit 35 \
    --top-k 8 \
    --max-new-tokens 192 \
    --count-scope generation_delta \
    --record-examples
fi

log "mix no-public per-example normalized route report"
python scripts/mix_route_reports.py \
  --normalization example \
  --input "${OUT_DIR}/official_hard_prompt_examples.json:0.40" \
  --input "${OUT_DIR}/official_all_prompt_examples.json:0.40" \
  --input "${OUT_DIR}/stage2_train_prompt_examples.json:0.20" \
  --output "${OUT_DIR}/mixed_example_norm_no_public_hard40_all40_train20.json"

log "mix public diagnostic per-example normalized route report"
python scripts/mix_route_reports.py \
  --normalization example \
  --input "${OUT_DIR}/public_visible_prompt_examples.json:0.25" \
  --input "${OUT_DIR}/official_hard_prompt_examples.json:0.30" \
  --input "${OUT_DIR}/official_all_prompt_examples.json:0.30" \
  --input "${OUT_DIR}/stage2_train_prompt_examples.json:0.15" \
  --output "${OUT_DIR}/mixed_example_norm_with_public_diagnostic.json"

if [[ "${RUN_INFERENCE:-0}" == "1" ]]; then
  EVAL_INPUT="${EVAL_INPUT:-data/processed/proxy_all_family_valid.jsonl}"
  PRED_DIR="${PRED_DIR:-data/processed/local_eval_predictions_v3}"
  mkdir -p "$PRED_DIR"
  log "run inference for B thin on ${EVAL_INPUT}"
  python -m src.student.inference \
    --config configs/train_stage2_thin.yaml \
    --input "$EVAL_INPUT" \
    --adapter-dir artifacts/adapter_stage2_thin \
    --output "${PRED_DIR}/b_thin_predictions.jsonl" \
    --runtime-eval
fi

log "compute-only v3 done"
