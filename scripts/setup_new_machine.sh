#!/usr/bin/env bash
# Full-auto bootstrap + train + package pipeline for a fresh GPU machine.
#
# Designed for RTX PRO 6000 Blackwell 96GB (or H100/H200 80GB+). The pipeline
# runs:
#   1. bootstrap deps (pip install)
#   2. kagglehub / tokenizer sanity probe
#   3. data prep (official + synth hard triads)
#   4. stage1 smoke (gradients + loss sanity gate)
#   5. stage1 full training + acceptance check
#   6. stage2 distill (full script, includes bestproxy selection)
#   7. stage3 repair (full script, includes bestproxy selection)
#   8. select_final_adapter (stage2-vs-stage3 decision)
#   9. validate_submission + build submission.zip
#
# Configuration via env vars:
#   WORKSPACE_DIR (default /workspace/nivida_h200_run)
#   KAGGLEHUB_CACHE (default /workspace/.cache/kagglehub)
#   HF_HOME (default /workspace/.cache/huggingface)
#   KAGGLE_USERNAME, KAGGLE_KEY - required for base model download.
#   SKIP_SMOKE=1 to skip the stage1 smoke gate.
#   DRY_RUN=1 prints commands without executing training (data still prepared).
#
# The script is idempotent: reruns skip a stage if its canonical output already
# exists.  Delete the relevant artifacts/ directory to force a rerun.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/nivida_h200_run}"
export KAGGLEHUB_CACHE="${KAGGLEHUB_CACHE:-/workspace/.cache/kagglehub}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

TOKENIZER_PATH="${TOKENIZER_PATH:-artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$KAGGLEHUB_CACHE" "$HF_HOME" artifacts data/processed data/synthetic logs

LOG_DIR="${LOG_DIR:-logs/setup_new_machine}"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

log()  { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
run()  {
  log "RUN: $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  eval "$@"
}

######################################################################
# 1. Bootstrap deps
######################################################################
if [[ ! -f .setup_new_machine.bootstrapped ]]; then
  log "Installing Python dependencies (pip)"
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  pip install -e .[train]

  # Stack for Nemotron-3-Nano-30B bf16 on Blackwell / Hopper:
  #   - torch 2.4.x + CUDA 12.4 wheels work on both H100 and RTX PRO 6000.
  #   - mamba-ssm 2.2.4+ requires causal-conv1d built for the same CUDA.
  #   - vLLM is optional for proxy eval; skip if the wheel fails to build.
  pip install --upgrade \
    "torch>=2.4,<2.6" \
    "transformers>=4.44.0" \
    "peft>=0.12.0" \
    "accelerate>=0.33.0" \
    "datasets>=2.20.0" \
    "kagglehub>=0.3.12" \
    "safetensors>=0.4.3" \
    "sentencepiece>=0.2.0" \
    "protobuf>=4.25"
  pip install "causal-conv1d>=1.4.0" "mamba-ssm>=2.2.4" || {
    log "mamba-ssm install failed - try matching CUDA wheels manually" >&2
    exit 1
  }

  touch .setup_new_machine.bootstrapped
else
  log "Bootstrap already done; skipping pip install"
fi

######################################################################
# 2. Kaggle / tokenizer sanity
######################################################################
if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
  log "WARNING: KAGGLE_USERNAME / KAGGLE_KEY not set. Base-model download will fail."
fi

if [[ ! -f artifacts/chat_template_probe.json ]]; then
  run python scripts/probe_chat_template.py
fi

######################################################################
# 3. Data prep
######################################################################
if [[ ! -f data/processed/official_train_tagged.jsonl || ! -f data/splits/official/splits.json ]]; then
  run python scripts/prepare_data.py --config configs/data_official.yaml
fi

if [[ ! -f data/synthetic/synth_hard_triads.jsonl ]]; then
  run python -m src.teacher.synth_generator --config configs/synth_hard_triads.yaml
fi

# stage1 SFT datasets (subset-free by construction; stage2/3 scripts build
# their own subsets).
if [[ ! -f data/processed/stage1_format_align_train.jsonl ]]; then
  run bash scripts/train_stage1_format_align.sh \
    2>&1 | tee "$LOG_DIR/${STAMP}.stage1_prep.log" || true
fi

######################################################################
# 4. Stage1 smoke gate
######################################################################
SMOKE_DIR=artifacts/smoke/adapter_stage1_format
if [[ "$SKIP_SMOKE" != "1" && ! -f "$SMOKE_DIR/adapter_model.safetensors" ]]; then
  log "Running stage1 smoke (50-step gradient / loss sanity gate)"
  run bash scripts/train_stage1_smoke.sh 2>&1 | tee "$LOG_DIR/${STAMP}.smoke.log"
  run python scripts/check_smoke_health.py \
    --adapter-dir "$SMOKE_DIR" \
    --max-loss 10.0 \
    --max-grad-norm 1e10 \
    --min-steps 3 \
    --output artifacts/smoke/health_report.json
  log "Smoke gate passed"
fi

######################################################################
# 5. Stage1 full training + acceptance
######################################################################
STAGE1_DIR=artifacts/adapter_stage1_format
if [[ ! -f "$STAGE1_DIR/adapter_model.safetensors" \
      || ! -f "$STAGE1_DIR/last_run_summary.json" ]]; then
  log "Running stage1 full training"
  run python -m src.student.lora_train \
    --config configs/train_stage1_format.yaml \
    --force-train 2>&1 | tee "$LOG_DIR/${STAMP}.stage1.log"
fi

run python scripts/check_stage1_acceptance.py \
  --adapter-dir "$STAGE1_DIR" \
  --output artifacts/adapter_stage1_format/acceptance_report.json

######################################################################
# 6. Stage2 distill + bestproxy
######################################################################
STAGE2_BEST_DIR=artifacts/adapter_stage2_bestproxy
if [[ ! -f "$STAGE2_BEST_DIR/adapter_model.safetensors" ]]; then
  log "Running stage2 distill + bestproxy selection"
  run bash scripts/train_stage2_distill.sh 2>&1 | tee "$LOG_DIR/${STAMP}.stage2.log"
fi

######################################################################
# 7. Stage3 repair + bestproxy
######################################################################
STAGE3_BEST_DIR=artifacts/adapter_stage3_bestproxy
if [[ ! -f "$STAGE3_BEST_DIR/adapter_model.safetensors" ]]; then
  log "Running stage3 repair + bestproxy selection"
  run bash scripts/train_stage3_repair.sh 2>&1 | tee "$LOG_DIR/${STAMP}.stage3.log"
fi

######################################################################
# 8. Select final adapter (stage2 vs stage3)
######################################################################
FINAL_DIR=artifacts/adapter_final
FINAL_JSON=artifacts/final_adapter_selection.json
run python scripts/select_final_adapter.py \
  --stage2-hard-eval data/processed/stage2_bestproxy_hard_eval.json \
  --stage2-all-eval  data/processed/stage2_bestproxy_all_eval.json \
  --stage2-adapter-dir "$STAGE2_BEST_DIR" \
  --stage3-hard-eval data/processed/stage3_bestproxy_hard_eval.json \
  --stage3-all-eval  data/processed/stage3_bestproxy_all_eval.json \
  --stage3-adapter-dir "$STAGE3_BEST_DIR" \
  --output-adapter-dir "$FINAL_DIR" \
  --output-json "$FINAL_JSON"

######################################################################
# 9. Validate submission + package zip
######################################################################
SUBMISSION_ZIP="${SUBMISSION_ZIP:-submission.zip}"
run python scripts/validate_submission.py \
  --config configs/train_stage3_repair.yaml \
  --adapter-dir "$FINAL_DIR" \
  --output artifacts/submission_validation.json \
  --package-output "$SUBMISSION_ZIP"

log "Done. Final adapter: $FINAL_DIR"
log "Submission zip   : $SUBMISSION_ZIP"
log "Selection report : $FINAL_JSON"
log "All step logs    : $LOG_DIR/${STAMP}.*.log"
