#!/usr/bin/env bash
set -euo pipefail

# Run on the GPU server from /workspace/nivida_h200_run.
# Batch 1 is eval-only: bootstrap/preflight vLLM, run a cheap smoke sweep, and
# optionally run the full exact-eval arena. Kaggle submission decisions stay local.

REPO_DIR="${REPO_DIR:-/workspace/nivida_h200_run}"
cd "$REPO_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"
}

BOOTSTRAP_VLLM="${BOOTSTRAP_VLLM:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FULL="${RUN_FULL:-0}"
PULL_LATEST="${PULL_LATEST:-1}"
VLLM_VERSION="${VLLM_VERSION:-0.14.0}"
VENV_ROOT="${VENV_ROOT:-/workspace/venvs/nemotron_vllm_${VLLM_VERSION//./_}}"
VENV="${VENV:-$VENV_ROOT}"
OUT_DIR="${OUT_DIR:-data/processed/vllm_exact_eval_v3_batch1}"
CONFIG="${CONFIG:-configs/train_stage2_official_balanced_answer_only.yaml}"
SMOKE_EVAL_INPUTS="${SMOKE_EVAL_INPUTS:-smoke_head6}"
FULL_EVAL_INPUTS="${FULL_EVAL_INPUTS:-combined_balanced_48pf,proxy_all_balanced_64pf,hard_triad_full}"
PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT:-data/processed/cloud_eval_batch1_preflight.json}"

SUBMIT_SAFE_ADAPTERS="${ADAPTERS:-official_balanced=artifacts/adapter_stage2_thin_official_balanced_20260424_161110Z,answer_final=artifacts/adapter_stage2_official_balanced_answer_only,short_trace_final=artifacts/adapter_stage2_official_balanced_short_trace,bit_rescue_v2_20260430_trained=artifacts/adapter_stage2_bit_rescue_v2}"
PREFLIGHT_CANDIDATE_ARGS=()
IFS=',' read -ra CANDIDATE_ITEMS <<< "$SUBMIT_SAFE_ADAPTERS"
for item in "${CANDIDATE_ITEMS[@]}"; do
  [[ -n "$item" ]] || continue
  PREFLIGHT_CANDIDATE_ARGS+=(--candidate "$item")
done

if [[ "$PULL_LATEST" == "1" ]]; then
  log "git pull --ff-only"
  git pull --ff-only
fi

if [[ "$BOOTSTRAP_VLLM" == "1" ]]; then
  log "bootstrap vLLM environment"
  VLLM_VERSION="$VLLM_VERSION" VENV_ROOT="$VENV_ROOT" bash scripts/bootstrap_cloud_vllm_env.sh
else
  log "check existing vLLM environment"
  VENV="$VENV" bash scripts/check_cloud_vllm_env.sh
fi

log "cloud eval batch1 artifact preflight"
python scripts/check_cloud_eval_inputs.py \
  --eval-inputs "$SMOKE_EVAL_INPUTS" \
  "${PREFLIGHT_CANDIDATE_ARGS[@]}" \
  --config "$CONFIG" \
  --output "$PREFLIGHT_OUTPUT"

if [[ "$RUN_SMOKE" == "1" ]]; then
  log "batch1 smoke exact-eval: ${SMOKE_EVAL_INPUTS}"
  VENV="$VENV" \
  OUT_DIR="$OUT_DIR" \
  CONFIG="$CONFIG" \
  EVAL_INPUTS="$SMOKE_EVAL_INPUTS" \
  ADAPTERS="$SUBMIT_SAFE_ADAPTERS" \
  bash scripts/run_cloud_vllm_exact_eval_v3.sh
fi

if [[ "$RUN_FULL" == "1" ]]; then
  log "batch1 full exact-eval: ${FULL_EVAL_INPUTS}"
  VENV="$VENV" \
  OUT_DIR="$OUT_DIR" \
  CONFIG="$CONFIG" \
  EVAL_INPUTS="$FULL_EVAL_INPUTS" \
  ADAPTERS="$SUBMIT_SAFE_ADAPTERS" \
  bash scripts/run_cloud_vllm_exact_eval_v3.sh
else
  log "smoke only complete; rerun with RUN_FULL=1 for full exact-eval"
fi

cat <<EOF

Cloud batch1 generation complete.

Pull back this directory and score locally:

  ${OUT_DIR}

Local scoring command:

  python scripts/score_vllm_exact_eval_outputs.py \\
    --predictions-root ${OUT_DIR} \\
    --output-root data/processed/eval/vllm_exact_eval_v3_batch1

No Kaggle submission was attempted by this script.
EOF
