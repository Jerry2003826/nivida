#!/usr/bin/env bash
set -euo pipefail

# Run on the GPU server from /workspace/nivida_h200_run.
# Builds an isolated vLLM environment new enough for NemotronH MoE LoRA
# exact-eval, then runs the cheap environment preflight before generation.

cd "${REPO_DIR:-/workspace/nivida_h200_run}"

VLLM_VERSION="${VLLM_VERSION:-0.14.0}"
VLLM_MIN_VERSION="${VLLM_MIN_VERSION:-0.14.0}"
VENV_ROOT="${VENV_ROOT:-/workspace/venvs/nemotron_vllm_${VLLM_VERSION//./_}}"
DRY_RUN="${DRY_RUN:-0}"

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
  run python -m venv "$VENV_ROOT"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf '+ source %q\n' "$VENV_ROOT/bin/activate"
else
  # shellcheck disable=SC1091
  source "$VENV_ROOT/bin/activate"
fi

run python -m pip install --upgrade pip setuptools wheel packaging
run python -m pip install \
  "vllm==${VLLM_VERSION}" \
  kagglehub \
  "huggingface_hub[hf_transfer]" \
  packaging

export VENV="$VENV_ROOT"
export VLLM_MIN_VERSION
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

run bash scripts/check_cloud_vllm_env.sh
