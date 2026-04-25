#!/usr/bin/env bash
set -euo pipefail

# Run on the GPU server from /workspace/nivida_h200_run.
# This script only generates prediction JSONL files. Scoring is local.
#
# Defaults evaluate the LB-selection candidate set against the main local
# exact proxy plus two auxiliary proxies:
#
#   EVAL_INPUTS="combined_balanced_48pf,proxy_all_balanced_64pf,hard_triad_full"
#
# Candidate adapters may be supplied as:
#
#   ADAPTERS="name=path,name2=path2"
#
# If ADAPTERS is empty, this script discovers B thin, continuation checkpoints,
# the prior final adapters, and the known shared/route variants.

cd "${REPO_DIR:-/workspace/nivida_h200_run}"

source "${VENV:-/workspace/venvs/nemotron_t241/bin/activate}"
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export KAGGLEHUB_CACHE="${KAGGLEHUB_CACHE:-/workspace/.cache/kagglehub}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

OUT_DIR="${OUT_DIR:-data/processed/local_eval_predictions_v3}"
CONFIG="${CONFIG:-configs/train_stage2_thin.yaml}"
EVAL_INPUTS="${EVAL_INPUTS:-combined_balanced_48pf,proxy_all_balanced_64pf,hard_triad_full}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
mkdir -p "$OUT_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"
}

sanitize_name() {
  printf '%s' "$1" | tr '/:. ' '____' | tr -cd 'A-Za-z0-9_.-'
}

resolve_eval_input() {
  local item="$1"
  if [[ -f "$item" ]]; then
    printf '%s' "$item"
    return
  fi
  if [[ -f "data/processed/local_eval_manifests/${item}.jsonl" ]]; then
    printf '%s' "data/processed/local_eval_manifests/${item}.jsonl"
    return
  fi
  printf '%s' "$item"
}

add_candidate() {
  local name="$1"
  local adapter="$2"
  [[ -d "$adapter" ]] || return 0
  CANDIDATES+=("$(sanitize_name "$name")=$adapter")
}

discover_checkpoints() {
  local prefix="$1"
  local stage_dir="$2"
  [[ -d "$stage_dir" ]] || return 0
  local checkpoint
  while IFS= read -r checkpoint; do
    local step
    step="$(basename "$checkpoint" | sed 's/^checkpoint-//')"
    add_candidate "${prefix}_ckpt_${step}" "$checkpoint"
  done < <(find "$stage_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
}

ensure_adapter_config() {
  local adapter="$1"
  local reference="$2"
  if [[ ! -f "$adapter/adapter_config.json" && -f "$reference/adapter_config.json" ]]; then
    cp "$reference/adapter_config.json" "$adapter/adapter_config.json"
  fi
}

run_adapter() {
  local name="$1"
  local adapter="$2"
  local eval_name="$3"
  local eval_input="$4"
  local eval_out_dir="${OUT_DIR}/${eval_name}"
  mkdir -p "$eval_out_dir"
  log "inference ${name}: adapter=${adapter} input=${eval_input}"
  local cmd=(
    python -m src.student.inference
    --config "$CONFIG"
    --input "$eval_input"
    --adapter-dir "$adapter"
    --output "${eval_out_dir}/${name}_predictions.jsonl"
    --runtime-eval
  )
  if [[ -n "$MAX_NEW_TOKENS" ]]; then
    cmd+=(--max-new-tokens "$MAX_NEW_TOKENS")
  fi
  "${cmd[@]}"
}

log "inference-only start"

CANDIDATES=()
if [[ -n "${ADAPTERS:-}" ]]; then
  IFS=',' read -ra ITEMS <<< "$ADAPTERS"
  for item in "${ITEMS[@]}"; do
    [[ -n "$item" ]] || continue
    name="${item%%=*}"
    path="${item#*=}"
    add_candidate "$name" "$path"
  done
else
  add_candidate b_thin artifacts/adapter_stage2_thin
  discover_checkpoints b_thin artifacts/adapter_stage2_thin

  for pattern in \
    'adapter_stage2_thin_official_balanced*' \
    'adapter_stage2_thin_answer_only*' \
    'adapter_stage2_thin_short_trace*' \
    'adapter_stage2_official_balanced_answer_only*' \
    'adapter_stage2_official_balanced_short_trace*'
  do
    while IFS= read -r continuation_dir; do
      continuation_name="$(basename "$continuation_dir")"
      add_candidate "$continuation_name" "$continuation_dir"
      discover_checkpoints "$continuation_name" "$continuation_dir"
    done < <(find artifacts -maxdepth 1 -type d -name "$pattern" 2>/dev/null | sort -V)
  done

  add_candidate stage2_selected_trace artifacts/adapter_stage2_selected_trace
  add_candidate norm_shared_s1 artifacts/adapter_stage2_thin_expertmean_shared_top8_s1
  add_candidate raw_routeweighted_mixed artifacts/adapter_stage2_thin_routeweighted_mixed
fi

if [[ -n "${EXTRA_ADAPTERS:-}" ]]; then
  IFS=',' read -ra ITEMS <<< "$EXTRA_ADAPTERS"
  for item in "${ITEMS[@]}"; do
    [[ -n "$item" ]] || continue
    name="${item%%=*}"
    path="${item#*=}"
    add_candidate "$name" "$path"
  done
fi

if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
  echo "No adapter candidates found." >&2
  exit 1
fi

if [[ -d artifacts/adapter_stage2_thin ]]; then
  for item in "${CANDIDATES[@]}"; do
    ensure_adapter_config "${item#*=}" artifacts/adapter_stage2_thin
  done
fi

IFS=',' read -ra EVAL_ITEMS <<< "$EVAL_INPUTS"
for eval_item in "${EVAL_ITEMS[@]}"; do
  [[ -n "$eval_item" ]] || continue
  eval_input="$(resolve_eval_input "$eval_item")"
  if [[ ! -f "$eval_input" ]]; then
    echo "Missing eval input: $eval_input" >&2
    exit 1
  fi
  eval_name="$(sanitize_name "$(basename "$eval_input" .jsonl)")"
  for item in "${CANDIDATES[@]}"; do
    run_adapter "${item%%=*}" "${item#*=}" "$eval_name" "$eval_input"
  done
done

log "inference-only done"
