#!/usr/bin/env bash
set -euo pipefail

# Run on the GPU server from /workspace/nivida_h200_run.
# This uses the vLLM official-runtime proxy to produce exact-eval JSON reports
# plus raw generations. Scoring and submission decisions remain local.
#
# Candidate adapters may be supplied as:
#
#   ADAPTERS="name=path,name2=path2"
#
# EVAL_INPUTS accepts manifest names or explicit JSONL paths:
#
#   EVAL_INPUTS="smoke_head6,combined_balanced_48pf"

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

OUT_DIR="${OUT_DIR:-data/processed/vllm_exact_eval_v3}"
CONFIG="${CONFIG:-configs/train_stage2_official_balanced_answer_only.yaml}"
EVAL_INPUTS="${EVAL_INPUTS:-smoke_head6}"
CONTRACT="${CONTRACT:-runtime}"
PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT:-data/processed/cloud_eval_preflight.json}"
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

materialize_builtin_eval_inputs() {
  if [[ ",${EVAL_INPUTS}," == *",smoke_head6,"* ]]; then
    local source="data/processed/local_eval_manifests/smoke_6pf.jsonl"
    local target="data/processed/local_eval_manifests/smoke_head6.jsonl"
    if [[ ! -f "$target" ]]; then
      if [[ ! -f "$source" ]]; then
        echo "Missing source manifest for smoke_head6: $source" >&2
        exit 1
      fi
      mkdir -p "$(dirname "$target")"
      head -n 6 "$source" > "$target"
      log "materialized ${target} from ${source}"
    fi
  fi
}

add_candidate() {
  local name="$1"
  local adapter="$2"
  [[ -d "$adapter" ]] || return 0
  CANDIDATES+=("$(sanitize_name "$name")=$adapter")
}

require_candidate() {
  local name="$1"
  local adapter="$2"
  if [[ ! -d "$adapter" ]]; then
    echo "Missing explicit adapter candidate ${name}: ${adapter}" >&2
    exit 1
  fi
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

run_eval_artifact_preflight() {
  local cmd=(python scripts/check_cloud_eval_inputs.py --eval-inputs "$EVAL_INPUTS" --config "$CONFIG" --output "$PREFLIGHT_OUTPUT")
  local item
  for item in "${CANDIDATES[@]}"; do
    cmd+=(--candidate "$item")
  done
  log "cloud eval artifact preflight"
  "${cmd[@]}"
}

run_preflight() {
  if [[ "${SKIP_VLLM_PREFLIGHT:-0}" == "1" ]]; then
    return 0
  fi
  log "vLLM environment preflight"
  VENV="$ACTIVATE_PATH" \
    bash scripts/check_cloud_vllm_env.sh
}

run_adapter() {
  local name="$1"
  local adapter="$2"
  local eval_name="$3"
  local eval_input="$4"
  local eval_out_dir="${OUT_DIR}/${eval_name}/${name}"
  mkdir -p "$eval_out_dir/raw"
  log "vLLM exact eval ${name}: adapter=${adapter} input=${eval_input}"
  python scripts/eval_official_vllm_proxy.py \
    --adapter-dir "$adapter" \
    --input "$eval_input" \
    --output "${eval_out_dir}/report.json" \
    --config "$CONFIG" \
    --write-raw-predictions \
    --raw-predictions-dir "${eval_out_dir}/raw" \
    --contract "$CONTRACT"
}

write_artifact_manifest() {
  local manifest_output="${OUT_DIR}/artifact_manifest.json"
  local cmd=(python scripts/write_cloud_artifact_manifest.py --out-dir "$OUT_DIR" --eval-inputs "$EVAL_INPUTS" --preflight "$PREFLIGHT_OUTPUT" --output "$manifest_output")
  local item
  for item in "${CANDIDATES[@]}"; do
    cmd+=(--candidate "$item")
  done
  log "write cloud artifact manifest ${manifest_output}"
  "${cmd[@]}"
}

log "vLLM exact-eval v3 start"

CANDIDATES=()
if [[ -n "${ADAPTERS:-}" ]]; then
  IFS=',' read -ra ITEMS <<< "$ADAPTERS"
  for item in "${ITEMS[@]}"; do
    [[ -n "$item" ]] || continue
    require_candidate "${item%%=*}" "${item#*=}"
  done
else
  add_candidate b_thin artifacts/adapter_stage2_thin
  add_candidate official_balanced artifacts/adapter_stage2_thin_official_balanced_20260424_161110Z
  add_candidate answer_final artifacts/adapter_stage2_official_balanced_answer_only
  add_candidate short_trace_final artifacts/adapter_stage2_official_balanced_short_trace
  add_candidate mixed_answer_short artifacts/adapter_stage2_mixed_answer_short
  add_candidate equation_rescue artifacts/adapter_stage2_equation_rescue
  add_candidate bit_rescue artifacts/adapter_stage2_bit_rescue
  add_candidate eq_bit_rescue artifacts/adapter_stage2_eq_bit_rescue
  add_candidate final_answer_weighted artifacts/adapter_stage2_final_answer_weighted
  if [[ "${INCLUDE_SUBMISSION_UNSAFE:-0}" == "1" ]]; then
    add_candidate rank64_answer_only artifacts/adapter_stage2_rank64_answer_only
  fi
  discover_checkpoints answer artifacts/adapter_stage2_official_balanced_answer_only
  discover_checkpoints short_trace artifacts/adapter_stage2_official_balanced_short_trace
  discover_checkpoints mixed artifacts/adapter_stage2_mixed_answer_short
  discover_checkpoints equation artifacts/adapter_stage2_equation_rescue
  discover_checkpoints bit artifacts/adapter_stage2_bit_rescue
  discover_checkpoints eq_bit artifacts/adapter_stage2_eq_bit_rescue
  discover_checkpoints weighted artifacts/adapter_stage2_final_answer_weighted
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

materialize_builtin_eval_inputs
run_eval_artifact_preflight
run_preflight

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

write_artifact_manifest
log "vLLM exact-eval v3 done"
