#!/usr/bin/env bash
# round1a_ablation_harness.sh
# ============================
# Round 1-A: 24-hour ablation gate for the GCD teacher-distillation
# recipe, comparing three short LoRA training runs head-to-head:
#
#   A. BASELINE  — current CoT distillation pipeline (Stage-1 + Stage-2
#                  recipe exactly as Round 0 produces today).  No change.
#   B. GCD-FIXED — Round 1 GCD with FIXED N=16 and single-trace retention
#                  (the original round1_gcd_teacher_distill.py behaviour).
#   C. GCD-ROUND3 — Round 1 GCD with adaptive N=4→8→16, multi-trace
#                   retention (up to 2), support-weighted sample weights,
#                   token_weights=3.0 on \boxed{} / 0.5 on rationale.
#
# Each run is capped at ~12 hours of H200 training wall-clock so three
# runs fit in a 36-hour gate.  Proxy eval on the competition's
# ``competition_correct_rate`` at step-end drives the go/no-go call.
#
# Decision gate (must pass ALL three to promote recipe C to main):
#   1. C.correct_rate >= B.correct_rate + 0.005
#   2. C.correct_rate >= A.correct_rate + 0.010
#   3. C.mean_output_tokens <= 1.15 * B.mean_output_tokens
#      (Round 3 compute-efficiency guard: no >15% generation-length
#       bloat at inference time)
#
# Output: artifacts/round1a_ablation_report.json with per-run metrics
# and a single go/no-go decision field.
#
# Environment assumptions (Nemotron training machine):
#   /workspace/nivida_h200_run/
#     venvs/nemotron_t241/       (activated below)
#     artifacts/                 (run outputs land here)
#     data/processed/            (preprocessed jsonl data)
#
# DO NOT run this on the main training host while stage2 is still in
# progress — it will compete for the GPU.  Round 1-A must wait for
# stage2 completion; the harness will refuse to launch if
# /workspace/nivida_h200_run/artifacts/adapter_stage2_bestproxy/adapter_model.safetensors
# is missing.
# ---------------------------------------------------------------------------

set -euo pipefail

ROOT="${ROOT:-/workspace/nivida_h200_run}"
VENV="${VENV:-/workspace/venvs/nemotron_t241/bin/activate}"
INPUT_JSONL="${INPUT_JSONL:-${ROOT}/data/processed/round1_nmath_v2.jsonl}"
BASE_CONFIG="${BASE_CONFIG:-${ROOT}/configs/train_lora.yaml}"
OUT_DIR="${OUT_DIR:-${ROOT}/artifacts/round1a}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/round1a}"

# Guard: only launch after Stage-2 LoRA is ready
STAGE2_ADAPTER="${ROOT}/artifacts/adapter_stage2_bestproxy/adapter_model.safetensors"
if [[ ! -f "${STAGE2_ADAPTER}" ]]; then
  echo "[round1a] ABORT: Stage-2 adapter not found at ${STAGE2_ADAPTER}. Wait for stage2 to finish." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# Ensure venv
# shellcheck disable=SC1090
source "${VENV}"

cd "${ROOT}"

# ---------------------------------------------------------------------------
# STEP 1 — Distil data for each variant (runs on CPU/GPU in <2h each)
# ---------------------------------------------------------------------------

echo "[round1a] === Step 1: distilling data variants ==="

# Variant B: fixed N=16, single-trace (Round 0 baseline GCD)
if [[ ! -s "${OUT_DIR}/gcd_fixed.jsonl" ]]; then
  python scripts/round1_gcd_teacher_distill.py \
    --input "${INPUT_JSONL}" \
    --output "${OUT_DIR}/gcd_fixed.jsonl" \
    --config "${BASE_CONFIG}" \
    --num-samples 16 \
    --adaptive-n-tiers 16 \
    --max-traces-per-prompt 1 \
    --teacher-temperature 1.0 \
    --workdir "${OUT_DIR}/workdir_fixed" \
    2>&1 | tee "${LOG_DIR}/distill_fixed.log"
fi

# Variant C: Round 3 adaptive N + multi-trace
if [[ ! -s "${OUT_DIR}/gcd_round3.jsonl" ]]; then
  python scripts/round1_gcd_teacher_distill.py \
    --input "${INPUT_JSONL}" \
    --output "${OUT_DIR}/gcd_round3.jsonl" \
    --config "${BASE_CONFIG}" \
    --num-samples 16 \
    --adaptive-n-tiers 4 8 16 \
    --max-traces-per-prompt 2 \
    --teacher-temperature 1.0 \
    --workdir "${OUT_DIR}/workdir_round3" \
    2>&1 | tee "${LOG_DIR}/distill_round3.log"
fi

# Variant A uses the current Stage-1+Stage-2 CoT jsonl already on disk.
BASELINE_JSONL="${BASELINE_JSONL:-${ROOT}/data/processed/stage2_cot.jsonl}"
if [[ ! -s "${BASELINE_JSONL}" ]]; then
  echo "[round1a] WARNING: baseline jsonl ${BASELINE_JSONL} missing." >&2
fi

# ---------------------------------------------------------------------------
# STEP 2 — Launch three 12-hour LoRA fine-tunes (sequential, not parallel;
#          single H200)
# ---------------------------------------------------------------------------

echo "[round1a] === Step 2: three LoRA fine-tunes ==="

MAX_HOURS="${MAX_HOURS:-12}"

run_one () {
  local tag="$1" data_jsonl="$2" extra_args="$3"
  local run_out="${OUT_DIR}/run_${tag}"
  mkdir -p "${run_out}"
  if [[ -f "${run_out}/adapter_model.safetensors" ]]; then
    echo "[round1a] ${tag}: adapter already exists, skipping training."
    return 0
  fi
  echo "[round1a] ${tag}: launching ${MAX_HOURS}h run on ${data_jsonl}"
  # shellcheck disable=SC2086
  timeout "${MAX_HOURS}h" python -m src.student.lora_train \
      --config "${BASE_CONFIG}" \
      --data "${data_jsonl}" \
      --resume-from "${ROOT}/artifacts/adapter_stage2_bestproxy" \
      --output-dir "${run_out}" \
      ${extra_args} \
      2>&1 | tee "${LOG_DIR}/train_${tag}.log" || true
}

run_one "A_baseline" "${BASELINE_JSONL}" "--stage ablation_A"
run_one "B_gcd_fixed" "${OUT_DIR}/gcd_fixed.jsonl" "--stage ablation_B"
run_one "C_gcd_round3" "${OUT_DIR}/gcd_round3.jsonl" "--stage ablation_C --use-sample-weight --use-token-weights"

# ---------------------------------------------------------------------------
# STEP 3 — Proxy-eval each run under runtime contract
# ---------------------------------------------------------------------------

echo "[round1a] === Step 3: proxy eval (runtime contract) ==="

eval_one () {
  local tag="$1"
  local run_out="${OUT_DIR}/run_${tag}"
  local report="${run_out}/bestproxy_eval.json"
  if [[ -f "${report}" ]]; then
    echo "[round1a] ${tag}: eval already exists, skipping."
    return 0
  fi
  python scripts/eval_official_vllm_proxy.py \
    --adapter "${run_out}" \
    --contract runtime \
    --subset-size 500 \
    --output "${report}" \
    2>&1 | tee "${LOG_DIR}/eval_${tag}.log"
}

eval_one "A_baseline"
eval_one "B_gcd_fixed"
eval_one "C_gcd_round3"

# ---------------------------------------------------------------------------
# STEP 4 — Aggregate + decision
# ---------------------------------------------------------------------------

echo "[round1a] === Step 4: aggregating + decision ==="

python - <<'PY' "${OUT_DIR}"
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
variants = {
    "A_baseline":   out / "run_A_baseline"   / "bestproxy_eval.json",
    "B_gcd_fixed":  out / "run_B_gcd_fixed"  / "bestproxy_eval.json",
    "C_gcd_round3": out / "run_C_gcd_round3" / "bestproxy_eval.json",
}
results = {}
for tag, p in variants.items():
    if not p.exists():
        results[tag] = {"missing": True, "correct_rate": None, "mean_output_tokens": None}
        continue
    data = json.loads(p.read_text())
    results[tag] = {
        "correct_rate": float(data.get("competition_correct_rate", 0.0) or 0.0),
        "mean_output_tokens": float(data.get("mean_output_tokens", 0.0) or 0.0),
    }

a = results["A_baseline"]
b = results["B_gcd_fixed"]
c = results["C_gcd_round3"]

decision = {
    "a_correct_rate": a.get("correct_rate"),
    "b_correct_rate": b.get("correct_rate"),
    "c_correct_rate": c.get("correct_rate"),
    "b_mean_output_tokens": b.get("mean_output_tokens"),
    "c_mean_output_tokens": c.get("mean_output_tokens"),
    "gate1_c_beats_b_by_005": None,
    "gate2_c_beats_a_by_010": None,
    "gate3_no_length_bloat_15pct": None,
    "final": None,
}
if all(results[k].get("correct_rate") is not None for k in variants):
    decision["gate1_c_beats_b_by_005"] = (c["correct_rate"] - b["correct_rate"]) >= 0.005
    decision["gate2_c_beats_a_by_010"] = (c["correct_rate"] - a["correct_rate"]) >= 0.010
    b_tok = b.get("mean_output_tokens") or 0.0
    c_tok = c.get("mean_output_tokens") or 0.0
    decision["gate3_no_length_bloat_15pct"] = (b_tok <= 0.0) or (c_tok <= 1.15 * b_tok)
    decision["final"] = (
        "PROMOTE_ROUND3"
        if all([
            decision["gate1_c_beats_b_by_005"],
            decision["gate2_c_beats_a_by_010"],
            decision["gate3_no_length_bloat_15pct"],
        ])
        else "STAY_ON_BASELINE"
    )
report = {"variants": results, "decision": decision}
(out / "ablation_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
PY

echo "[round1a] Done.  See ${OUT_DIR}/ablation_report.json"
