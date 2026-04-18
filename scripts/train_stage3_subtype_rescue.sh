#!/usr/bin/env bash
set -euo pipefail

# Pre-wired stage3 scaffold for the subtype-rescue branch.
#
# This script intentionally reuses the canonical stage3 orchestration by
# exporting branch-specific paths, then delegating to
# scripts/train_stage3_repair.sh. It stays dormant until
# scripts/decide_subtype_branch_promotion.py says the branch earned a stage3.

SUBTYPE_BRANCH_PROMOTION_JSON="${SUBTYPE_BRANCH_PROMOTION_JSON:-data/processed/stage2_subtype_rescue_promotion.json}"
ALLOW_UNPROMOTED_SUBTYPE_STAGE3="${ALLOW_UNPROMOTED_SUBTYPE_STAGE3:-0}"
I_UNDERSTAND_SUBTYPE_STAGE3_WAS_NOT_PROMOTED="${I_UNDERSTAND_SUBTYPE_STAGE3_WAS_NOT_PROMOTED:-0}"

if [[ "$ALLOW_UNPROMOTED_SUBTYPE_STAGE3" == "1" ]]; then
  if [[ "$I_UNDERSTAND_SUBTYPE_STAGE3_WAS_NOT_PROMOTED" != "1" ]]; then
    echo "Manual subtype stage3 override requires I_UNDERSTAND_SUBTYPE_STAGE3_WAS_NOT_PROMOTED=1" >&2
    exit 1
  fi
  python scripts/check_subtype_branch_promotion.py \
    --promotion-json "$SUBTYPE_BRANCH_PROMOTION_JSON" \
    --allow-unpromoted
else
  python scripts/check_subtype_branch_promotion.py \
    --promotion-json "$SUBTYPE_BRANCH_PROMOTION_JSON"
fi

export TOKENIZER_PATH="${TOKENIZER_PATH:-artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default}"

export STAGE2_ADAPTER_DIR="${STAGE2_ADAPTER_DIR:-artifacts/adapter_stage2_subtype_rescue_bestproxy}"
export STAGE2_INFERENCE_CONFIG="${STAGE2_INFERENCE_CONFIG:-configs/train_stage2_selected_trace_subtype_rescue.yaml}"
export STAGE2_BESTPROXY_HARD_EVAL="${STAGE2_BESTPROXY_HARD_EVAL:-data/processed/stage2_subtype_rescue_bestproxy_hard_eval.json}"
export STAGE2_BESTPROXY_ALL_EVAL="${STAGE2_BESTPROXY_ALL_EVAL:-data/processed/stage2_subtype_rescue_bestproxy_all_eval.json}"

export FINAL_STAGE3_ADAPTER_DIR="${FINAL_STAGE3_ADAPTER_DIR:-artifacts/adapter_stage3_subtype_rescue}"
export STAGE3_BESTPROXY_DIR="${STAGE3_BESTPROXY_DIR:-artifacts/adapter_stage3_subtype_rescue_bestproxy}"
export STAGE3_BESTPROXY_HARD_EVAL="${STAGE3_BESTPROXY_HARD_EVAL:-data/processed/stage3_subtype_rescue_bestproxy_hard_eval.json}"
export STAGE3_BESTPROXY_ALL_EVAL="${STAGE3_BESTPROXY_ALL_EVAL:-data/processed/stage3_subtype_rescue_bestproxy_all_eval.json}"
export STAGE3_BESTPROXY_SELECTION_JSON="${STAGE3_BESTPROXY_SELECTION_JSON:-data/processed/stage3_subtype_rescue_best_checkpoint_selection.json}"
export STAGE3_BESTPROXY_WORKDIR="${STAGE3_BESTPROXY_WORKDIR:-artifacts/_proxy_checkpoint_scratch/stage3_subtype_rescue}"

export REPAIR_TRAIN_SUBSET="${REPAIR_TRAIN_SUBSET:-data/processed/stage3_subtype_rescue_subset_train.jsonl}"
export REPLAY_TRAIN_SUBSET="${REPLAY_TRAIN_SUBSET:-data/processed/stage3_subtype_rescue_replay_pool_train.jsonl}"
export VALID_SUBSET="${VALID_SUBSET:-data/processed/stage3_subtype_rescue_subset_valid.jsonl}"

export FULL_TRAIN_PREDICTIONS="${FULL_TRAIN_PREDICTIONS:-data/processed/stage2_subtype_rescue_full_train_predictions.jsonl}"
export VALID_PREDICTIONS="${VALID_PREDICTIONS:-data/processed/stage2_subtype_rescue_valid_predictions.jsonl}"

export FULL_TRAIN_EVAL="${FULL_TRAIN_EVAL:-data/processed/stage2_subtype_rescue_full_train_eval.json}"
export VALID_EVAL="${VALID_EVAL:-data/processed/stage2_subtype_rescue_model_eval_valid.json}"

export TRAIN_FAILURES="${TRAIN_FAILURES:-data/processed/stage2_subtype_rescue_model_failures_train.json}"
export TRAIN_SUCCESSES="${TRAIN_SUCCESSES:-data/processed/stage2_subtype_rescue_model_successes_all_train.json}"
export VALID_FAILURES="${VALID_FAILURES:-data/processed/stage2_subtype_rescue_model_failures_valid.json}"

export STAGE3_TRAIN_REPORT="${STAGE3_TRAIN_REPORT:-data/processed/stage3_subtype_rescue_train_report.json}"
export STAGE3_VALID_REPORT="${STAGE3_VALID_REPORT:-data/processed/stage3_subtype_rescue_valid_report.json}"
export REPAIR_STAGE3_TRAIN_DATASET="${REPAIR_STAGE3_TRAIN_DATASET:-data/processed/stage3_subtype_rescue_train.jsonl}"
export REPAIR_STAGE3_VALID_DATASET="${REPAIR_STAGE3_VALID_DATASET:-data/processed/stage3_subtype_rescue_valid.jsonl}"

export STAGE3_DECISION="${STAGE3_DECISION:-data/processed/stage3_subtype_rescue_decision.json}"
export STAGE3_PROXY_VALID_PREDICTIONS="${STAGE3_PROXY_VALID_PREDICTIONS:-data/processed/stage3_subtype_rescue_proxy_valid_predictions.jsonl}"
export STAGE3_PROXY_VALID_EVAL="${STAGE3_PROXY_VALID_EVAL:-data/processed/stage3_subtype_rescue_proxy_valid_eval.json}"
export STAGE3_PROXY_ALL_VALID_PREDICTIONS="${STAGE3_PROXY_ALL_VALID_PREDICTIONS:-data/processed/stage3_subtype_rescue_proxy_all_valid_predictions.jsonl}"
export STAGE3_PROXY_ALL_VALID_EVAL="${STAGE3_PROXY_ALL_VALID_EVAL:-data/processed/stage3_subtype_rescue_proxy_all_valid_eval.json}"

export REPLAY_RATIO="${REPLAY_RATIO:-0.25}"

STAGE3_BRANCH_TEMPLATE="$(mktemp)"
trap 'rm -f "$STAGE3_BRANCH_TEMPLATE"' EXIT

python - "$STAGE3_BRANCH_TEMPLATE" \
  "$FINAL_STAGE3_ADAPTER_DIR" \
  "$REPAIR_STAGE3_TRAIN_DATASET" \
  "$REPAIR_STAGE3_VALID_DATASET" <<'PY'
from pathlib import Path
import sys
import yaml

output_path = Path(sys.argv[1])
output_dir, train_dataset, valid_dataset = sys.argv[2:5]
payload = yaml.safe_load(
    Path("configs/train_stage3_repair.yaml").read_text(encoding="utf-8")
) or {}
training = payload.setdefault("training", {})
training["output_dir"] = output_dir
training["dataset_path"] = train_dataset
training["eval_path"] = valid_dataset
output_path.write_text(
    yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY

export STAGE3_CONFIG_TEMPLATE="$STAGE3_BRANCH_TEMPLATE"

bash scripts/train_stage3_repair.sh
