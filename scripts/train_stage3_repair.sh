#!/usr/bin/env bash
set -euo pipefail

# Harness-aligned prompt format (chat_thinking) requires the Nemotron tokenizer.
# Populate via `python scripts/probe_chat_template.py` (tokenizer-only mode).

TOKENIZER_PATH="${TOKENIZER_PATH:-artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default}"

STAGE2_ADAPTER_DIR="${STAGE2_ADAPTER_DIR:-artifacts/adapter_stage2_bestproxy}"
STAGE2_INFERENCE_CONFIG="${STAGE2_INFERENCE_CONFIG:-configs/train_stage2_selected_trace.yaml}"
STAGE2_BESTPROXY_HARD_EVAL="${STAGE2_BESTPROXY_HARD_EVAL:-data/processed/stage2_bestproxy_hard_eval.json}"
STAGE2_BESTPROXY_ALL_EVAL="${STAGE2_BESTPROXY_ALL_EVAL:-data/processed/stage2_bestproxy_all_eval.json}"
STAGE3_CONFIG_TEMPLATE="${STAGE3_CONFIG_TEMPLATE:-configs/train_stage3_repair.yaml}"
FINAL_STAGE3_ADAPTER_DIR="${FINAL_STAGE3_ADAPTER_DIR:-artifacts/adapter_stage3_repair}"
STAGE3_BESTPROXY_DIR="${STAGE3_BESTPROXY_DIR:-artifacts/adapter_stage3_bestproxy}"
STAGE3_BESTPROXY_HARD_EVAL="${STAGE3_BESTPROXY_HARD_EVAL:-data/processed/stage3_bestproxy_hard_eval.json}"
STAGE3_BESTPROXY_ALL_EVAL="${STAGE3_BESTPROXY_ALL_EVAL:-data/processed/stage3_bestproxy_all_eval.json}"
STAGE3_BESTPROXY_SELECTION_JSON="${STAGE3_BESTPROXY_SELECTION_JSON:-data/processed/stage3_best_checkpoint_selection.json}"
STAGE3_BESTPROXY_WORKDIR="${STAGE3_BESTPROXY_WORKDIR:-artifacts/_proxy_checkpoint_scratch/stage3_canonical}"

FULL_TRAIN_INPUT="${FULL_TRAIN_INPUT:-data/processed/official_train_tagged.jsonl}"

REPAIR_TRAIN_SUBSET="${REPAIR_TRAIN_SUBSET:-data/processed/stage3_repair_subset_train.jsonl}"
REPLAY_TRAIN_SUBSET="${REPLAY_TRAIN_SUBSET:-data/processed/stage3_replay_pool_train.jsonl}"
VALID_SUBSET="${VALID_SUBSET:-data/processed/stage3_repair_subset_valid.jsonl}"

FULL_TRAIN_PREDICTIONS="${FULL_TRAIN_PREDICTIONS:-data/processed/stage2_full_train_predictions.jsonl}"
VALID_PREDICTIONS="${VALID_PREDICTIONS:-data/processed/stage2_valid_predictions.jsonl}"

FULL_TRAIN_EVAL="${FULL_TRAIN_EVAL:-data/processed/stage2_full_train_eval.json}"
VALID_EVAL="${VALID_EVAL:-data/processed/stage2_model_eval_valid.json}"

TRAIN_FAILURES="${TRAIN_FAILURES:-data/processed/stage2_model_failures_train.json}"
TRAIN_SUCCESSES="${TRAIN_SUCCESSES:-data/processed/stage2_model_successes_all_train.json}"
VALID_FAILURES="${VALID_FAILURES:-data/processed/stage2_model_failures_valid.json}"

STAGE3_TRAIN_REPORT="${STAGE3_TRAIN_REPORT:-data/processed/stage3_repair_train_report.json}"
STAGE3_VALID_REPORT="${STAGE3_VALID_REPORT:-data/processed/stage3_repair_valid_report.json}"
REPAIR_STAGE3_TRAIN_DATASET="${REPAIR_STAGE3_TRAIN_DATASET:-data/processed/stage3_repair_train.jsonl}"
REPAIR_STAGE3_VALID_DATASET="${REPAIR_STAGE3_VALID_DATASET:-data/processed/stage3_repair_valid.jsonl}"

STAGE3_DECISION="${STAGE3_DECISION:-data/processed/stage3_decision.json}"
STAGE3_PROXY_VALID_PREDICTIONS="${STAGE3_PROXY_VALID_PREDICTIONS:-data/processed/stage3_proxy_valid_predictions.jsonl}"
STAGE3_PROXY_VALID_EVAL="${STAGE3_PROXY_VALID_EVAL:-data/processed/stage3_proxy_valid_eval.json}"
ALL_FAMILY_PROXY_VALID_SUBSET="${ALL_FAMILY_PROXY_VALID_SUBSET:-data/processed/proxy_all_family_valid.jsonl}"
STAGE3_PROXY_ALL_VALID_PREDICTIONS="${STAGE3_PROXY_ALL_VALID_PREDICTIONS:-data/processed/stage3_proxy_all_valid_predictions.jsonl}"
STAGE3_PROXY_ALL_VALID_EVAL="${STAGE3_PROXY_ALL_VALID_EVAL:-data/processed/stage3_proxy_all_valid_eval.json}"

REPLAY_RATIO="${REPLAY_RATIO:-0.25}"

if [[ ! -d "$STAGE2_ADAPTER_DIR" ]]; then
  echo "Missing stage2 adapter: $STAGE2_ADAPTER_DIR" >&2
  exit 1
fi

python scripts/prepare_data.py --config configs/data_official.yaml

# Hard-triad train subset -> used later to restrict repair failures.
python scripts/export_split_subset.py \
  --input "$FULL_TRAIN_INPUT" \
  --output "$REPAIR_TRAIN_SUBSET" \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad_rule_novelty \
  --split-role train

# All-family replay pool:
#   rule_novelty_all/train minus hard_triad_rule_novelty/valid
# This keeps easy-triad anchors available for replay without leaking any
# hard-triad valid id into stage3 training.
python scripts/export_split_subset.py \
  --input "$FULL_TRAIN_INPUT" \
  --output "$REPLAY_TRAIN_SUBSET" \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty_all \
  --split-role train \
  --exclude-split-file data/splits/official/splits.json \
  --exclude-split-name hard_triad_rule_novelty \
  --exclude-split-role valid

# Hard-triad valid subset.
python scripts/export_split_subset.py \
  --input "$FULL_TRAIN_INPUT" \
  --output "$VALID_SUBSET" \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad_rule_novelty \
  --split-role valid

# Leak-free all-family proxy subset:
#   rule_novelty_all/valid minus hard_triad_rule_novelty/train
# Same construction as in stage2; producing it again here keeps stage3
# independently runnable.
python scripts/export_split_subset.py \
  --input "$FULL_TRAIN_INPUT" \
  --output "$ALL_FAMILY_PROXY_VALID_SUBSET" \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty_all \
  --split-role valid \
  --exclude-split-file data/splits/official/splits.json \
  --exclude-split-name hard_triad_rule_novelty \
  --exclude-split-role train

# Run stage2 adapter on the FULL official train pool, not only the replay pool.
# The two splits are not nested in split_builder, so a hard-triad train id can
# be absent from rule_novelty_all/train; running over the full pool guarantees
# no hard-triad failure is silently dropped from the stage3 repair artifact.
python -m src.student.inference \
  --config "$STAGE2_INFERENCE_CONFIG" \
  --input "$FULL_TRAIN_INPUT" \
  --adapter-dir "$STAGE2_ADAPTER_DIR" \
  --output "$FULL_TRAIN_PREDICTIONS" \
  --max-new-tokens 2048

python -m src.student.inference \
  --config "$STAGE2_INFERENCE_CONFIG" \
  --input "$VALID_SUBSET" \
  --adapter-dir "$STAGE2_ADAPTER_DIR" \
  --output "$VALID_PREDICTIONS" \
  --max-new-tokens 2048

# Strict coverage gate: any missing / unexpected / duplicate prediction id will
# raise and stop the pipeline instead of silently polluting the repair set.
python -m src.experiments.eval_competition_replica \
  --predictions "$FULL_TRAIN_PREDICTIONS" \
  --labels "$FULL_TRAIN_INPUT" \
  --output "$FULL_TRAIN_EVAL" \
  --require-complete-coverage

python -m src.experiments.eval_competition_replica \
  --predictions "$VALID_PREDICTIONS" \
  --labels "$VALID_SUBSET" \
  --output "$VALID_EVAL" \
  --require-complete-coverage

# Repair failures: only hard-triad train ids.
python scripts/split_eval_artifact.py \
  --input "$FULL_TRAIN_EVAL" \
  --restrict-ids "$REPAIR_TRAIN_SUBSET" \
  --output-failures "$TRAIN_FAILURES"

# Replay successes: all-family train ids minus hard-triad valid (already
# leak-free by construction of REPLAY_TRAIN_SUBSET).
python scripts/split_eval_artifact.py \
  --input "$FULL_TRAIN_EVAL" \
  --restrict-ids "$REPLAY_TRAIN_SUBSET" \
  --output-successes "$TRAIN_SUCCESSES"

# Valid failures: hard-triad valid only.
python scripts/split_eval_artifact.py \
  --input "$VALID_EVAL" \
  --restrict-ids "$VALID_SUBSET" \
  --output-failures "$VALID_FAILURES"

# Stage3 gate: decide whether to skip stage3 entirely and/or disable its eval.
python scripts/decide_stage3_gate.py \
  --train-failures "$TRAIN_FAILURES" \
  --valid-failures "$VALID_FAILURES" \
  --output "$STAGE3_DECISION"

STAGE3_SKIP="$(python - "$STAGE3_DECISION" <<'PY'
from pathlib import Path
import json, sys
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("1" if payload.get("skip_stage3") else "0")
PY
)"

STAGE3_DISABLE_EVAL="$(python - "$STAGE3_DECISION" <<'PY'
from pathlib import Path
import json, sys
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("1" if payload.get("disable_eval_dataset") else "0")
PY
)"

if [[ "$STAGE3_SKIP" != "1" ]]; then
  python -m src.student.sft_dataset_builder \
    --input "$FULL_TRAIN_INPUT" \
    --output "$REPAIR_STAGE3_TRAIN_DATASET" \
    --selection-profile stage3 \
    --prompt-mode chat_thinking \
    --tokenizer-path "$TOKENIZER_PATH" \
    --completion-style short_trace \
    --beam-width 8 \
    --max-depth 2 \
    --top-k 2 \
    --repair-artifact "$TRAIN_FAILURES" \
    --replay-input "$TRAIN_SUCCESSES" \
    --replay-ratio "$REPLAY_RATIO" \
    --report-output "$STAGE3_TRAIN_REPORT"

  if [[ "$STAGE3_DISABLE_EVAL" != "1" ]]; then
    python -m src.student.sft_dataset_builder \
      --input "$VALID_SUBSET" \
      --output "$REPAIR_STAGE3_VALID_DATASET" \
      --selection-profile stage3 \
      --prompt-mode chat_thinking \
      --tokenizer-path "$TOKENIZER_PATH" \
      --completion-style short_trace \
      --beam-width 8 \
      --max-depth 2 \
      --top-k 2 \
      --repair-artifact "$VALID_FAILURES" \
      --report-output "$STAGE3_VALID_REPORT"
  fi
fi

STAGE3_RUNTIME_CONFIG="$(mktemp)"
trap 'rm -f "$STAGE3_RUNTIME_CONFIG"' EXIT

python - "$STAGE3_CONFIG_TEMPLATE" "$STAGE2_ADAPTER_DIR" "$STAGE3_RUNTIME_CONFIG" "$STAGE3_DECISION" <<'PY'
from pathlib import Path
import json
import sys
import yaml

template_path, init_adapter_dir, output_path, decision_path = sys.argv[1:5]
payload = yaml.safe_load(Path(template_path).read_text(encoding="utf-8")) or {}
training = payload.setdefault("training", {})
training["init_adapter_dir"] = init_adapter_dir

decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
if decision.get("disable_eval_dataset", False):
    training.pop("eval_path", None)

Path(output_path).write_text(
    yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY

if [[ "$STAGE3_SKIP" == "1" ]]; then
  # Stage2 was strong enough that no hard-triad train failures remain; reuse
  # the stage2 adapter as the canonical stage3 output so downstream packaging
  # / validation / proxy eval can stay oblivious to the skip. We copy both
  # the weights and the proxy eval artifacts so select_final_adapter.py sees
  # matching (adapter_stage3_bestproxy, stage3_bestproxy_*_eval.json) pairs.
  rm -rf "$FINAL_STAGE3_ADAPTER_DIR"
  mkdir -p "$FINAL_STAGE3_ADAPTER_DIR"
  cp -a "$STAGE2_ADAPTER_DIR"/. "$FINAL_STAGE3_ADAPTER_DIR"/

  rm -rf "$STAGE3_BESTPROXY_DIR"
  mkdir -p "$STAGE3_BESTPROXY_DIR"
  cp -a "$STAGE2_ADAPTER_DIR"/. "$STAGE3_BESTPROXY_DIR"/

  mkdir -p "$(dirname "$STAGE3_BESTPROXY_HARD_EVAL")" \
           "$(dirname "$STAGE3_BESTPROXY_ALL_EVAL")"
  cp "$STAGE2_BESTPROXY_HARD_EVAL" "$STAGE3_BESTPROXY_HARD_EVAL"
  cp "$STAGE2_BESTPROXY_ALL_EVAL"  "$STAGE3_BESTPROXY_ALL_EVAL"

  python - "$STAGE3_DECISION" "$FINAL_STAGE3_ADAPTER_DIR/stage3_skipped.json" "$STAGE2_ADAPTER_DIR" <<'PY'
from pathlib import Path
import json
import sys

decision_path, output_path, reused_adapter_dir = sys.argv[1:4]
payload = json.loads(Path(decision_path).read_text(encoding="utf-8"))
payload["reused_adapter_dir"] = reused_adapter_dir
payload["reason"] = "stage2 produced zero hard-triad train failures; stage3 repair skipped"
Path(output_path).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY

  python - "$STAGE3_BESTPROXY_SELECTION_JSON" "$STAGE2_ADAPTER_DIR" "$STAGE3_BESTPROXY_DIR" <<'PY'
from pathlib import Path
import json
import sys

out_path, reused, final_dir = sys.argv[1:4]
payload = {
    "stage_output_dir": final_dir,
    "selected_candidate": "reused_stage2_bestproxy",
    "selected_adapter_dir": final_dir,
    "reused_adapter_dir": reused,
    "reason": "stage3 repair skipped; stage2 bestproxy adapter reused as stage3 bestproxy",
}
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
Path(out_path).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY
else
  python -m src.student.lora_train --config "$STAGE3_RUNTIME_CONFIG" --force-train

  # Stage3 bestproxy selection: pick the best checkpoint inside the stage3
  # output directory using the same rule as stage2 (all-family primary,
  # hard-triad tiebreak, prefer final on complete tie).
  python scripts/select_best_proxy_checkpoint.py \
    --config "$STAGE3_RUNTIME_CONFIG" \
    --stage-output-dir "$FINAL_STAGE3_ADAPTER_DIR" \
    --hard-proxy-input "$VALID_SUBSET" \
    --all-proxy-input "$ALL_FAMILY_PROXY_VALID_SUBSET" \
    --workdir "$STAGE3_BESTPROXY_WORKDIR" \
    --output-best-dir "$STAGE3_BESTPROXY_DIR" \
    --output-hard-eval "$STAGE3_BESTPROXY_HARD_EVAL" \
    --output-all-eval "$STAGE3_BESTPROXY_ALL_EVAL" \
    --output-json "$STAGE3_BESTPROXY_SELECTION_JSON" \
    --max-new-tokens 2048
fi

# Stage3 proxy eval: evaluate the final adapter (real stage3, or the stage2
# clone in the skip path) against the full hard-triad valid subset. This is
# the competition_correct_rate we trust for comparing against stage2_proxy
# and for deciding whether to submit.
python -m src.student.inference \
  --config "$STAGE3_RUNTIME_CONFIG" \
  --input "$VALID_SUBSET" \
  --adapter-dir "$FINAL_STAGE3_ADAPTER_DIR" \
  --output "$STAGE3_PROXY_VALID_PREDICTIONS" \
  --max-new-tokens 2048

python -m src.experiments.eval_competition_replica \
  --predictions "$STAGE3_PROXY_VALID_PREDICTIONS" \
  --labels "$VALID_SUBSET" \
  --output "$STAGE3_PROXY_VALID_EVAL" \
  --require-complete-coverage

python - "$STAGE3_PROXY_VALID_EVAL" <<'PY'
from pathlib import Path
import json, sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(
    {
        "proxy_name": "stage3_hard_triad",
        "competition_correct_rate": payload.get("competition_correct_rate"),
        "num_examples": payload.get("num_examples"),
        "coverage": payload.get("coverage", {}),
    },
    ensure_ascii=False,
    indent=2,
))
PY

# Stage3 all-family proxy eval: feeds into select_final_adapter as the
# primary metric. A stage3 adapter with better hard-triad proxy but worse
# all-family proxy is a sign that easy-triad retention was sacrificed; the
# selector decides from this artifact whether stage3 actually beats stage2.
python -m src.student.inference \
  --config "$STAGE3_RUNTIME_CONFIG" \
  --input "$ALL_FAMILY_PROXY_VALID_SUBSET" \
  --adapter-dir "$FINAL_STAGE3_ADAPTER_DIR" \
  --output "$STAGE3_PROXY_ALL_VALID_PREDICTIONS" \
  --max-new-tokens 2048

python -m src.experiments.eval_competition_replica \
  --predictions "$STAGE3_PROXY_ALL_VALID_PREDICTIONS" \
  --labels "$ALL_FAMILY_PROXY_VALID_SUBSET" \
  --output "$STAGE3_PROXY_ALL_VALID_EVAL" \
  --require-complete-coverage

python - "$STAGE3_PROXY_ALL_VALID_EVAL" <<'PY'
from pathlib import Path
import json, sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(
    {
        "proxy_name": "stage3_all_family",
        "competition_correct_rate": payload.get("competition_correct_rate"),
        "num_examples": payload.get("num_examples"),
        "coverage": payload.get("coverage", {}),
    },
    ensure_ascii=False,
    indent=2,
))
PY
