#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Stage2 subtype-prior rescue (v1) — experimental branch.
#
# This is a sibling of scripts/train_stage2_distill.sh. The ONLY experimental
# variable vs. canonical stage2 is:
#
#   --stage2-use-search-subtype-hint
#
# passed to the train-side builder. Everything else — prompt mode, tokenizer,
# LoRA hparams, schedule, rescue beam/depth/top-k, rescue-family scope
# (equation only), silver pool, oversample factor, and the two-proxy
# evaluation — intentionally mirrors canonical stage2 so any delta on the
# hard-triad proxy is attributable to the subtype hint alone.
#
# Outputs live under isolated paths (adapter_stage2_subtype_rescue*,
# stage2_subtype_rescue_*.jsonl/.json) so the canonical pipeline is never
# overwritten and the two runs can be compared head-to-head.
# ============================================================================

# Harness-aligned prompt format (chat_thinking) requires the Nemotron tokenizer.
# Populate via `python scripts/probe_chat_template.py` (tokenizer-only mode).

TOKENIZER_PATH="${TOKENIZER_PATH:-artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default}"
STAGE1_ADAPTER_DIR="${STAGE1_ADAPTER_DIR:-artifacts/adapter_stage1_format}"
STAGE2_CONFIG_TEMPLATE="${STAGE2_CONFIG_TEMPLATE:-configs/train_stage2_selected_trace_subtype_rescue.yaml}"

# Isolated report / dataset / adapter paths so this branch never collides
# with canonical stage2 outputs.
STAGE2_REPORT="${STAGE2_REPORT:-data/processed/stage2_subtype_rescue_report.json}"
STAGE2_VALID_REPORT="${STAGE2_VALID_REPORT:-data/processed/stage2_subtype_rescue_valid_report.json}"

STAGE2_TRAIN_DATASET="${STAGE2_TRAIN_DATASET:-data/processed/stage2_subtype_rescue_train.jsonl}"
STAGE2_VALID_DATASET="${STAGE2_VALID_DATASET:-data/processed/stage2_subtype_rescue_valid.jsonl}"

STAGE2_TRAIN_OFFICIAL_SUBSET="${STAGE2_TRAIN_OFFICIAL_SUBSET:-data/processed/stage2_official_train_no_hard_valid.jsonl}"
STAGE2_VALID_OFFICIAL_SUBSET="${STAGE2_VALID_OFFICIAL_SUBSET:-data/processed/stage2_official_valid_hard_triad.jsonl}"
ALL_FAMILY_PROXY_VALID_SUBSET="${ALL_FAMILY_PROXY_VALID_SUBSET:-data/processed/proxy_all_family_valid.jsonl}"

STAGE2_ADAPTER_DIR="${STAGE2_ADAPTER_DIR:-artifacts/adapter_stage2_subtype_rescue}"
STAGE2_PROXY_VALID_PREDICTIONS="${STAGE2_PROXY_VALID_PREDICTIONS:-data/processed/stage2_subtype_rescue_proxy_valid_predictions.jsonl}"
STAGE2_PROXY_VALID_EVAL="${STAGE2_PROXY_VALID_EVAL:-data/processed/stage2_subtype_rescue_proxy_valid_eval.json}"
STAGE2_PROXY_ALL_VALID_PREDICTIONS="${STAGE2_PROXY_ALL_VALID_PREDICTIONS:-data/processed/stage2_subtype_rescue_proxy_all_valid_predictions.jsonl}"
STAGE2_PROXY_ALL_VALID_EVAL="${STAGE2_PROXY_ALL_VALID_EVAL:-data/processed/stage2_subtype_rescue_proxy_all_valid_eval.json}"

STAGE2_BESTPROXY_DIR="${STAGE2_BESTPROXY_DIR:-artifacts/adapter_stage2_subtype_rescue_bestproxy}"
STAGE2_BESTPROXY_HARD_EVAL="${STAGE2_BESTPROXY_HARD_EVAL:-data/processed/stage2_subtype_rescue_bestproxy_hard_eval.json}"
STAGE2_BESTPROXY_ALL_EVAL="${STAGE2_BESTPROXY_ALL_EVAL:-data/processed/stage2_subtype_rescue_bestproxy_all_eval.json}"
STAGE2_BESTPROXY_SELECTION_JSON="${STAGE2_BESTPROXY_SELECTION_JSON:-data/processed/stage2_subtype_rescue_best_checkpoint_selection.json}"

if [[ ! -d "$STAGE1_ADAPTER_DIR" ]]; then
  echo "Missing stage1 adapter: $STAGE1_ADAPTER_DIR" >&2
  exit 1
fi

# The canonical prepare_data + synth_hard_triads outputs are shared between
# canonical stage2 and this branch; no reason to regenerate them here unless
# they are missing.
if [[ ! -f "data/processed/official_train_tagged.jsonl" ]]; then
  python scripts/prepare_data.py --config configs/data_official.yaml
fi
if [[ ! -f "data/synthetic/synth_hard_triads.jsonl" ]]; then
  python -m src.teacher.synth_generator --config configs/synth_hard_triads.yaml
fi

# The three split subsets below mirror canonical stage2 exactly; regenerate
# only when missing so sibling runs share a stable definition of train / valid.
if [[ ! -f "$STAGE2_TRAIN_OFFICIAL_SUBSET" ]]; then
  python scripts/export_split_subset.py \
    --input data/processed/official_train_tagged.jsonl \
    --output "$STAGE2_TRAIN_OFFICIAL_SUBSET" \
    --split-file data/splits/official/splits.json \
    --split-name rule_novelty_all \
    --split-role train \
    --exclude-split-file data/splits/official/splits.json \
    --exclude-split-name hard_triad_rule_novelty \
    --exclude-split-role valid
fi

if [[ ! -f "$STAGE2_VALID_OFFICIAL_SUBSET" ]]; then
  python scripts/export_split_subset.py \
    --input data/processed/official_train_tagged.jsonl \
    --output "$STAGE2_VALID_OFFICIAL_SUBSET" \
    --split-file data/splits/official/splits.json \
    --split-name hard_triad_rule_novelty \
    --split-role valid
fi

if [[ ! -f "$ALL_FAMILY_PROXY_VALID_SUBSET" ]]; then
  python scripts/export_split_subset.py \
    --input data/processed/official_train_tagged.jsonl \
    --output "$ALL_FAMILY_PROXY_VALID_SUBSET" \
    --split-file data/splits/official/splits.json \
    --split-name rule_novelty_all \
    --split-role valid \
    --exclude-split-file data/splits/official/splits.json \
    --exclude-split-name hard_triad_rule_novelty \
    --exclude-split-role train
fi

# ---------------------------------------------------------------------------
# Build the stage2 subtype-rescue TRAIN dataset.
#
# All canonical stage2 switches (silver pool, rescue, oversample, balance)
# stay identical so the ONLY experimental delta is
# --stage2-use-search-subtype-hint. Rescue scope is still equation-only to
# match canonical stage2; extending to cipher happens in a subsequent v2 run,
# gated on v1 beating the baseline on the hard-triad proxy.
# ---------------------------------------------------------------------------
python -m src.student.sft_dataset_builder \
  --input "$STAGE2_TRAIN_OFFICIAL_SUBSET,data/synthetic/synth_hard_triads.jsonl" \
  --output "$STAGE2_TRAIN_DATASET" \
  --selection-profile stage2 \
  --prompt-mode chat_thinking \
  --tokenizer-path "$TOKENIZER_PATH" \
  --completion-style token_trace \
  --beam-width 10 \
  --max-depth 3 \
  --top-k 3 \
  --oversample-hard-triad \
  --balance-by-family \
  --hard-triad-repeat-factor 2 \
  --max-per-signature-bucket 64 \
  --stage2-enable-silver-official \
  --stage2-silver-hard-confidence 0.65 \
  --stage2-silver-hard-support 0.67 \
  --stage2-silver-max-fraction 0.25 \
  --stage2-silver-max-absolute 800 \
  --stage2-second-pass-hard-triad \
  --stage2-rescue-families equation \
  --stage2-second-pass-beam-width 12 \
  --stage2-second-pass-max-depth 4 \
  --stage2-second-pass-top-k 3 \
  --stage2-use-search-subtype-hint \
  --report-output "$STAGE2_REPORT"

# Stage2 valid must stay canonical: no family scheduling, no hard-triad
# duplication, no per-signature cap, NO subtype hint, NO silver, NO rescue.
# Applying any of those transforms to valid would reshape the evaluation
# distribution and break the A/B comparison with the canonical baseline.
python -m src.student.sft_dataset_builder \
  --input "$STAGE2_VALID_OFFICIAL_SUBSET" \
  --output "$STAGE2_VALID_DATASET" \
  --selection-profile stage2 \
  --prompt-mode chat_thinking \
  --tokenizer-path "$TOKENIZER_PATH" \
  --completion-style token_trace \
  --beam-width 10 \
  --max-depth 3 \
  --top-k 3 \
  --no-balance-by-family \
  --hard-triad-repeat-factor 1 \
  --max-per-signature-bucket 0 \
  --report-output "$STAGE2_VALID_REPORT"

# Fail fast: oversample must have materialised on train, and valid must stay
# unweighted. These mirror the canonical stage2 gates; failing here means the
# experimental builder drifted from canonical conventions.
python - "$STAGE2_REPORT" <<'PY'
from pathlib import Path
import json
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("num_records", 0) <= 0:
    raise SystemExit("stage2 subtype-rescue train dataset is empty")
if report.get("duplication_ratio", 0.0) <= 0.0:
    raise SystemExit(
        "stage2 subtype-rescue train report shows duplication_ratio <= 0.0; "
        "hard-triad oversampling did not materialise. "
        "Check that --oversample-hard-triad is passed and hard_triad_repeat_factor > 1."
    )
hint_diag = report.get("subtype_hint_diagnostics") or {}
if not hint_diag.get("enabled"):
    raise SystemExit(
        "stage2 subtype-rescue train report shows subtype_hint_diagnostics.enabled=false; "
        "the experimental hint path did not activate. "
        "Check that --stage2-use-search-subtype-hint is passed."
    )
PY

python - "$STAGE2_VALID_REPORT" <<'PY'
from pathlib import Path
import json
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("num_records", 0) <= 0:
    raise SystemExit("stage2 subtype-rescue valid dataset is empty")
duplication = report.get("duplication_ratio", 0.0)
if duplication > 0.0:
    raise SystemExit(
        f"stage2 subtype-rescue valid dataset should be unweighted, got duplication_ratio={duplication}"
    )
hint_diag = report.get("subtype_hint_diagnostics") or {}
if hint_diag.get("enabled"):
    raise SystemExit(
        "stage2 subtype-rescue valid report shows subtype_hint_diagnostics.enabled=true; "
        "valid must stay canonical so the experimental override only applies to train."
    )
PY

STAGE2_RUNTIME_CONFIG="$(mktemp)"
trap 'rm -f "$STAGE2_RUNTIME_CONFIG"' EXIT

python - "$STAGE2_CONFIG_TEMPLATE" "$STAGE1_ADAPTER_DIR" "$STAGE2_RUNTIME_CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml

template_path, init_adapter_dir, output_path = sys.argv[1:4]
payload = yaml.safe_load(Path(template_path).read_text(encoding="utf-8")) or {}
payload.setdefault("training", {})["init_adapter_dir"] = init_adapter_dir
Path(output_path).write_text(
    yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY

python -m src.student.lora_train --config "$STAGE2_RUNTIME_CONFIG" --force-train

# Stage2 subtype-rescue proxy eval: run the trained adapter over the full
# hard-triad valid subset to get the real competition_correct_rate. This is
# the headline metric to compare against the canonical stage2 baseline
# (data/processed/stage2_bestproxy_hard_eval.json).
python -m src.student.inference \
  --config "$STAGE2_RUNTIME_CONFIG" \
  --input "$STAGE2_VALID_OFFICIAL_SUBSET" \
  --adapter-dir "$STAGE2_ADAPTER_DIR" \
  --output "$STAGE2_PROXY_VALID_PREDICTIONS" \
  --max-new-tokens 2048

python -m src.experiments.eval_competition_replica \
  --predictions "$STAGE2_PROXY_VALID_PREDICTIONS" \
  --labels "$STAGE2_VALID_OFFICIAL_SUBSET" \
  --output "$STAGE2_PROXY_VALID_EVAL" \
  --require-complete-coverage

python - "$STAGE2_PROXY_VALID_EVAL" <<'PY'
from pathlib import Path
import json, sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(
    {
        "proxy_name": "stage2_subtype_rescue_hard_triad",
        "competition_correct_rate": payload.get("competition_correct_rate"),
        "num_examples": payload.get("num_examples"),
        "coverage": payload.get("coverage", {}),
    },
    ensure_ascii=False,
    indent=2,
))
PY

# All-family proxy: the regression guard. The branch must not significantly
# degrade easy-triad anchors; scripts/select_final_adapter.py reads both
# metrics with a tolerance, and this branch is only promoted to stage3 when
# all-family stays close to baseline AND hard-triad wins.
python -m src.student.inference \
  --config "$STAGE2_RUNTIME_CONFIG" \
  --input "$ALL_FAMILY_PROXY_VALID_SUBSET" \
  --adapter-dir "$STAGE2_ADAPTER_DIR" \
  --output "$STAGE2_PROXY_ALL_VALID_PREDICTIONS" \
  --max-new-tokens 2048

python -m src.experiments.eval_competition_replica \
  --predictions "$STAGE2_PROXY_ALL_VALID_PREDICTIONS" \
  --labels "$ALL_FAMILY_PROXY_VALID_SUBSET" \
  --output "$STAGE2_PROXY_ALL_VALID_EVAL" \
  --require-complete-coverage

python - "$STAGE2_PROXY_ALL_VALID_EVAL" <<'PY'
from pathlib import Path
import json, sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(
    {
        "proxy_name": "stage2_subtype_rescue_all_family",
        "competition_correct_rate": payload.get("competition_correct_rate"),
        "num_examples": payload.get("num_examples"),
        "coverage": payload.get("coverage", {}),
    },
    ensure_ascii=False,
    indent=2,
))
PY

# Bestproxy selection: evaluate every saved checkpoint (and the final
# adapter) against the hard-triad + all-family proxies, then copy the
# winner to the isolated stage2_subtype_rescue_bestproxy path.
python scripts/select_best_proxy_checkpoint.py \
  --config "$STAGE2_RUNTIME_CONFIG" \
  --stage-output-dir "$STAGE2_ADAPTER_DIR" \
  --hard-proxy-input "$STAGE2_VALID_OFFICIAL_SUBSET" \
  --all-proxy-input "$ALL_FAMILY_PROXY_VALID_SUBSET" \
  --output-best-dir "$STAGE2_BESTPROXY_DIR" \
  --output-hard-eval "$STAGE2_BESTPROXY_HARD_EVAL" \
  --output-all-eval "$STAGE2_BESTPROXY_ALL_EVAL" \
  --output-json "$STAGE2_BESTPROXY_SELECTION_JSON" \
  --max-new-tokens 2048

python - "$STAGE2_BESTPROXY_SELECTION_JSON" "$STAGE2_BESTPROXY_HARD_EVAL" "$STAGE2_BESTPROXY_ALL_EVAL" <<'PY'
from pathlib import Path
import json, sys

selection = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
hard = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
all_family = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
print(json.dumps(
    {
        "branch": "stage2_subtype_rescue_v1",
        "selected_candidate": selection.get("selected_candidate"),
        "hard_triad_correct_rate": hard.get("competition_correct_rate"),
        "all_family_correct_rate": all_family.get("competition_correct_rate"),
        "promotion_note": (
            "Compare against canonical stage2_bestproxy_*_eval.json. "
            "Promote to stage3 only when all-family stays within tolerance "
            "AND hard-triad improves by at least one sample resolution."
        ),
    },
    ensure_ascii=False,
    indent=2,
))
PY
