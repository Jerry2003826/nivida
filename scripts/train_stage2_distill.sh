#!/usr/bin/env bash
set -euo pipefail

# Harness-aligned prompt format (chat_thinking) requires the Nemotron tokenizer.
# Populate via `python scripts/probe_chat_template.py` (tokenizer-only mode).

TOKENIZER_PATH="${TOKENIZER_PATH:-artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default}"
STAGE1_ADAPTER_DIR="${STAGE1_ADAPTER_DIR:-artifacts/adapter_stage1_format}"
STAGE2_CONFIG_TEMPLATE="${STAGE2_CONFIG_TEMPLATE:-configs/train_stage2_selected_trace.yaml}"
STAGE2_REPORT="${STAGE2_REPORT:-data/processed/stage2_distill_report.json}"
STAGE2_VALID_REPORT="${STAGE2_VALID_REPORT:-data/processed/stage2_distill_valid_report.json}"

STAGE2_TRAIN_OFFICIAL_SUBSET="${STAGE2_TRAIN_OFFICIAL_SUBSET:-data/processed/stage2_official_train_no_hard_valid.jsonl}"
STAGE2_VALID_OFFICIAL_SUBSET="${STAGE2_VALID_OFFICIAL_SUBSET:-data/processed/stage2_official_valid_hard_triad.jsonl}"

if [[ ! -d "$STAGE1_ADAPTER_DIR" ]]; then
  echo "Missing stage1 adapter: $STAGE1_ADAPTER_DIR" >&2
  exit 1
fi

python scripts/prepare_data.py --config configs/data_official.yaml
python -m src.teacher.synth_generator --config configs/synth_hard_triads.yaml

# Leak-free official train subset:
#   rule_novelty_all/train minus hard_triad_rule_novelty/valid
# The two splits are built independently in split_builder (different seeds),
# so the exclude ensures no hard-triad valid id is ever trained on.
python scripts/export_split_subset.py \
  --input data/processed/official_train_tagged.jsonl \
  --output "$STAGE2_TRAIN_OFFICIAL_SUBSET" \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty_all \
  --split-role train \
  --exclude-split-file data/splits/official/splits.json \
  --exclude-split-name hard_triad_rule_novelty \
  --exclude-split-role valid

# Hard-triad valid subset for evaluation.
python scripts/export_split_subset.py \
  --input data/processed/official_train_tagged.jsonl \
  --output "$STAGE2_VALID_OFFICIAL_SUBSET" \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad_rule_novelty \
  --split-role valid

python -m src.student.sft_dataset_builder \
  --input "$STAGE2_TRAIN_OFFICIAL_SUBSET,data/synthetic/synth_hard_triads.jsonl" \
  --output data/processed/stage2_distill_train.jsonl \
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
  --report-output "$STAGE2_REPORT"

# Stage2 valid must stay unweighted: no family scheduling, no hard-triad
# duplication, no per-signature cap. All those transforms belong to train
# only; applying them to valid would reshape the evaluation distribution.
python -m src.student.sft_dataset_builder \
  --input "$STAGE2_VALID_OFFICIAL_SUBSET" \
  --output data/processed/stage2_distill_valid.jsonl \
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

# Fail fast if the reports show that oversampling did not materialise on
# train, or that valid accidentally ended up with duplicated records.
python - "$STAGE2_REPORT" <<'PY'
from pathlib import Path
import json
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("num_records", 0) <= 0:
    raise SystemExit("stage2 train dataset is empty")
if report.get("duplication_ratio", 0.0) <= 0.0:
    raise SystemExit(
        "stage2 train report shows duplication_ratio <= 0.0; "
        "hard-triad oversampling did not materialise. "
        "Check that --oversample-hard-triad is passed and hard_triad_repeat_factor > 1."
    )
PY

python - "$STAGE2_VALID_REPORT" <<'PY'
from pathlib import Path
import json
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("num_records", 0) <= 0:
    raise SystemExit("stage2 valid dataset is empty")
duplication = report.get("duplication_ratio", 0.0)
if duplication > 0.0:
    raise SystemExit(
        f"stage2 valid dataset should be unweighted, got duplication_ratio={duplication}"
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
