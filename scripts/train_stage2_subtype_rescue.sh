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
REFRESH_SUBTYPE_RESCUE_INPUTS="${REFRESH_SUBTYPE_RESCUE_INPUTS:-0}"
ALLOW_SUBTYPE_RESCUE_REGENERATE_INPUTS="${ALLOW_SUBTYPE_RESCUE_REGENERATE_INPUTS:-0}"

# Isolated report / dataset / adapter paths so this branch never collides
# with canonical stage2 outputs.
STAGE2_REPORT="${STAGE2_REPORT:-data/processed/stage2_subtype_rescue_report.json}"
STAGE2_VALID_REPORT="${STAGE2_VALID_REPORT:-data/processed/stage2_subtype_rescue_valid_report.json}"
STAGE2_INPUT_MANIFEST="${STAGE2_INPUT_MANIFEST:-data/processed/stage2_subtype_rescue_input_manifest.json}"
STAGE2_SKIPPED_ARTIFACT="${STAGE2_SKIPPED_ARTIFACT:-data/processed/stage2_subtype_rescue_skipped.json}"

STAGE2_TRAIN_DATASET="${STAGE2_TRAIN_DATASET:-data/processed/stage2_subtype_rescue_train.jsonl}"
STAGE2_VALID_DATASET="${STAGE2_VALID_DATASET:-data/processed/stage2_subtype_rescue_valid.jsonl}"

CANONICAL_STAGE2_TRAIN_OFFICIAL_SUBSET="${CANONICAL_STAGE2_TRAIN_OFFICIAL_SUBSET:-data/processed/stage2_official_train_no_hard_valid.jsonl}"
CANONICAL_STAGE2_VALID_OFFICIAL_SUBSET="${CANONICAL_STAGE2_VALID_OFFICIAL_SUBSET:-data/processed/stage2_official_valid_hard_triad.jsonl}"
CANONICAL_ALL_FAMILY_PROXY_VALID_SUBSET="${CANONICAL_ALL_FAMILY_PROXY_VALID_SUBSET:-data/processed/proxy_all_family_valid.jsonl}"
CANONICAL_SYNTH_HARD_TRIADS_PATH="${CANONICAL_SYNTH_HARD_TRIADS_PATH:-data/synthetic/synth_hard_triads.jsonl}"
CANONICAL_SYNTH_HARD_TRIADS_SUMMARY_PATH="${CANONICAL_SYNTH_HARD_TRIADS_SUMMARY_PATH:-data/synthetic/synth_hard_triads_summary.json}"

STAGE2_TRAIN_OFFICIAL_SUBSET="${STAGE2_TRAIN_OFFICIAL_SUBSET:-data/processed/stage2_subtype_rescue_official_train_no_hard_valid.jsonl}"
STAGE2_VALID_OFFICIAL_SUBSET="${STAGE2_VALID_OFFICIAL_SUBSET:-data/processed/stage2_subtype_rescue_official_valid_hard_triad.jsonl}"
ALL_FAMILY_PROXY_VALID_SUBSET="${ALL_FAMILY_PROXY_VALID_SUBSET:-data/processed/stage2_subtype_rescue_proxy_all_family_valid.jsonl}"
SYNTH_HARD_TRIADS_PATH="${SYNTH_HARD_TRIADS_PATH:-data/synthetic/synth_hard_triads_subtype_rescue.jsonl}"
SYNTH_HARD_TRIADS_SUMMARY_PATH="${SYNTH_HARD_TRIADS_SUMMARY_PATH:-data/synthetic/synth_hard_triads_subtype_rescue_summary.json}"

STAGE2_ADAPTER_DIR="${STAGE2_ADAPTER_DIR:-artifacts/adapter_stage2_subtype_rescue}"
STAGE2_PROXY_VALID_PREDICTIONS="${STAGE2_PROXY_VALID_PREDICTIONS:-data/processed/stage2_subtype_rescue_proxy_valid_predictions.jsonl}"
STAGE2_PROXY_VALID_EVAL="${STAGE2_PROXY_VALID_EVAL:-data/processed/stage2_subtype_rescue_proxy_valid_eval.json}"
STAGE2_PROXY_ALL_VALID_PREDICTIONS="${STAGE2_PROXY_ALL_VALID_PREDICTIONS:-data/processed/stage2_subtype_rescue_proxy_all_valid_predictions.jsonl}"
STAGE2_PROXY_ALL_VALID_EVAL="${STAGE2_PROXY_ALL_VALID_EVAL:-data/processed/stage2_subtype_rescue_proxy_all_valid_eval.json}"

STAGE2_BESTPROXY_DIR="${STAGE2_BESTPROXY_DIR:-artifacts/adapter_stage2_subtype_rescue_bestproxy}"
STAGE2_BESTPROXY_HARD_EVAL="${STAGE2_BESTPROXY_HARD_EVAL:-data/processed/stage2_subtype_rescue_bestproxy_hard_eval.json}"
STAGE2_BESTPROXY_ALL_EVAL="${STAGE2_BESTPROXY_ALL_EVAL:-data/processed/stage2_subtype_rescue_bestproxy_all_eval.json}"
STAGE2_BESTPROXY_SELECTION_JSON="${STAGE2_BESTPROXY_SELECTION_JSON:-data/processed/stage2_subtype_rescue_best_checkpoint_selection.json}"
STAGE2_BESTPROXY_WORKDIR="${STAGE2_BESTPROXY_WORKDIR:-artifacts/_proxy_checkpoint_scratch/stage2_subtype_rescue}"

if [[ ! -d "$STAGE1_ADAPTER_DIR" ]]; then
  echo "Missing stage1 adapter: $STAGE1_ADAPTER_DIR" >&2
  exit 1
fi

copy_into_branch_path() {
  local source_path="$1"
  local target_path="$2"
  mkdir -p "$(dirname "$target_path")"
  local target_tmp
  target_tmp="$(mktemp "${target_path}.tmp.XXXXXX")"
  cp "$source_path" "$target_tmp"
  mv "$target_tmp" "$target_path"
}

require_canonical_input_or_override() {
  local label="$1"
  local canonical_path="$2"
  if [[ ! -f "$canonical_path" && "$ALLOW_SUBTYPE_RESCUE_REGENERATE_INPUTS" != "1" ]]; then
    echo "Missing canonical $label: $canonical_path" >&2
    echo "Run canonical stage2 first, or set ALLOW_SUBTYPE_RESCUE_REGENERATE_INPUTS=1 for a deliberate standalone experiment." >&2
    exit 1
  fi
}

assert_branch_local_matches_canonical() {
  local canonical_path="$1"
  local branch_path="$2"
  local label="$3"
  python - "$canonical_path" "$branch_path" "$label" <<'PY'
from pathlib import Path
import hashlib
import sys

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

canonical_path, branch_path, label = sys.argv[1:4]
canonical = Path(canonical_path)
branch = Path(branch_path)
if sha256(canonical) != sha256(branch):
    raise SystemExit(
        f"{label} differs from canonical source: {branch}. "
        "Set REFRESH_SUBTYPE_RESCUE_INPUTS=1 to recopy/regenerate deliberately."
    )
PY
}

write_input_manifest() {
  python - "$STAGE2_INPUT_MANIFEST" \
    "$SYNTH_HARD_TRIADS_PATH" "$SYNTH_HARD_TRIADS_SOURCE_TYPE" "$SYNTH_HARD_TRIADS_SOURCE_PATH" \
    "$SYNTH_HARD_TRIADS_SUMMARY_PATH" "$SYNTH_HARD_TRIADS_SUMMARY_SOURCE_TYPE" "$SYNTH_HARD_TRIADS_SUMMARY_SOURCE_PATH" \
    "$STAGE2_TRAIN_OFFICIAL_SUBSET" "$STAGE2_TRAIN_OFFICIAL_SUBSET_SOURCE_TYPE" "$STAGE2_TRAIN_OFFICIAL_SUBSET_SOURCE_PATH" \
    "$STAGE2_VALID_OFFICIAL_SUBSET" "$STAGE2_VALID_OFFICIAL_SUBSET_SOURCE_TYPE" "$STAGE2_VALID_OFFICIAL_SUBSET_SOURCE_PATH" \
    "$ALL_FAMILY_PROXY_VALID_SUBSET" "$ALL_FAMILY_PROXY_VALID_SUBSET_SOURCE_TYPE" "$ALL_FAMILY_PROXY_VALID_SUBSET_SOURCE_PATH" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

output = Path(sys.argv[1])
triples = sys.argv[2:]
if len(triples) % 3 != 0:
    raise SystemExit("write_input_manifest expected path/source_type/source_path triples")

files: dict[str, dict[str, object]] = {}
for index in range(0, len(triples), 3):
    path = Path(triples[index])
    source_type = triples[index + 1]
    source_path = triples[index + 2]
    files[str(path)] = {
        "source_type": source_type,
        "source_path": source_path,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }

payload = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "files": files,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

# Materialize branch-local inputs. By default, reuse canonical stabilized inputs
# by copying them into subtype-rescue-specific paths; only regenerate when the
# canonical artifact is absent or when REFRESH_SUBTYPE_RESCUE_INPUTS=1 requests
# a deliberate branch-local refresh.
if [[ ! -f "data/processed/official_train_tagged.jsonl" ]]; then
  python scripts/prepare_data.py --config configs/data_official.yaml
fi

require_canonical_input_or_override "stage2 train subset" "$CANONICAL_STAGE2_TRAIN_OFFICIAL_SUBSET"
require_canonical_input_or_override "stage2 valid subset" "$CANONICAL_STAGE2_VALID_OFFICIAL_SUBSET"
require_canonical_input_or_override "stage2 all-family proxy subset" "$CANONICAL_ALL_FAMILY_PROXY_VALID_SUBSET"
require_canonical_input_or_override "hard-triad synth input" "$CANONICAL_SYNTH_HARD_TRIADS_PATH"
require_canonical_input_or_override "hard-triad synth summary" "$CANONICAL_SYNTH_HARD_TRIADS_SUMMARY_PATH"

if [[ "$REFRESH_SUBTYPE_RESCUE_INPUTS" != "1" && -f "$SYNTH_HARD_TRIADS_PATH" ]]; then
  if [[ -f "$CANONICAL_SYNTH_HARD_TRIADS_PATH" && -f "$CANONICAL_SYNTH_HARD_TRIADS_SUMMARY_PATH" ]]; then
    assert_branch_local_matches_canonical "$CANONICAL_SYNTH_HARD_TRIADS_PATH" "$SYNTH_HARD_TRIADS_PATH" "branch-local synth input"
    assert_branch_local_matches_canonical "$CANONICAL_SYNTH_HARD_TRIADS_SUMMARY_PATH" "$SYNTH_HARD_TRIADS_SUMMARY_PATH" "branch-local synth summary"
  fi
  SYNTH_HARD_TRIADS_SOURCE_TYPE="reused_branch_local"
  SYNTH_HARD_TRIADS_SOURCE_PATH="$SYNTH_HARD_TRIADS_PATH"
  SYNTH_HARD_TRIADS_SUMMARY_SOURCE_TYPE="reused_branch_local"
  SYNTH_HARD_TRIADS_SUMMARY_SOURCE_PATH="$SYNTH_HARD_TRIADS_SUMMARY_PATH"
  echo "[stage2-subtype] Reusing existing branch-local input: $SYNTH_HARD_TRIADS_PATH"
elif [[ "$REFRESH_SUBTYPE_RESCUE_INPUTS" != "1" && -f "$CANONICAL_SYNTH_HARD_TRIADS_PATH" && -f "$CANONICAL_SYNTH_HARD_TRIADS_SUMMARY_PATH" ]]; then
  copy_into_branch_path "$CANONICAL_SYNTH_HARD_TRIADS_PATH" "$SYNTH_HARD_TRIADS_PATH"
  copy_into_branch_path "$CANONICAL_SYNTH_HARD_TRIADS_SUMMARY_PATH" "$SYNTH_HARD_TRIADS_SUMMARY_PATH"
  SYNTH_HARD_TRIADS_SOURCE_TYPE="copied_from_canonical"
  SYNTH_HARD_TRIADS_SOURCE_PATH="$CANONICAL_SYNTH_HARD_TRIADS_PATH"
  SYNTH_HARD_TRIADS_SUMMARY_SOURCE_TYPE="copied_from_canonical"
  SYNTH_HARD_TRIADS_SUMMARY_SOURCE_PATH="$CANONICAL_SYNTH_HARD_TRIADS_SUMMARY_PATH"
  echo "[stage2-subtype] Copied canonical synth inputs into branch-local paths."
else
  mkdir -p "$(dirname "$SYNTH_HARD_TRIADS_PATH")" "$(dirname "$SYNTH_HARD_TRIADS_SUMMARY_PATH")"
  synth_tmp_output="$(mktemp "${SYNTH_HARD_TRIADS_PATH}.tmp.XXXXXX")"
  synth_tmp_summary="$(mktemp "${SYNTH_HARD_TRIADS_SUMMARY_PATH}.tmp.XXXXXX")"
  synth_tmp_config="$(mktemp "${TMPDIR:-/tmp}/synth_hard_triads.XXXXXX.yaml")"
  python - "configs/synth_hard_triads.yaml" "$synth_tmp_config" "$synth_tmp_output" "$synth_tmp_summary" <<'PY'
from pathlib import Path
import sys
import yaml

template_path, config_path, output_path, summary_path = sys.argv[1:5]
payload = yaml.safe_load(Path(template_path).read_text(encoding="utf-8")) or {}
payload["output_path"] = output_path
payload["summary_path"] = summary_path
Path(config_path).write_text(
    yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY
  python -m src.teacher.synth_generator --config "$synth_tmp_config"
  mv "$synth_tmp_output" "$SYNTH_HARD_TRIADS_PATH"
  mv "$synth_tmp_summary" "$SYNTH_HARD_TRIADS_SUMMARY_PATH"
  rm -f "$synth_tmp_config"
  SYNTH_HARD_TRIADS_SOURCE_TYPE="regenerated_branch_local"
  SYNTH_HARD_TRIADS_SOURCE_PATH="configs/synth_hard_triads.yaml"
  SYNTH_HARD_TRIADS_SUMMARY_SOURCE_TYPE="regenerated_branch_local"
  SYNTH_HARD_TRIADS_SUMMARY_SOURCE_PATH="configs/synth_hard_triads.yaml"
fi

# The three split subsets below mirror canonical stage2 exactly, but they live
# under branch-local names so subtype-rescue never overwrites canonical inputs.
if [[ "$REFRESH_SUBTYPE_RESCUE_INPUTS" != "1" && -f "$STAGE2_TRAIN_OFFICIAL_SUBSET" ]]; then
  if [[ -f "$CANONICAL_STAGE2_TRAIN_OFFICIAL_SUBSET" ]]; then
    assert_branch_local_matches_canonical "$CANONICAL_STAGE2_TRAIN_OFFICIAL_SUBSET" "$STAGE2_TRAIN_OFFICIAL_SUBSET" "branch-local train subset"
  fi
  STAGE2_TRAIN_OFFICIAL_SUBSET_SOURCE_TYPE="reused_branch_local"
  STAGE2_TRAIN_OFFICIAL_SUBSET_SOURCE_PATH="$STAGE2_TRAIN_OFFICIAL_SUBSET"
  echo "[stage2-subtype] Reusing existing branch-local input: $STAGE2_TRAIN_OFFICIAL_SUBSET"
elif [[ "$REFRESH_SUBTYPE_RESCUE_INPUTS" != "1" && -f "$CANONICAL_STAGE2_TRAIN_OFFICIAL_SUBSET" ]]; then
  copy_into_branch_path "$CANONICAL_STAGE2_TRAIN_OFFICIAL_SUBSET" "$STAGE2_TRAIN_OFFICIAL_SUBSET"
  STAGE2_TRAIN_OFFICIAL_SUBSET_SOURCE_TYPE="copied_from_canonical"
  STAGE2_TRAIN_OFFICIAL_SUBSET_SOURCE_PATH="$CANONICAL_STAGE2_TRAIN_OFFICIAL_SUBSET"
  echo "[stage2-subtype] Copied canonical train subset into branch-local path."
else
  mkdir -p "$(dirname "$STAGE2_TRAIN_OFFICIAL_SUBSET")"
  stage2_train_subset_tmp="$(mktemp "${STAGE2_TRAIN_OFFICIAL_SUBSET}.tmp.XXXXXX")"
  python scripts/export_split_subset.py \
    --input data/processed/official_train_tagged.jsonl \
    --output "$stage2_train_subset_tmp" \
    --split-file data/splits/official/splits.json \
    --split-name rule_novelty_all \
    --split-role train \
    --exclude-split-file data/splits/official/splits.json \
    --exclude-split-name hard_triad_rule_novelty \
    --exclude-split-role valid
  mv "$stage2_train_subset_tmp" "$STAGE2_TRAIN_OFFICIAL_SUBSET"
  STAGE2_TRAIN_OFFICIAL_SUBSET_SOURCE_TYPE="regenerated_branch_local"
  STAGE2_TRAIN_OFFICIAL_SUBSET_SOURCE_PATH="data/processed/official_train_tagged.jsonl"
fi

if [[ "$REFRESH_SUBTYPE_RESCUE_INPUTS" != "1" && -f "$STAGE2_VALID_OFFICIAL_SUBSET" ]]; then
  if [[ -f "$CANONICAL_STAGE2_VALID_OFFICIAL_SUBSET" ]]; then
    assert_branch_local_matches_canonical "$CANONICAL_STAGE2_VALID_OFFICIAL_SUBSET" "$STAGE2_VALID_OFFICIAL_SUBSET" "branch-local valid subset"
  fi
  STAGE2_VALID_OFFICIAL_SUBSET_SOURCE_TYPE="reused_branch_local"
  STAGE2_VALID_OFFICIAL_SUBSET_SOURCE_PATH="$STAGE2_VALID_OFFICIAL_SUBSET"
  echo "[stage2-subtype] Reusing existing branch-local input: $STAGE2_VALID_OFFICIAL_SUBSET"
elif [[ "$REFRESH_SUBTYPE_RESCUE_INPUTS" != "1" && -f "$CANONICAL_STAGE2_VALID_OFFICIAL_SUBSET" ]]; then
  copy_into_branch_path "$CANONICAL_STAGE2_VALID_OFFICIAL_SUBSET" "$STAGE2_VALID_OFFICIAL_SUBSET"
  STAGE2_VALID_OFFICIAL_SUBSET_SOURCE_TYPE="copied_from_canonical"
  STAGE2_VALID_OFFICIAL_SUBSET_SOURCE_PATH="$CANONICAL_STAGE2_VALID_OFFICIAL_SUBSET"
  echo "[stage2-subtype] Copied canonical valid subset into branch-local path."
else
  mkdir -p "$(dirname "$STAGE2_VALID_OFFICIAL_SUBSET")"
  stage2_valid_subset_tmp="$(mktemp "${STAGE2_VALID_OFFICIAL_SUBSET}.tmp.XXXXXX")"
  python scripts/export_split_subset.py \
    --input data/processed/official_train_tagged.jsonl \
    --output "$stage2_valid_subset_tmp" \
    --split-file data/splits/official/splits.json \
    --split-name hard_triad_rule_novelty \
    --split-role valid
  mv "$stage2_valid_subset_tmp" "$STAGE2_VALID_OFFICIAL_SUBSET"
  STAGE2_VALID_OFFICIAL_SUBSET_SOURCE_TYPE="regenerated_branch_local"
  STAGE2_VALID_OFFICIAL_SUBSET_SOURCE_PATH="data/processed/official_train_tagged.jsonl"
fi

if [[ "$REFRESH_SUBTYPE_RESCUE_INPUTS" != "1" && -f "$ALL_FAMILY_PROXY_VALID_SUBSET" ]]; then
  if [[ -f "$CANONICAL_ALL_FAMILY_PROXY_VALID_SUBSET" ]]; then
    assert_branch_local_matches_canonical "$CANONICAL_ALL_FAMILY_PROXY_VALID_SUBSET" "$ALL_FAMILY_PROXY_VALID_SUBSET" "branch-local all-family proxy subset"
  fi
  ALL_FAMILY_PROXY_VALID_SUBSET_SOURCE_TYPE="reused_branch_local"
  ALL_FAMILY_PROXY_VALID_SUBSET_SOURCE_PATH="$ALL_FAMILY_PROXY_VALID_SUBSET"
  echo "[stage2-subtype] Reusing existing branch-local input: $ALL_FAMILY_PROXY_VALID_SUBSET"
elif [[ "$REFRESH_SUBTYPE_RESCUE_INPUTS" != "1" && -f "$CANONICAL_ALL_FAMILY_PROXY_VALID_SUBSET" ]]; then
  copy_into_branch_path "$CANONICAL_ALL_FAMILY_PROXY_VALID_SUBSET" "$ALL_FAMILY_PROXY_VALID_SUBSET"
  ALL_FAMILY_PROXY_VALID_SUBSET_SOURCE_TYPE="copied_from_canonical"
  ALL_FAMILY_PROXY_VALID_SUBSET_SOURCE_PATH="$CANONICAL_ALL_FAMILY_PROXY_VALID_SUBSET"
  echo "[stage2-subtype] Copied canonical all-family proxy subset into branch-local path."
else
  mkdir -p "$(dirname "$ALL_FAMILY_PROXY_VALID_SUBSET")"
  all_family_proxy_subset_tmp="$(mktemp "${ALL_FAMILY_PROXY_VALID_SUBSET}.tmp.XXXXXX")"
  python scripts/export_split_subset.py \
    --input data/processed/official_train_tagged.jsonl \
    --output "$all_family_proxy_subset_tmp" \
    --split-file data/splits/official/splits.json \
    --split-name rule_novelty_all \
    --split-role valid \
    --exclude-split-file data/splits/official/splits.json \
    --exclude-split-name hard_triad_rule_novelty \
    --exclude-split-role train
  mv "$all_family_proxy_subset_tmp" "$ALL_FAMILY_PROXY_VALID_SUBSET"
  ALL_FAMILY_PROXY_VALID_SUBSET_SOURCE_TYPE="regenerated_branch_local"
  ALL_FAMILY_PROXY_VALID_SUBSET_SOURCE_PATH="data/processed/official_train_tagged.jsonl"
fi

write_input_manifest

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
  --input "$STAGE2_TRAIN_OFFICIAL_SUBSET,$SYNTH_HARD_TRIADS_PATH" \
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
rm -f "$STAGE2_SKIPPED_ARTIFACT"
python - "$STAGE2_REPORT" "$STAGE2_SKIPPED_ARTIFACT" <<'PY'
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

# Zero-treatment gate: the v1 branch is equation-only, so it only differs
# from canonical stage2 when (a) at least one equation sample had its
# subtype temporarily overridden for the rescue search AND (b) at least
# one of those overrides produced a promoted annotation. If either
# counter is zero the branch dataset is effectively identical to
# canonical stage2 data and the 20-28 H100 hours that would follow are
# wasted. Fail fast BEFORE lora_train so the machine goes back to doing
# useful work.
rescue_hint_attempted = hint_diag.get("rescue_hint_attempted") or {}
rescue_promoted_with_hint = hint_diag.get("rescue_promoted_with_hint") or {}
equation_attempted = int(rescue_hint_attempted.get("equation", 0) or 0)
equation_promoted = int(rescue_promoted_with_hint.get("equation", 0) or 0)

if equation_attempted <= 0:
    skip_artifact = Path(sys.argv[2])
    skip_payload = {
        "skipped": True,
        "reason": "zero equation rescue_hint_attempted",
        "equation_attempted": equation_attempted,
        "equation_promoted": equation_promoted,
        "report": sys.argv[1],
    }
    skip_artifact.parent.mkdir(parents=True, exist_ok=True)
    skip_artifact.write_text(
        json.dumps(skip_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    raise SystemExit(
        "stage2 subtype-rescue branch has zero equation rescue_hint_attempted. "
        "No equation sample had a valid subtype hint to apply, so the rescue "
        "second pass saw the same subtypes as canonical stage2. "
        "Skip training this branch: it would consume GPU without any treatment."
    )

if equation_promoted <= 0:
    skip_artifact = Path(sys.argv[2])
    skip_payload = {
        "skipped": True,
        "reason": "zero equation rescue_promoted_with_hint",
        "equation_attempted": equation_attempted,
        "equation_promoted": equation_promoted,
        "report": sys.argv[1],
    }
    skip_artifact.parent.mkdir(parents=True, exist_ok=True)
    skip_artifact.write_text(
        json.dumps(skip_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    raise SystemExit(
        "stage2 subtype-rescue branch has zero equation rescue_promoted_with_hint "
        f"(attempted={equation_attempted}). The hint triggered the second pass "
        "but the search found no strictly-better candidate under the hinted "
        "subtype prior, so every attempt was rolled back. The resulting train "
        "dataset is equivalent to canonical stage2. "
        "Skip training this branch: it would consume GPU without any treatment."
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

python - "$STAGE2_CONFIG_TEMPLATE" \
  "$STAGE1_ADAPTER_DIR" \
  "$STAGE2_RUNTIME_CONFIG" \
  "$STAGE2_ADAPTER_DIR" \
  "$STAGE2_TRAIN_DATASET" \
  "$STAGE2_VALID_DATASET" <<'PY'
from pathlib import Path
import sys
import yaml

template_path, init_adapter_dir, output_path, adapter_dir, train_dataset, valid_dataset = sys.argv[1:7]
payload = yaml.safe_load(Path(template_path).read_text(encoding="utf-8")) or {}
training = payload.setdefault("training", {})
training["init_adapter_dir"] = init_adapter_dir
training["output_dir"] = adapter_dir
training["dataset_path"] = train_dataset
training["eval_path"] = valid_dataset
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
  --workdir "$STAGE2_BESTPROXY_WORKDIR" \
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
