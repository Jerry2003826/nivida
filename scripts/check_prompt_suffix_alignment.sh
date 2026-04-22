#!/usr/bin/env bash
# Audit that training prompts end with the exact suffix the Kaggle harness
# appends to every test prompt. Divergence here is one of the classic
# "train-test distribution drift" footguns for instruction-finetuned models.
#
# Authoritative suffix (copied verbatim from
# 官方资料/kaggle_metric/nvidia-nemotron-metric.ipynb, function
# generate_predictions):
#
#   \nPlease put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
#
# Why a shell script: this runs fast on the server over large JSONL files,
# is trivially grep-able, and has no python startup tax.
#
# Usage:
#   scripts/check_prompt_suffix_alignment.sh [PATH...]
#
# If PATH arguments are omitted, defaults to:
#   data/processed/competition_train.jsonl
#   data/processed/competition_val.jsonl
#   data/processed/*.jsonl
#
# Exit codes:
#   0 — every file checked ends every prompt with the expected suffix, and
#       every completion contains a \boxed{...} answer.
#   1 — at least one row is missing either the prompt suffix or a boxed
#       answer. Offending rows are printed to stderr (up to 5 per file).
#   2 — usage / environment error (file missing, jq missing).

set -Eeuo pipefail

if ! command -v jq >/dev/null 2>&1; then
  echo "check_prompt_suffix_alignment.sh: jq is required but not on PATH" >&2
  exit 2
fi

# Exact byte-for-byte suffix expected on every training prompt.
EXPECTED_SUFFIX=$'\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

if [[ $# -gt 0 ]]; then
  FILES=("$@")
else
  # Default search set. Missing files are tolerated (other environments may
  # only have a subset) but at least one must exist.
  FILES=()
  for candidate in \
    data/processed/competition_train.jsonl \
    data/processed/competition_val.jsonl \
    data/processed/hard_triad_proxy.jsonl \
    data/processed/all_family_proxy.jsonl \
    ; do
    if [[ -f "$candidate" ]]; then
      FILES+=("$candidate")
    fi
  done
  if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "check_prompt_suffix_alignment.sh: no default data files found under data/processed/. Pass paths explicitly." >&2
    exit 2
  fi
fi

total_bad=0
total_rows=0
for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "check_prompt_suffix_alignment.sh: file not found: $f" >&2
    exit 2
  fi

  rows=$(wc -l < "$f" | tr -d ' ')
  # Count rows whose 'prompt' field does NOT end with EXPECTED_SUFFIX.
  bad_prompt=$(
    jq -c --arg suffix "$EXPECTED_SUFFIX" '
      select((.prompt // .raw_prompt // "") | endswith($suffix) | not)
    ' "$f" | wc -l | tr -d ' '
  )
  # Count rows whose 'completion' / 'target_completion' does NOT contain a
  # boxed{...} substring. (We accept any boxed content — the metric
  # extractor uses the last non-empty one.)
  bad_completion=$(
    jq -c '
      select(
        ((.completion // .target_completion // .target // "") | test("\\\\boxed\\{")) | not
      )
    ' "$f" | wc -l | tr -d ' '
  )

  file_bad=$(( bad_prompt + bad_completion ))
  total_bad=$(( total_bad + file_bad ))
  total_rows=$(( total_rows + rows ))

  if [[ "$file_bad" -eq 0 ]]; then
    echo "[OK]    $f rows=$rows suffix_ok=yes boxed_ok=yes"
  else
    echo "[FAIL]  $f rows=$rows bad_suffix=$bad_prompt bad_completion=$bad_completion" >&2
    if [[ "$bad_prompt" -gt 0 ]]; then
      echo "  first 5 rows missing the expected prompt suffix:" >&2
      jq -c --arg suffix "$EXPECTED_SUFFIX" '
        select((.prompt // .raw_prompt // "") | endswith($suffix) | not)
        | {id: (.id // "?"), tail: ((.prompt // .raw_prompt // "") | .[-120:])}
      ' "$f" | head -5 >&2 || true
    fi
    if [[ "$bad_completion" -gt 0 ]]; then
      echo "  first 5 rows missing a \\boxed{...} answer:" >&2
      jq -c '
        select(
          ((.completion // .target_completion // .target // "") | test("\\\\boxed\\{")) | not
        )
        | {id: (.id // "?"), tail: ((.completion // .target_completion // .target // "") | .[-120:])}
      ' "$f" | head -5 >&2 || true
    fi
  fi
done

echo "---"
echo "summary: files=${#FILES[@]} rows=$total_rows bad=$total_bad"
if [[ "$total_bad" -gt 0 ]]; then
  exit 1
fi
exit 0
