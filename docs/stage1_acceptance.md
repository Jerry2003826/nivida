# Stage1 Acceptance

Use this checklist immediately after `scripts/train_stage1_format_align.sh`
finishes. The goal is to decide whether the stage1 adapter is safe to feed into
canonical stage2.

## One-Command Check

```bash
python scripts/check_stage1_acceptance.py \
  --adapter-dir artifacts/adapter_stage1_format \
  --log-path artifacts/run_logs/stage1_format_align.log \
  --output artifacts/stage1_acceptance.json
```

The script exits nonzero on failure and writes a single JSON summary on
success.

## What It Enforces

- `adapter_model.safetensors` exists
- `adapter_model.safetensors` is non-empty
- `adapter_config.json` exists
- `adapter_config.json` reports a LoRA rank in `1..32`
- `adapter_config.json` reports non-empty `target_modules`
- `training_metadata.json` exists
- `last_run_summary.json` exists
- `preflight.status == "ok"`
- `preflight.chat_template_sha16 == EXPECTED_CHAT_TEMPLATE_SHA16`
- `dataset_stats.length_unit == "bpe_tokens"`
- `num_train_records > 0`
- `num_matched_target_modules > 0`
- if a log path is provided, at least one training progress line was observed

## Pass Criteria

Treat stage1 as accepted only when all of the following are true:

1. `scripts/check_stage1_acceptance.py` exits with code `0`
2. `artifacts/stage1_acceptance.json` reports `"accepted": true`
3. the adapter directory contains no obvious partial-save corruption
4. the log shows real training progress, not only tokenizer/model loading

## Common Failure Modes

- Missing `last_run_summary.json`
  - The training process died before the final summary write completed.
- `preflight.status != "ok"`
  - The run ignored or bypassed a training preflight problem.
- `chat_template_sha16` mismatch
  - The tokenizer / chat template probe drifted from the harness-aligned contract.
- `dataset_stats.length_unit != "bpe_tokens"`
  - The run used fallback accounting rather than tokenizer-backed accounting.
- adapter rank missing / invalid / `> 32`
  - The adapter config is malformed or no longer matches the competition limit.
- `adapter_config.target_modules` empty
  - The saved adapter metadata is incomplete and should not seed stage2.
- `num_train_records <= 0`
  - The stage1 dataset was empty or the metadata write was corrupted.
- `num_matched_target_modules <= 0`
  - The LoRA target regex did not match the model; the run should be discarded.
- No progress line in the log
  - The job never left model-loading / startup.

## Next Step

If stage1 passes, the next canonical command is:

```bash
bash scripts/train_stage2_distill.sh
```

If stage1 fails, do not start stage2. Fix the failure, regenerate the stage1
adapter, and rerun the acceptance check.
