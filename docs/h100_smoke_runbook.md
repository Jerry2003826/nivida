# H100 Smoke Runbook

This runbook is the shortest path from a fresh H100 machine to the first real
Nemotron stage1 smoke run.

The committed smoke config routes `KAGGLEHUB_CACHE` to `/workspace/.cache/kagglehub`
so the 47 GB base-model download lands on the large workspace volume instead of
the container root overlay. It also sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which was required on the
first real H100 smoke to avoid a step-0 optimizer OOM caused by allocator
fragmentation.

## 1. Bootstrap

```bash
bash scripts/bootstrap.sh
pip install -e .[train]
```

## 2. Kaggle / Tokenizer Sanity

Run the probe before any training command:

```bash
python scripts/probe_chat_template.py
```

Expected result:

- tokenizer-only mode succeeds
- `artifacts/chat_template_probe.json` exists
- the summary reports `trace-as-thinking viable         : True`
- `artifacts/chat_template_probe.json` reports `conclusions.first_public_sample_fits_budget == true`

## 3. Optional Dry-Run Verification

If the smoke datasets do not exist yet, prepare them with the same build steps
the smoke script uses:

```bash
python scripts/prepare_data.py --config configs/data_official.yaml

python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/smoke/stage1_format_align_train.jsonl \
  --selection-profile stage1 \
  --prompt-mode chat_thinking \
  --tokenizer-path artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty_all \
  --split-role train

python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/smoke/stage1_format_align_valid.jsonl \
  --selection-profile stage1 \
  --prompt-mode chat_thinking \
  --tokenizer-path artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty_all \
  --split-role valid

python scripts/subsample_jsonl.py \
  --input data/processed/smoke/stage1_format_align_train.jsonl \
  --output data/processed/smoke/stage1_format_align_train.jsonl \
  --limit 64

python scripts/subsample_jsonl.py \
  --input data/processed/smoke/stage1_format_align_valid.jsonl \
  --output data/processed/smoke/stage1_format_align_valid.jsonl \
  --limit 32
```

Then run the trainer in dry-run mode:

```bash
python -m src.student.lora_train --config configs/smoke/train_stage1_smoke.yaml --dry-run
```

Expected result:

- `artifacts/smoke/adapter_stage1_format/dry_run_manifest.json` exists
- `dataset_stats.length_unit == "bpe_tokens"`
- `preflight.status == "ok"`
- `preflight.disk_free_gb == null` and `preflight.kaggle_model_cached == null` in dry-run mode are expected because remote/runtime checks are intentionally skipped

## 4. First Real Smoke

```bash
bash scripts/train_stage1_smoke.sh
```

## 5. Success Criteria

The smoke run is good enough when all of the following are true:

- at least one save/eval cycle completes
- `artifacts/smoke/adapter_stage1_format/adapter_config.json` exists
- `artifacts/smoke/adapter_stage1_format/adapter_model.safetensors` exists
- `artifacts/smoke/adapter_stage1_format/training_metadata.json` exists
- `training_metadata.json` reports `preflight.status == "ok"`
- `training_metadata.json` reports `dataset_stats.length_unit == "bpe_tokens"`
- `training_metadata.json` reports a non-empty `matched_target_modules`

With `save_steps=4` and `eval_steps=4`, a successful smoke run should attempt
two saves across the 8 optimizer steps.

## 6. Time Budget

- cold start, first Kaggle model download included: about `20-25 minutes`
- warm rerun, model already cached: about `5-8 minutes`

The cold start is dominated by the first full Nemotron model download. The
trainer itself should start within a few minutes after dependencies and model
files are ready.

## 7. Cleanup After Failure

```bash
rm -rf artifacts/smoke/ data/processed/smoke/
```

Run cleanup before retrying after a partial or interrupted smoke run. This
avoids stale trainer state or half-written artifacts changing the next run.

## 8. Troubleshooting

### Missing tokenizer cache

- Run `python scripts/probe_chat_template.py`
- Confirm `artifacts/chat_template_probe.json` exists
- Confirm `artifacts/_tokenizer_cache/.../tokenizer_config.json` exists

### Kaggle auth / T&C failure

- Confirm `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`
- Confirm the model terms were accepted on Kaggle
- Retry `python scripts/probe_chat_template.py`

### `mamba_ssm` import failure

- Re-run `pip install -e .[train]`
- If the base image still lacks a compatible build, install the environment-
  specific `mamba-ssm` / `causal-conv1d` pair for that CUDA image before retrying

### Disk-space preflight failure

- Free space on the volume used by `preflight.disk_check_path`
- For uncached Kaggle models, confirm `KAGGLEHUB_CACHE` points at a large volume
- If a failed run filled the root overlay with a partial download, clear `/root/.cache/kagglehub/` before retrying

### Unwritable output directory

- Check permissions on `artifacts/` and its parent
- Clean up a stale `artifacts/smoke/adapter_stage1_format/` directory if needed

### Dataset path missing

- Re-run the optional dry-run preparation commands above
- Or run `bash scripts/train_stage1_smoke.sh`, which rebuilds the smoke datasets

### Preflight failure vs trainer/runtime failure

- If the run fails before model loading, inspect the `preflight` block in the
  manifest or metadata first
- If preflight passes but training fails later, inspect the trainer log for
  model load, checkpoint save, or CUDA/runtime issues
