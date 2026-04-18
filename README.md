# NVIDIA Nemotron Reasoning Challenge Repo

This repo is organized around a three-stage single-LoRA competition pipeline:

- harness-aligned `chat_thinking` prompts
- local competition-proxy evaluation with `competition_correct`
- staged fine-tuning: stage1 format align -> stage2 distill -> stage3 repair

## Canonical Training Order

Run these in order before packaging a submission:

```bash
python scripts/probe_chat_template.py
python scripts/inspect_target_modules.py --config configs/train_stage1_format.yaml

bash scripts/train_stage1_format_align.sh
bash scripts/train_stage2_distill.sh
bash scripts/train_stage3_repair.sh

python scripts/validate_submission.py \
  --config configs/train_stage3_repair.yaml \
  --adapter-dir artifacts/adapter_stage3_repair \
  --smoke-input data/processed/official_train_tagged.jsonl \
  --labels data/processed/official_train_tagged.jsonl \
  --splits data/splits/official/splits.json \
  --package-output submission.zip
```

## Data Preparation

Prepare the canonical official dataset and maintained splits:

```bash
python scripts/prepare_data.py --config configs/data_official.yaml
```

This writes:

- `data/processed/official_train_tagged.jsonl`
- `data/splits/official/splits.json`

Generate hard-triad synthetic data for stage2:

```bash
python -m src.teacher.synth_generator --config configs/synth_hard_triads.yaml
```

## SFT Dataset Builders

Stage1 build:

```bash
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/stage1_format_align_train.jsonl \
  --selection-profile stage1 \
  --prompt-mode chat_thinking \
  --tokenizer-path artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty_all \
  --split-role train \
  --exclude-split-file data/splits/official/splits.json \
  --exclude-split-name hard_triad_rule_novelty \
  --exclude-split-role valid
```

Because `rule_novelty_all` and `hard_triad_rule_novelty` are built
independently (different seeds, different subsets), stage1 train explicitly
excludes `hard_triad_rule_novelty/valid` so later hard-triad validation stays
unseen across the full staged pipeline (`stage1 -> stage2 init_adapter_dir=stage1 -> stage3 init_adapter_dir=stage2`).

Stage2 build with family balancing and stronger teacher search:

```bash
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl,data/synthetic/synth_hard_triads.jsonl \
  --output data/processed/stage2_distill_train.jsonl \
  --selection-profile stage2 \
  --prompt-mode chat_thinking \
  --tokenizer-path artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default \
  --completion-style token_trace \
  --beam-width 10 \
  --max-depth 3 \
  --top-k 3 \
  --balance-by-family \
  --hard-triad-repeat-factor 2 \
  --max-per-signature-bucket 64 \
  --report-output data/processed/stage2_distill_report.json \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty_all \
  --split-role train
```

Stage3 build from stage2 model failures plus all-family replay:

```bash
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/stage3_repair_train.jsonl \
  --selection-profile stage3 \
  --prompt-mode chat_thinking \
  --tokenizer-path artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default \
  --completion-style short_trace \
  --beam-width 8 \
  --max-depth 2 \
  --top-k 2 \
  --repair-artifact data/processed/stage2_model_failures_train.json \
  --replay-input data/processed/stage2_model_successes_all_train.json \
  --replay-ratio 0.25 \
  --report-output data/processed/stage3_repair_train_report.json
```

Stage3 failure / success buckets are produced by `scripts/train_stage3_repair.sh`:

- repair failures come from `hard_triad_rule_novelty/train` - the stage2 adapter
  predictions that missed on the hard triad, the only samples we want to correct
- replay successes come from `rule_novelty_all/train` - the stage2 adapter
  predictions that were correct across all six families, so replay can keep
  easy-triad families anchored and avoid catastrophic forgetting
- the builder receives `data/processed/official_train_tagged.jsonl` as input so
  `build_repair_set` can materialise replay records for ids that live outside
  the hard-triad subset

## Training Notes

- canonical train configs are:
  - `configs/train_stage1_format.yaml`
  - `configs/train_stage2_selected_trace.yaml`
  - `configs/train_stage3_repair.yaml`
- stage2 and stage3 continue from the previous stage adapter via `training.init_adapter_dir`
- `lora_train.py` supports `--force-train`; canonical stage scripts pass it explicitly
- `target_modules` are hard-validated before training starts
- `max_seq_length: auto` uses tokenizer-aware BPE accounting when a tokenizer path is configured
- stage2 / stage3 local inference defaults to `max_new_tokens: 2048` to avoid truncating `chat_thinking` generations
- `max_depth=3` is intentionally more expensive for stage2 selection; expect noticeably higher CPU time than `max_depth=2`

## Smoke and Validation

H100 smoke path:

```bash
bash scripts/train_stage1_smoke.sh
```

Legacy local smoke entry now forwards to the canonical stage1 smoke script:

```bash
bash scripts/train_smoke_local.sh
```

Validate a trained adapter before packaging:

```bash
python scripts/validate_submission.py \
  --config configs/train_stage3_repair.yaml \
  --adapter-dir artifacts/adapter_stage3_repair \
  --smoke-input data/processed/official_train_tagged.jsonl \
  --labels data/processed/official_train_tagged.jsonl \
  --splits data/splits/official/splits.json \
  --max-new-tokens 2048 \
  --package-output submission.zip
```

The validator hard-fails when:

- `adapter_config.json` is missing or its rank cannot be parsed
- the parsed rank exceeds 32
- `--labels` is passed without `--smoke-input`, or `--package-output` is passed
  without both `--smoke-input` and `--labels`
- `--package-output` runs before a successful `local_eval`

`--max-new-tokens` is optional; it overrides the inference token budget from
the config (defaults to 2048 for `chat_thinking` in stage2 / stage3 configs).

## Tests

```bash
python -m pytest -q
```
