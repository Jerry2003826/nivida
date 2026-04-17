# NVIDIA Nemotron Reasoning Challenge Repo

This repo is now organized around a single-LoRA competition pipeline:

- `raw_prompt` aligned training and inference
- competition-proxy evaluation with `competition_correct`
- teacher distillation via `program_signature` and compact traces
- hard-triad repair focused on `bit`, `cipher`, and `equation`

## Main Pipeline

### 1. Prepare official data

```bash
python scripts/prepare_data.py --config configs/data_official.yaml
```

This writes:

- `data/processed/official_train_tagged.jsonl`
- `data/splits/official/splits.json`

Processed examples use the unified schema:

- `id`
- `raw_prompt`
- `official_instruction`
- `parsed_examples`
- `query`
- `target_answer`
- `metadata`

`metadata` includes:

- `official_family`
- `subtype`
- `family_scores`
- `teacher_confidence`
- `program_signature`
- `difficulty`
- `source`
- `split`
- `extras`

### 2. Build synthetic stage-2 data

```bash
python -m src.teacher.synth_generator --config configs/synth.yaml
```

### 3. Build SFT datasets

Stage 1:

```bash
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/stage1_format_align_train.jsonl \
  --selection-profile stage1 \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty \
  --split-role train
```

Stage 2:

```bash
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl,data/synthetic/stage2_synth.jsonl \
  --output data/processed/stage2_distill_train.jsonl \
  --selection-profile stage2 \
  --completion-style token_trace \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad \
  --split-role train
```

Stage 3:

```bash
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/stage3_repair_train.jsonl \
  --selection-profile stage3 \
  --completion-style short_trace \
  --repair-artifact data/processed/baseline_eval.json \
  --split-file data/splits/official/splits.json \
  --split-name hard_triad \
  --split-role train
```

### 4. Train

Stage 1:

```bash
bash scripts/train_stage1_format_align.sh
```

Stage 2:

```bash
bash scripts/train_stage2_distill.sh
```

Stage 3:

```bash
bash scripts/train_stage3_repair.sh
```

### 5. Evaluate the local competition proxy

Teacher baseline:

```bash
python -m src.experiments.run_baseline \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/baseline_eval.json
```

Replica evaluator:

```bash
python -m src.experiments.eval_competition_replica \
  --predictions data/processed/inference_predictions.jsonl \
  --labels data/processed/official_train_tagged.jsonl \
  --splits data/splits/official/splits.json \
  --output data/processed/competition_replica_eval.json
```

Headline outputs:

- `competition_correct_rate`
- `exact_match_rate`
- `numeric_match_rate`
- `family_wise_competition_correct`

## Training Notes

- Default LoRA rank is `32`
- Default alpha is `64`
- `target_modules` may be a regex or an explicit list
- `max_seq_length: auto` uses a token-length audit on the built SFT dataset
- `rule_novelty` and `hard_triad` are the only maintained validation splits

## Useful Commands

Run the full local smoke workflow:

```bash
bash scripts/train_smoke_local.sh
```

Audit candidate LoRA module targets:

```bash
python -m src.student.audit_target_modules --config configs/train_lora.yaml
```

Package a trained adapter:

```bash
python -m src.student.package_submission \
  --adapter-dir artifacts/adapter_stage2_distill \
  --output submission.zip
```

Run tests:

```bash
python -m pytest -q
```
