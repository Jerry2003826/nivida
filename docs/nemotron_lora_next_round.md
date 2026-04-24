# Nemotron LoRA Next Round

This round optimizes for local parsed-exact ranking before Kaggle submission.
`eval_loss` is no longer a submission trigger.

## Local First

Build fixed labeled manifests:

```bash
python scripts/build_local_eval_manifests.py
```

Main manifest:

- `data/processed/local_eval_manifests/combined_balanced_48pf.jsonl`

Auxiliary manifests:

- `data/processed/local_eval_manifests/proxy_all_balanced_64pf.jsonl`
- `data/processed/local_eval_manifests/hard_triad_full.jsonl`

## Cloud Inference Only

Default discovery evaluates B thin, B thin checkpoints, official-balanced
continuations, and known shared/route variants:

```bash
bash scripts/run_cloud_inference_only_v3.sh
```

Explicit candidate list:

```bash
ADAPTERS="b_thin=artifacts/adapter_stage2_thin,ckpt314=artifacts/adapter_stage2_thin_official_balanced_20260424_161110Z/checkpoint-314" \
bash scripts/run_cloud_inference_only_v3.sh
```

Smoke first when GPU time is tight:

```bash
EVAL_INPUTS=smoke_6pf bash scripts/run_cloud_inference_only_v3.sh
```

## Local Scoring And Ranking

After copying prediction JSONL files back:

```bash
python scripts/score_local_eval_predictions.py \
  --pred-dir data/processed/local_eval_predictions_v3 \
  --manifest-dir data/processed/local_eval_manifests \
  --baseline b_thin
```

The main ranking is written to:

- `data/processed/eval/exact_reports_v3/combined_balanced_48pf/ranking.md`
- `data/processed/eval/exact_reports_v3/combined_balanced_48pf/ranking.json`

Submit only if the top candidate passes:

- `overall official_verify_accuracy > b_thin`
- no large family regresses by more than one sample

Tie-breakers are boxed-valid rate, shorter output, then smaller family/subtype
accuracy variance. The ranking also reports the worst family and subtype delta
against baseline.

## Training Variants

Build answer-focused continuation data:

```bash
bash scripts/build_stage2_answer_focused_data.sh
```

Then train only after inference ranking fails to find a clear winner:

```bash
python -m src.student.lora_train --config configs/train_stage2_thin_answer_only.yaml
python -m src.student.lora_train --config configs/train_stage2_thin_short_trace.yaml
```

`train_stage2_thin_short_trace.yaml` enables optional final-answer weighted
loss via `training.final_answer_loss`, multiplying the last `\boxed{...}` span
by `3.0`. Baseline and answer-only configs keep the original prompt-masked
loss behavior.
