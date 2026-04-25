# Nemotron LoRA Next Round

This round optimizes for local parsed-exact ranking before Kaggle submission,
but the larger objective has changed: build solver/verifier coverage by family,
then train LoRA on solver-verified data. `eval_loss` is no longer a submission
trigger.

Current public leaderboard baseline:

- `official-balanced thin continuation from B thin 20260424_161110Z`: `0.57`
- Prior B thin / norm-shared baseline: `0.54`

Treat the official-balanced adapter as the new baseline. B thin is now only a
diagnostic reference.

The 0.87 target is treated as a task-system problem, not a plain continuation
training problem. See `docs/nemotron_087_strategy.md`.

## Solver Coverage First

Run the local rule/search coverage audit before spending GPU:

```bash
python scripts/audit_solver_coverage.py
```

The markdown report is written to `docs/solver_coverage_audit_latest.md`.

Current audit summary:

| manifest | query accuracy | support-full rate | main gap |
| --- | ---: | ---: | --- |
| `combined_balanced_48pf` | 0.6493 | 0.9722 | hard triad extrapolation |
| `proxy_all_balanced_64pf` | 0.6932 | 0.9545 | hard triad extrapolation |
| `hard_triad_full` | 0.3329 | 0.9408 | equation/cipher/bit |

Prioritize solver work in this order:

1. `equation_position`
2. `cipher_char_sub`
3. `bit_permutation`

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
continuations, answer-focused continuations, and known shared/route variants:

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
  --manifest-dir data/processed/local_eval_manifests
```

The scoring script defaults to `--baseline auto`: it prefers the
official-balanced final adapter, then falls back to `b_thin` only when the
official-balanced predictions are absent.

The main ranking is written to:

- `data/processed/eval/exact_reports_v3/combined_balanced_48pf/ranking.md`
- `data/processed/eval/exact_reports_v3/combined_balanced_48pf/ranking.json`

Submit only if the top candidate passes:

- `overall official_verify_accuracy > official-balanced baseline`
- no large family regresses by more than one sample

Tie-breakers are boxed-valid rate, shorter output, then smaller family/subtype
accuracy variance. The ranking also reports the worst family and subtype delta
against baseline.

## Training Variants

Build answer-focused continuation data:

```bash
bash scripts/build_stage2_answer_focused_data.sh
```

Then train only after inference ranking fails to find a clear winner. The
preferred new variants warm-start from the `0.57` official-balanced adapter:

```bash
python -m src.student.lora_train --config configs/train_stage2_official_balanced_answer_only.yaml
python -m src.student.lora_train --config configs/train_stage2_official_balanced_short_trace.yaml
```

The older B-thin warm-start variants are kept as controls:

```bash
python -m src.student.lora_train --config configs/train_stage2_thin_answer_only.yaml
python -m src.student.lora_train --config configs/train_stage2_thin_short_trace.yaml
```

The short-trace configs enable optional final-answer weighted loss via
`training.final_answer_loss`, multiplying the last `\boxed{...}` span by `3.0`.
Baseline and answer-only configs keep the original prompt-masked loss behavior.
