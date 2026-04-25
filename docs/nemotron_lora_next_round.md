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
The equation-template diagnostic report is written to
`docs/equation_template_diagnostic_latest.md`.
The bit-permutation diagnostic report is written to
`docs/bit_permutation_diagnostic_latest.md`.

Current audit summary:

| manifest | query accuracy | oracle@k | support-full rate | main gap |
| --- | ---: | ---: | ---: | --- |
| `combined_balanced_48pf` | 0.7222 | 0.7257 | 0.9965 | bit/equation extrapolation |
| `proxy_all_balanced_64pf` | 0.7472 | 0.7528 | 0.9744 | bit/equation extrapolation |
| `hard_triad_full` | 0.4993 | 0.5205 | 0.9788 | equation/bit |

Note: these numbers use the official binary guard. Pure `0/1` answers are
strict strings, so older audits overstated bit accuracy.

Prioritize solver work in this order:

1. symbolic equation templates previously mislabeled as `equation_position`
2. `bit_permutation`
3. residual `cipher_char_sub`

For `equation_template`, the current diagnostic is blunt: across
`combined_balanced_48pf`, `proxy_all_balanced_64pf`, and `hard_triad_full`,
only 21 / 275 rows are top-1 correct and only 23 / 275 are oracle-at-10. On
`hard_triad_full`, only 17 / 190 have a correct candidate in the top 10. The
bounded literal-alternative expansion and query-copy tie-break lifted
`hard_triad_full` `equation_template` top1 from 14 / 190 to 16 / 190, so the
signal is real but small. The next useful local change is broader
template/operator generation plus a verifier that can reject underconstrained
support-perfect programs.

Stage2 trace selection is now query-aware on labeled data. If a solver fits all
support examples but its query prediction disagrees with the known target, the
sample is rejected from the strict trace bucket and can only enter as
answer-only silver supervision when that pool is enabled.

Stage2 strict selection also rejects high-risk `equation_template` traces when
the diagnostic labels them as `ranker_miss_oracle_hit`,
`operator_gap_oracle_miss`, or `unseen_literal_high_risk`. This keeps bad
template rationales out of trace training while still allowing the final answer
to be used in answer-only variants.

For cipher char-substitution prompts, chain search now prefers
`vocabulary_cipher` before raw `fixed_substitution`. This lets the solver
complete unseen query letters against the Wonderland vocabulary instead of
passing unknown ciphertext through unchanged.

For bit affine fits, GF(2) free variables are now selected by a sparse-solution
prior. A new `binary_boolean_expr` operator also fits per-output-bit constants,
copy, NOT, AND/OR/XOR, NAND/NOR, majority, and choice expressions. Together
these changes raise `hard_triad_full` bit accuracy from `0.2750` to `0.4500`
under the official binary-strict metric, and `bit_permutation` from `0.3975` to
`0.4477` in the latest audit.

The audit now reports support-full oracle@k. For `bit_permutation`, oracle@k is
now above top1 (`hard_triad_full`: 0.5063 oracle@k vs 0.4477 query accuracy),
so the next bit work can combine verifier/ranker features with candidate-space
cleanup. The boolean-expression operator is intentionally wider and slows full
audits, so do not treat it as free complexity.

Use the bit diagnostic to split the next work:

```bash
python scripts/diagnose_bit_permutation.py
```

Latest default diagnostic: 149 low-risk top1 hits, 32 ranker-miss/oracle-hit
rows, 161 operator-gap rows, and 2 support-incomplete rows across the three
local manifests.

For equation tags, position transduction now only covers outputs explainable
from input-position characters. If the output introduces literal symbols or
uses a symbol more often than the input provides, the sample is tagged as an
equation template instead. On the current hard-triad manifest, this reclassifies
190 / 193 old `equation_position` rows as template-like in the current audit.

For numeric equations, `binary_equation_rule` can now use exact lookup only for
unrelated support operators while keeping a general rule for the query-relevant
operator. Search also has a query-aware prior for unseen `+` and `-` operators.
This raises `hard_triad_full` equation-numeric query accuracy from `0.4500` to
`0.5750`.

## Local First

Build fixed labeled manifests:

```bash
python scripts/build_local_eval_manifests.py
python scripts/diagnose_equation_template.py
python scripts/diagnose_bit_permutation.py
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
