# Cloud Compute Only Plan

Cloud time is reserved for model loading and vLLM generation. All scoring,
ranking, reporting, and submission decisions happen locally.

## Current Policy

- Batch 1 is eval-only; training starts only after local exact arena shows a
  useful signal.
- No route/shared transplant submission.
- Solver-assisted and prompt-ensemble candidates are research-only because the
  Kaggle package is adapter-only.
- Submit-safe research candidates are model-only adapters and merged adapters
  created by `scripts/merge_lora_adapters.py`.
- No Kaggle submission from the server.
- Use `official_balanced` as the ranking baseline whenever present.
- Every cloud run must write `data/processed/vllm_exact_eval_v3/artifact_manifest.json`.
- Shut the server down as soon as raw vLLM predictions and the artifact
  manifest are copied back.

## Server Commands

```bash
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST>
cd /workspace/nivida_h200_run
git pull
bash scripts/check_cloud_vllm_env.sh
```

Run the smallest smoke first:

```bash
EVAL_INPUTS=smoke_head6 \
bash scripts/run_cloud_vllm_exact_eval_v3.sh
```

Only if smoke passes, run the registry-backed candidate sweep. The script
auto-discovers existing research adapters and checkpoints. It intentionally
skips the rank-64 candidate unless `INCLUDE_SUBMISSION_UNSAFE=1`.

```bash
EVAL_INPUTS=combined_balanced_48pf,proxy_all_balanced_64pf,hard_triad_full \
bash scripts/run_cloud_vllm_exact_eval_v3.sh
```

For an explicit sweep, keep names aligned with
`configs/research_breakout_candidates.json`:

```bash
EVAL_INPUTS=combined_balanced_48pf,proxy_all_balanced_64pf,hard_triad_full \
ADAPTERS="official_balanced=artifacts/adapter_stage2_thin_official_balanced_20260424_161110Z,answer_only_continuation=artifacts/adapter_stage2_official_balanced_answer_only,short_trace_continuation=artifacts/adapter_stage2_official_balanced_short_trace,mixed_answer_short=artifacts/adapter_stage2_mixed_answer_short,equation_rescue=artifacts/adapter_stage2_equation_rescue,bit_rescue=artifacts/adapter_stage2_bit_rescue,eq_bit_rescue=artifacts/adapter_stage2_eq_bit_rescue,final_answer_weighted_loss=artifacts/adapter_stage2_final_answer_weighted,soup_answer_short=artifacts/merged/soup_answer_short,soup_eq_bit=artifacts/merged/soup_eq_bit,soup_all_rescue=artifacts/merged/soup_all_rescue,soup_official_answer_rescue=artifacts/merged/soup_official_answer_rescue" \
bash scripts/run_cloud_vllm_exact_eval_v3.sh
```

## Pull Results Back

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 -r \
  root@<HOST>:/workspace/nivida_h200_run/data/processed/vllm_exact_eval_v3 \
  data/processed/
```

After this copy finishes, shut down the GPU server.

## Local Follow-Up

```bash
python scripts/score_vllm_exact_eval_outputs.py \
  --predictions-root data/processed/vllm_exact_eval_v3 \
  --output-root data/processed/eval/vllm_exact_eval_v3

python scripts/rank_research_candidates.py \
  --report official_balanced=data/processed/eval/vllm_exact_eval_v3/combined_balanced_48pf/official_balanced/report.json \
  --report answer_only_continuation=data/processed/eval/vllm_exact_eval_v3/combined_balanced_48pf/answer_only_continuation/report.json
```

The submit gate is:

- overall `official_verify_accuracy` must beat `official_balanced`;
- no large family can regress by more than one sample;
- tie-break by boxed-valid rate, shorter output, then lower family/subtype
  variance.

After Kaggle returns a public score, append the correlation record locally:

```bash
python scripts/update_lb_correlation_log.py \
  --candidate <candidate_name> \
  --public-score <score> \
  --exact-report data/processed/eval/vllm_exact_eval_v3/combined_balanced_48pf/<candidate_name>/report.json \
  --adapter-path <adapter_path>
```
