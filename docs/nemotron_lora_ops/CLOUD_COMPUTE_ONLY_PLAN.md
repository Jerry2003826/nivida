# Cloud Compute Only Plan

Cloud time is reserved for model loading and vLLM generation. All scoring,
ranking, reporting, and submission decisions happen locally.

## Current Policy

- No training on the next boot.
- No route/shared transplant submission.
- No Kaggle submission from the server.
- Use `official_balanced` as the ranking baseline whenever present.
- Shut the server down as soon as raw vLLM predictions are copied back.

## Server Commands

```bash
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST>
cd /workspace/nivida_h200_run
git pull
test -f artifacts/adapter_stage2_official_balanced_answer_only/adapter_model.safetensors
bash scripts/check_cloud_vllm_env.sh
```

Run the smallest smoke first:

```bash
head -n 6 data/processed/local_eval_manifests/smoke_6pf.jsonl \
  > data/processed/local_eval_manifests/smoke_head6.jsonl

EVAL_INPUTS=smoke_head6 \
ADAPTERS="answer_final=artifacts/adapter_stage2_official_balanced_answer_only" \
bash scripts/run_cloud_vllm_exact_eval_v3.sh
```

Only if smoke passes, run the candidate sweep:

```bash
EVAL_INPUTS=combined_balanced_48pf,proxy_all_balanced_64pf,hard_triad_full \
ADAPTERS="b_thin=artifacts/adapter_stage2_thin,official_balanced=artifacts/adapter_stage2_thin_official_balanced_20260424_161110Z,answer_final=artifacts/adapter_stage2_official_balanced_answer_only,answer_ckpt_100=artifacts/adapter_stage2_official_balanced_answer_only/checkpoint-100,answer_ckpt_200=artifacts/adapter_stage2_official_balanced_answer_only/checkpoint-200,answer_ckpt_294=artifacts/adapter_stage2_official_balanced_answer_only/checkpoint-294" \
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
```

The submit gate is:

- overall `official_verify_accuracy` must beat `official_balanced`;
- no large family can regress by more than one sample;
- tie-break by boxed-valid rate, shorter output, then lower family/subtype
  variance.
