# Cloud Compute Only Plan

This plan keeps cloud time limited to model forward/generation work. All analysis, scoring, report rendering, and submission decisions happen locally.

## What Runs Locally

- Route selection stability rendering.
- Safe expert intersection analysis.
- Data family distribution analysis.
- Exact-eval scoring from prediction JSONL.
- Candidate ranking after converting vLLM raw generations into exact-eval reports.
- Submission decision.

## What Runs on the GPU Server

- Per-example prompt-only route probes.
- Optional visible-public generation diagnostic.
- Optional vLLM exact-eval generation for labeled holdout.

No training. No Kaggle submission. No report writing beyond raw JSON outputs.

## Files to Sync to Server

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 tools/probe_nemotron_expert_routes.py root@<HOST>:/workspace/nivida_h200_run/scripts/probe_nemotron_expert_routes.py
scp -P <PORT> -i ~/.ssh/id_ed25519 tools/probe_nemotron_expert_routes_batch.py root@<HOST>:/workspace/nivida_h200_run/scripts/probe_nemotron_expert_routes_batch.py
scp -P <PORT> -i ~/.ssh/id_ed25519 tools/mix_route_reports.py root@<HOST>:/workspace/nivida_h200_run/scripts/mix_route_reports.py
scp -P <PORT> -i ~/.ssh/id_ed25519 tools/run_cloud_compute_only_v3.sh root@<HOST>:/workspace/nivida_h200_run/scripts/run_cloud_compute_only_v3.sh
scp -P <PORT> -i ~/.ssh/id_ed25519 tools/run_cloud_inference_only_v3.sh root@<HOST>:/workspace/nivida_h200_run/scripts/run_cloud_inference_only_v3.sh
```

## Minimal Server Run

```bash
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST>
cd /workspace/nivida_h200_run
bash scripts/run_cloud_compute_only_v3.sh
```

By default this uses `probe_nemotron_expert_routes_batch.py`, so the model and adapter are loaded once for all prompt-only route jobs. Set `USE_BATCH_PROBE=0` only if the batch script fails.

Optional generation diagnostic:

```bash
RUN_PUBLIC_GENERATION_DIAG=1 bash scripts/run_cloud_compute_only_v3.sh
```

Optional inference prediction generation:

```bash
bash scripts/run_cloud_inference_only_v3.sh
```

The HF/PEFT inference script is useful only for tiny smoke checks. For
candidate selection, prefer the vLLM proxy path:

```bash
bash scripts/check_cloud_vllm_env.sh

EVAL_INPUTS=smoke_head6 \
ADAPTERS="answer_final=artifacts/adapter_stage2_official_balanced_answer_only" \
bash scripts/run_cloud_vllm_exact_eval_v3.sh
```

For the full candidate pass, use:

```bash
EVAL_INPUTS=combined_balanced_48pf,proxy_all_balanced_64pf,hard_triad_full \
bash scripts/run_cloud_vllm_exact_eval_v3.sh
```

`VENV` may point either at a venv directory or at its `bin/activate` file. Keep scoring local.

## Pull Results Back

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 -r root@<HOST>:/workspace/nivida_h200_run/data/processed/route_probe_v3 data/processed/
scp -P <PORT> -i ~/.ssh/id_ed25519 -r root@<HOST>:/workspace/nivida_h200_run/data/processed/local_eval_predictions_v3 data/processed/
scp -P <PORT> -i ~/.ssh/id_ed25519 -r root@<HOST>:/workspace/nivida_h200_run/data/processed/vllm_exact_eval_v3 data/processed/
```

After this, the server can be shut down.

## Local Follow-up Commands

```bash
python tools/analyze_route_selection_stability.py \
  --route-dir data/processed/route_probe_v3 \
  --source public_visible=public_visible_prompt_examples.json:0.25 \
  --source official_hard=official_hard_prompt_examples.json:0.30 \
  --source official_all=official_all_prompt_examples.json:0.30 \
  --source stage2_train=stage2_train_prompt_examples.json:0.15 \
  --no-public-source official_hard=official_hard_prompt_examples.json:0.40 \
  --no-public-source official_all=official_all_prompt_examples.json:0.40 \
  --no-public-source stage2_train=stage2_train_prompt_examples.json:0.20 \
  --output data/processed/route_probe_v3/selection_stability_v3.json

python tools/evaluate_predictions_exact.py \
  --predictions data/processed/local_eval_predictions_v3/b_thin_predictions.jsonl \
  --labels data/processed/proxy_all_family_valid.jsonl \
  --output-json data/processed/eval/b_thin_proxy_all_exact_eval.json \
  --output-csv data/processed/eval/b_thin_proxy_all_exact_eval_records.csv \
  --output-md LOCAL_EXACT_EVAL_b_thin_proxy_all.md

python scripts/score_vllm_exact_eval_outputs.py \
  --predictions-root data/processed/vllm_exact_eval_v3 \
  --output-root data/processed/eval/vllm_exact_eval_v3
```

## Stop Criteria

Do not build or submit a new adapter unless:

- no-public per-example route selection has stable overlap with norm-based selection;
- the proposed expert set stays small;
- local parsed-exact inference eval does not show systematic family damage;
- visible public remains diagnostic only.
