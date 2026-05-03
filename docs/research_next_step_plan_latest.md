# Research Next Step Plan

- primary_action: `run_gpu_eval_batch1`
- gpu_allowed: `True`
- reason: local gate is ready, but batch1 vLLM exact-eval has not been scored
- public_stall_count: `0`

## Next Commands

- `cd /workspace/nivida_h200_run`
- `RUN_FULL=0 bash scripts/run_cloud_eval_batch1.sh`
- `RUN_SMOKE=0 RUN_FULL=1 bash scripts/run_cloud_eval_batch1.sh`

## Exploration Tracks

### 1. equation_operator_dsl
- gpu_required: `False`
- reason: equation oracle@k=0.098, operator_gap_rate=0.191; current search space is the bottleneck
- cluster target_not_expressible examples
- add only low-risk operator families
- keep high-risk rows answer-only

### 2. bit_ranker_v2
- gpu_required: `False`
- reason: bit top1=0.420, oracle@k=0.516, ranker_miss=33; rerank headroom remains
- leave-one-out stability weighting
- complexity penalty calibration
- oracle-distance feature ablation

### 3. bit_rescue_training_data
- gpu_required: `False`
- reason: bit safe_override_possible_rate=0.420; enough high-confidence rows for answer-only rescue data
- refresh bit_rescue_v2 data
- train only after batch1 exact eval is scored
- keep solver finalizer research-only

### 4. eval_correlation_log
- gpu_required: `False`
- reason: public/local correlation remains the long-term stop rule
- append every Kaggle result
- pause training after two local wins without public movement
