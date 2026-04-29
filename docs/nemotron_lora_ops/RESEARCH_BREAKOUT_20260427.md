# Research Breakout 2026-04-27

Current public baseline is `official-balanced = 0.57`. The next push is a
portfolio, not a single LoRA guess: exact-eval calibration, weak-family
solver/verifier work, answer-focused data recipes, training variants, prompt
profiles, adapter soup/merge, and solver-assisted diagnostics all enter the
same research arena. Kaggle submission remains adapter-only, so solver-assisted
and prompt-ensemble candidates are research-only unless the competition
submission contract changes.

## Local Arena

- Canonical registry: `configs/research_breakout_candidates.json`
- Registry check: `python scripts/build_research_candidate_registry.py --check`
- Unified ranking: `python scripts/rank_research_candidates.py --report official_balanced=... --report candidate=...`
- Weak-family upper bound: `python scripts/run_solver_breakout_v2.py --limit 64`
- Baseline policy: `baseline=auto` must resolve to `official_balanced`; `b_thin`
  is only a historical reference.
- Submission gate: overall official-verify must beat `official_balanced`, no
  large family may regress by more than one sample, and submission-unsafe
  candidates are never selected automatically.
- Submit-safe classes: `model_only` and `merged_adapter`.
- Research-only classes: `solver_assisted` and `prompt_ensemble`.

## Research Lines

| line | implementation | decision rule |
| --- | --- | --- |
| Existing adapters | registry entries for `b_thin`, `official_balanced`, answer/short trace variants | eval-only sweep first |
| Adapter soup | `scripts/merge_lora_adapters.py` writes submit-safe merged adapters plus `merge_manifest.json` | evaluate merged adapters before submitting any specialist |
| Weak-family data | `scripts/build_research_rescue_data.py` writes mixed, equation, bit, eq+bit, and v2 rescue recipes plus per-output provenance | train only after no-GPU gate passes and solver breakout v2 has been reviewed |
| Solver-assisted inference | `scripts/apply_solver_assisted_finalizer.py` overrides only high-confidence equation/bit rows | research-only upper bound; use its wins to make training data |
| Solver breakout v2 | `scripts/run_solver_breakout_v2.py` reports top1, oracle@k, ranker miss, operator gap, safe override count, and gain ceiling | decide whether the next GPU batch should train weak-family specialists or return to CPU operator work |
| Prompt profiles | `scripts/materialize_prompt_profile_manifest.py` builds `short_answer_biased` and `format_strict` manifests | research-only unless prompts become part of a model-only adapter recipe |
| Cloud manifest | `scripts/write_cloud_artifact_manifest.py` records git/runtime/adapter hashes and prediction counts | every paid GPU sweep must produce it |
| Public/local correlation | `scripts/update_lb_correlation_log.py` appends Kaggle public score, local exact metrics, adapter hashes, and merge weights | pause training after two local wins without public movement |

## First GPU Batch

Run eval-only first. Do not train until the existing candidates plus
solver-assisted variants have been scored locally.

```bash
cd /workspace/nivida_h200_run
git pull
bash scripts/check_cloud_vllm_env.sh

EVAL_INPUTS=smoke_head6 \
bash scripts/run_cloud_vllm_exact_eval_v3.sh

EVAL_INPUTS=combined_balanced_48pf,proxy_all_balanced_64pf,hard_triad_full \
bash scripts/run_cloud_vllm_exact_eval_v3.sh
```

Pull back `data/processed/vllm_exact_eval_v3`, then score locally:

```bash
python scripts/score_vllm_exact_eval_outputs.py \
  --predictions-root data/processed/vllm_exact_eval_v3 \
  --output-root data/processed/eval/vllm_exact_eval_v3
```

## Training Batch

Only start training if Batch 1 shows local exact-eval signal. The fixed first
matrix is:

```text
answer_only_continuation
short_trace_continuation
mixed_answer_short
equation_rescue
bit_rescue
eq_bit_rescue
equation_rescue_v2
bit_rescue_v2
eq_bit_rescue_v2
rank64_answer_only
final_answer_weighted_loss
```

The v2 rescue entries are submission-unsafe data/training candidates until a
real adapter is trained, evaluated in the exact arena, and passes the
`official_balanced` gate.

`rank64_answer_only` is intentionally marked `submission_safe=false` because
the current Kaggle runtime contract has `max_lora_rank=32`; keep it research
only unless that contract changes.

After training, build submit-safe soups before choosing a Kaggle candidate:

```bash
python scripts/merge_lora_adapters.py \
  --method linear \
  --clean \
  --output artifacts/merged/soup_answer_short \
  --adapter answer_only_continuation=artifacts/adapter_stage2_official_balanced_answer_only:0.5 \
  --adapter short_trace_continuation=artifacts/adapter_stage2_official_balanced_short_trace:0.5
```

Use `svd-rank32` only when rank growth or a rank-64 research adapter needs to
be compressed into the current submit-safe rank budget.

## Stop Rules

- If two GPU batches improve local exact but do not move Kaggle, pause training
  and recalibrate local evaluation.
- If gains mostly come from solver-assisted overrides, convert those wins into
  answer-only or safe short-trace data instead of treating the override as a
  submission candidate.
- Route/shared transplant remains downgraded until exact-eval proves a stable
  `norm-and-no-public-route` gain.
