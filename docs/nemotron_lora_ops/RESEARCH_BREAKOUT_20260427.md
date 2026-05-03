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
- Full weak-family upper bound: `python scripts/run_solver_breakout_v2.py --output-dir data/processed/solver_breakout_v2_full --output-md docs/solver_breakout_v2_full_latest.md`
- Baseline policy: `baseline=auto` must resolve to `official_balanced`; `b_thin`
  is only a historical reference.
- Submission gate: overall official-verify must beat `official_balanced`, no
  large family may regress by more than one sample, and submission-unsafe
  candidates are never selected automatically.
- Submit-safe classes: `model_only`/`trained_adapter` and `merged_adapter`.
- Research-only classes: `solver_assisted` and `prompt_ensemble`.
- Cloud vLLM preflight defaults to `VLLM_MIN_VERSION=0.14.0` because
  `vllm==0.11.2` fails NemotronH MoE LoRA loading with a missing
  `get_expert_mapping` implementation.
- Use `scripts/bootstrap_cloud_vllm_env.sh` on a fresh GPU pod to create the
  vLLM environment and immediately run the cheap version/ABI preflight.
- Use `scripts/run_cloud_eval_batch1.sh` as the first paid-GPU entrypoint; it
  wraps bootstrap/preflight, submit-safe smoke eval, optional full eval, and
  leaves scoring/submission decisions local.

## Research Lines

| line | implementation | decision rule |
| --- | --- | --- |
| Existing adapters | registry entries for `b_thin`, `official_balanced`, answer/short trace variants | eval-only sweep first |
| Adapter soup | `scripts/merge_lora_adapters.py` writes submit-safe merged adapters plus `merge_manifest.json` | evaluate merged adapters before submitting any specialist |
| Weak-family data | `scripts/build_research_rescue_data.py` writes mixed, equation, bit, eq+bit, and v2 rescue recipes plus per-output provenance | train only after no-GPU gate passes and solver breakout v2 has been reviewed |
| Solver-assisted inference | `scripts/apply_solver_assisted_finalizer.py` overrides only high-confidence equation/bit rows | research-only upper bound; use its wins to make training data |
| Solver breakout v2 | `scripts/run_solver_breakout_v2.py` reports top1, oracle@k, ranker miss, operator gap, safe override count, gain ceiling, and operator-gap clusters | decide whether the next GPU batch should train weak-family specialists or return to CPU operator work |
| Prompt profiles | `scripts/materialize_prompt_profile_manifest.py` builds `short_answer_biased` and `format_strict` manifests | research-only unless prompts become part of a model-only adapter recipe |
| Cloud manifest | `scripts/write_cloud_artifact_manifest.py` records git/runtime/adapter hashes and prediction counts | every paid GPU sweep must produce it |
| Public/local correlation | `scripts/update_lb_correlation_log.py` appends Kaggle public score, local exact metrics, adapter hashes, and merge weights | pause training after two local wins without public movement |
| Next-step planner | `scripts/plan_research_next_steps.py` reads readiness, batch gate, solver breakout, and public/local correlation artifacts | converts the current state into a primary action plus CPU/GPU exploration tracks |

## First GPU Batch

Run eval-only first. Do not train until the existing candidates plus
solver-assisted variants have been scored locally.

```bash
cd /workspace/nivida_h200_run
RUN_FULL=0 bash scripts/run_cloud_eval_batch1.sh

# Only if smoke is healthy and the budget allows the full arena:
RUN_SMOKE=0 RUN_FULL=1 bash scripts/run_cloud_eval_batch1.sh
```

Pull back `data/processed/vllm_exact_eval_v3_batch1`, then score locally:

```bash
python scripts/finalize_cloud_eval_batch1.py \
  --predictions-root data/processed/vllm_exact_eval_v3_batch1 \
  --output-root data/processed/eval/vllm_exact_eval_v3_batch1
```

Only package a Kaggle adapter if
`data/processed/eval/vllm_exact_eval_v3_batch1/batch1_gate_summary.json` says
`ready_to_submit=true`.

After scoring or after any Kaggle feedback, refresh the research router:

```bash
python scripts/plan_research_next_steps.py
```

The current generated plan is `docs/research_next_step_plan_latest.md`.

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
bit_rescue_v2_20260430_trained
eq_bit_rescue_v2
rank64_answer_only
final_answer_weighted_loss
```

The v2 rescue data/training entries are submission-unsafe until a real adapter
is trained, evaluated in the exact arena, and passes the `official_balanced`
gate. `bit_rescue_v2_20260430_trained` is the first trained submit-safe v2
adapter; its smoke eval tied `official_balanced` at `2 / 6`, so it still needs
a full vLLM exact sweep before any Kaggle submission decision.

The v2 config files are prebuilt:

```text
configs/train_stage2_equation_rescue_v2.yaml
configs/train_stage2_bit_rescue_v2.yaml
configs/train_stage2_eq_bit_rescue_v2.yaml
```

All three warm-start from the `official_balanced` adapter, use rank-32 LoRA,
and keep final-answer weighted loss enabled.

Full local solver breakout currently suggests:

```text
equation_template: top1 0.0936, oracle@k 0.0979, ranker miss 1, operator gap 45
bit_permutation:   top1 0.4203, oracle@k 0.5159, ranker miss 33, operator gap 161
```

This made `bit_rescue_v2` and `eq_bit_rescue_v2` the more plausible immediate
GPU training candidates. The first `bit_rescue_v2` adapter has been trained and
registered as `bit_rescue_v2_20260430_trained`; the next paid step is exact
eval, not another blind training run. Equation still needs CPU
operator/generator work because the current top-k barely contains the answer.

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
