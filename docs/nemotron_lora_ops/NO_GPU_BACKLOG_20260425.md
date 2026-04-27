# No-GPU Backlog 2026-04-25

This is the local-only checklist before spending more GPU time. The current
leaderboard baseline is the official-balanced continuation at `0.57`; local
ranking must use `official_balanced` as the baseline whenever that candidate is
present.

## Completed Locally

| item | status | evidence |
| --- | --- | --- |
| Local eval manifests | done | `python scripts/build_local_eval_manifests.py` wrote 8 manifests |
| Solver coverage audit | done | `python scripts/audit_solver_coverage.py` wrote 1349 records |
| Equation-template diagnostic | done | `python scripts/diagnose_equation_template.py` wrote 275 rows |
| Bit-permutation diagnostic | done | `python scripts/diagnose_bit_permutation.py` wrote 345 rows |
| Tokenizer-only probe | done | chat template SHA16 `ab7813c3abdd9cb6`, first public sample length 275 |
| Chat-template SHA recheck | done | `python scripts/recheck_chat_template_sha16.py --output data/processed/recheck_chat_template_sha16.json` |
| Answer-focused datasets | done | answer-only train/valid rows `4674/658`; short-trace train/valid rows `4674/658` |
| Cross-platform answer-focused builder | done | `python scripts/build_stage2_answer_focused_data.py --dry-run` resolves parent data on Windows |
| Stage2 teacher provenance rebuild | done | `python scripts/rebuild_stage2_teacher_inputs.py` rebuilds parent official tagged data, splits, stage2 subsets, and subset provenance |
| No-GPU readiness gate | done | `make no-gpu-readiness` runs the full local pre-GPU gate and writes `data/processed/no_gpu_readiness_gate.json` |
| Cloud artifact preflight plan | done | `python scripts/check_cloud_eval_inputs.py --dry-run ...` is part of the readiness gate |
| Research breakout registry | done | `python scripts/build_research_candidate_registry.py --check` keeps `official_balanced` as the default arena baseline |
| Research weak-family data recipes | done | `python scripts/build_research_rescue_data.py` prepares mixed, equation, bit, and eq+bit SFT recipes with provenance |
| Submit-safe adapter soup tooling | done | `python scripts/merge_lora_adapters.py --method linear ...` merges adapter-only LoRA candidates and writes `merge_manifest.json` |
| Public/local correlation log | done | `python scripts/update_lb_correlation_log.py ...` records Kaggle public score beside local exact metrics and adapter hashes |
| Prompt/boxed guard check | done | `sh scripts/check_prompt_suffix_alignment.sh ...` checked 10664 rows, bad `0` |
| Fast local tests | done | targeted diagnostic/cloud/shell/tokenizer tests are part of `make no-gpu-readiness` |
| Full local tests | done | latest `python -m pytest -q` -> `482 passed, 9 skipped` |

## Tooling Changes Completed

- `scripts/diagnose_equation_template.py` now supports `--failure-class`,
  `--subtype`, and `--limit`, and exports support inputs/outputs alongside
  query, target, top prediction, risk class, and target expressibility.
- `scripts/diagnose_bit_permutation.py` now supports the same filters and
  exports ranker-miss features: top/oracle Hamming distance, oracle prediction,
  expression op sequence, and expression complexity.
- `scripts/check_prompt_suffix_alignment.sh` now has a Python fallback when
  `jq` is unavailable and accepts both raw guarded prompts and chat-template
  prompts that contain the official guard inside the rendered user turn.
- `scripts/probe_chat_template.py` can be run directly as
  `python scripts/probe_chat_template.py`; it no longer requires `python -m`.
- `scripts/build_stage2_answer_focused_data.py` is now the canonical
  cross-platform entrypoint for answer-only and short-trace data preparation.
  The `.sh` file remains a thin Linux wrapper.
- `scripts/run_no_gpu_readiness_gate.py` is now the canonical pre-GPU local
  gate. It regenerates canonical reports and stage2 teacher provenance, runs
  local checks and tests, and fails if teacher parity lacks provenance or if
  tracked reports drift.
- `scripts/rebuild_stage2_teacher_inputs.py` is the canonical CPU-only way to
  rebuild current-code stage2 teacher inputs before answer-focused data or
  teacher-gate parity audits.
- `scripts/check_cloud_eval_inputs.py` is the canonical CPU-only preflight for
  cloud exact-eval inputs and adapters. It checks ignored eval manifests,
  duplicate ids, target labels, adapter weights/configs, config files, and
  critical cloud scripts before vLLM is touched.
- `configs/research_breakout_candidates.json` is the canonical candidate
  registry for the aggressive research matrix. It records adapter paths, prompt
  profiles, data recipes, family focus, GPU need, expected runtime, and
  submission-safety.
- `scripts/rank_research_candidates.py` is the registry-aware exact arena
  ranker. It defaults to `official_balanced` and refuses to auto-select
  submission-unsafe candidates such as the rank-64 research arm. It also treats
  solver-assisted and prompt-ensemble candidates as research-only even if a
  registry entry is accidentally marked safe.
- `scripts/merge_lora_adapters.py` is the CPU-only adapter soup tool. It can
  linearly merge same-schema LoRA adapters or compress LoRA A/B deltas with
  `svd-rank32`, while checking the current rank budget and adapter-only
  packaging shape.
- `scripts/apply_solver_assisted_finalizer.py` creates solver-assisted raw
  predictions while preserving the original model generation for audit. These
  outputs are research-only and should be converted into training data, not
  treated as Kaggle-submittable predictions.
- `scripts/materialize_prompt_profile_manifest.py` prepares profiled eval
  manifests for `short_answer_biased` and `format_strict` prompt sweeps.
- `scripts/build_research_rescue_data.py` builds weak-family train/valid
  recipes from answer-only plus safe short-trace rows. Each output JSONL now
  has a `.provenance.json` sidecar and each row records risk class, source hash,
  answer hash, and weak-family diagnostic features.
- `scripts/update_lb_correlation_log.py` appends public leaderboard feedback to
  `data/processed/eval/lb_correlation_log.json` so local exact metrics can be
  audited against Kaggle movement over time.
- `scripts/write_cloud_artifact_manifest.py` writes cloud run manifests with
  git/runtime versions, adapter hashes, preflight hash, and raw prediction
  line counts.
- `scripts/run_cloud_vllm_exact_eval_v3.sh` and
  `scripts/run_cloud_inference_only_v3.sh` now fail on missing explicit
  adapters, repair missing checkpoint `adapter_config.json` from the B-thin
  reference when possible, and write a cloud artifact preflight JSON report.
- `scripts/run_cloud_vllm_exact_eval_v3.sh` materializes
  `data/processed/local_eval_manifests/smoke_head6.jsonl` from `smoke_6pf`
  automatically when the smoke default is used, so the first GPU smoke does not
  require a manual `head` command.

## Current Solver Read

- `equation_template` remains the highest-value CPU target.
  - Current ops can fit support+query target for `182 / 235` rows.
  - Only `21` have an operator-template fit where the query key was seen in
    support.
  - `unseen_key_template_miss` remains unsafe for strict trace training; keep
    it answer-only/silver only.
- `bit_permutation` has bounded ranker upside but a larger operator gap.
  - Low-risk top1 rows: `146` across the three current diagnostic manifests.
  - Ranker-miss/oracle-hit rows: `23`.
  - Operator-gap rows: `170`.
  - The ranker misses are usually 1-bit top/oracle differences, so future
    changes should prefer verifier/ranker features over widening boolean search.

## Resolved Local Blocker

- Teacher-gate parity is no longer allowed to pass with missing provenance.
  `scripts/rebuild_stage2_teacher_inputs.py` rebuilds
  `../data/processed/stage2_official_train_no_hard_valid.jsonl` and its
  `.provenance.json`; `scripts/audit_teacher_gate_extractor_parity.py` now
  checks the provenance output hash against the actual JSONL before rerunning
  chain search.

## Next GPU Boot

Use GPU only for vLLM generation. Do not train first.

Before starting any paid machine, run the local readiness gate and review the
JSON report:

```bash
make no-gpu-readiness
```

Proceed only when `status` is `pass`, `ready_for_gpu` is `true`, and
`known_blockers` is empty. Teacher parity `insufficient_evidence` and tracked
report drift are both hard failures.

For the research breakout path, also review:

```bash
python scripts/build_research_candidate_registry.py --check
python scripts/build_research_rescue_data.py
```

Solver-assisted and prompt-ensemble rows are diagnostics only. The submit-safe
research path is model-only adapters plus merged adapters from
`scripts/merge_lora_adapters.py`.

```bash
cd /workspace/nivida_h200_run
git pull
bash scripts/check_cloud_vllm_env.sh
python scripts/check_cloud_eval_inputs.py \
  --eval-inputs smoke_head6 \
  --candidate answer_final=artifacts/adapter_stage2_official_balanced_answer_only

EVAL_INPUTS=smoke_head6 \
ADAPTERS="answer_final=artifacts/adapter_stage2_official_balanced_answer_only" \
bash scripts/run_cloud_vllm_exact_eval_v3.sh

EVAL_INPUTS=combined_balanced_48pf,proxy_all_balanced_64pf,hard_triad_full \
bash scripts/run_cloud_vllm_exact_eval_v3.sh
```

Pull back only `data/processed/vllm_exact_eval_v3`, shut the GPU down, then
score locally:

```bash
python scripts/score_vllm_exact_eval_outputs.py \
  --predictions-root data/processed/vllm_exact_eval_v3 \
  --output-root data/processed/eval/vllm_exact_eval_v3
```

Submit only if the top candidate beats `official_balanced` overall and has no
large-family regression worse than one sample.

Research arena ranking can be run explicitly with:

```bash
python scripts/rank_research_candidates.py \
  --report official_balanced=data/processed/eval/vllm_exact_eval_v3/combined_balanced_48pf/official_balanced/report.json \
  --report answer_only_continuation=data/processed/eval/vllm_exact_eval_v3/combined_balanced_48pf/answer_final/report.json
```
