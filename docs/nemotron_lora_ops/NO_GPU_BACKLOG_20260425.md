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
| Bit-permutation diagnostic | done | `python scripts/diagnose_bit_permutation.py` wrote 344 rows |
| Tokenizer-only probe | done | chat template SHA16 `ab7813c3abdd9cb6`, first public sample length 275 |
| Chat-template SHA recheck | done | `python scripts/recheck_chat_template_sha16.py --output data/processed/recheck_chat_template_sha16.json` |
| Answer-focused datasets | done | answer-only train/valid rows `4689/658`; short-trace train/valid rows `4689/658` |
| Prompt/boxed guard check | done | `sh scripts/check_prompt_suffix_alignment.sh ...` checked 10694 rows, bad `0` |
| Fast local tests | done | `23 passed, 1 skipped` for diagnostic/cloud/shell/tokenizer tests |
| Full local tests | done | `python -m pytest -q` -> `449 passed, 9 skipped` |

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

## Current Solver Read

- `equation_template` remains the highest-value CPU target.
  - Current ops can fit support+query target for `207 / 275` rows.
  - Only `22` have an operator-template fit where the query key was seen in
    support.
  - `unseen_key_template_miss` remains unsafe for strict trace training; keep
    it answer-only/silver only.
- `bit_permutation` has bounded ranker upside but a larger operator gap.
  - Low-risk top1 rows: `152`.
  - Ranker-miss/oracle-hit rows: `40`.
  - Operator-gap rows: `151`.
  - The ranker misses are usually 1-bit top/oracle differences, so future
    changes should prefer verifier/ranker features over widening boolean search.

## Blocked Or Incomplete Without Better Local Artifacts

- `scripts/audit_teacher_gate_extractor_parity.py` cannot fully audit the
  current parent `stage2_official_train_no_hard_valid.jsonl` because cached
  `support_pairs/query_prediction` are absent and the expected provenance file
  is missing:

```json
{
  "status": "insufficient_evidence",
  "reason": "stage2 provenance missing or mismatched",
  "required": {
    "provenance_path": "..\\data\\processed\\stage2_official_train_no_hard_valid.jsonl.provenance.json"
  },
  "found": null
}
```

Do not treat this audit as passed until the stage2 annotation provenance is
available or the dataset is rebuilt with provenance.

## Next GPU Boot

Use GPU only for vLLM generation. Do not train first.

```bash
cd /workspace/nivida_h200_run
git pull
test -f artifacts/adapter_stage2_official_balanced_answer_only/adapter_model.safetensors
bash scripts/check_cloud_vllm_env.sh

head -n 6 data/processed/local_eval_manifests/smoke_6pf.jsonl \
  > data/processed/local_eval_manifests/smoke_head6.jsonl

EVAL_INPUTS=smoke_head6 \
ADAPTERS="answer_final=artifacts/adapter_stage2_official_balanced_answer_only" \
bash scripts/run_cloud_vllm_exact_eval_v3.sh

EVAL_INPUTS=combined_balanced_48pf,proxy_all_balanced_64pf,hard_triad_full \
ADAPTERS="b_thin=artifacts/adapter_stage2_thin,official_balanced=artifacts/adapter_stage2_thin_official_balanced_20260424_161110Z,answer_final=artifacts/adapter_stage2_official_balanced_answer_only,answer_ckpt_100=artifacts/adapter_stage2_official_balanced_answer_only/checkpoint-100,answer_ckpt_200=artifacts/adapter_stage2_official_balanced_answer_only/checkpoint-200,answer_ckpt_294=artifacts/adapter_stage2_official_balanced_answer_only/checkpoint-294" \
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
