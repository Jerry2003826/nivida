# Blackwell-Managed Training (2026-04 recovery)

This is the canonical single-branch pipeline for a fresh RTX PRO 6000 Blackwell
(96 GB) or H100/H200 (80 GB+) machine after the April 2026 Mamba-`in_proj`
overflow post-mortem.  It replaces all earlier per-stage runbooks.

## Why this branch exists

On the original H200 run the wide LoRA target pattern

```
target_modules: ".*\\.(in_proj|out_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)$"
```

applied to Nemotron-3-Nano-30B caused gradient overflow in bf16 on the Mamba
`in_proj` projection (output dim 20480 per Mamba block across 23 Mamba layers).
Symptoms:

- stage2 `grad_norm = inf`, `clip_coefficient ~ 1e-34`, train_loss ~ 41-48,
  eval_loss frozen at 2.2251603603363037 across every checkpoint.
- stage1 with the wider target pattern silently showed the same pathology; the
  only usable stage1 artifact on H200 was the **narrow** variant
  (`out_proj/up_proj/down_proj` only, no `in_proj`, eval_loss 2.6176).

## Fix applied

Three edits, applied uniformly to stage1 / stage1 smoke / stage2 / stage3 /
legacy `train_lora*.yaml` configs:

1. Drop `in_proj` from `target_modules`.  Keep Mamba `out_proj`, MoE
   `up_proj` / `down_proj`, and attention `q/k/v/o_proj`.
2. `alpha: 64` -> `alpha: 32` (LoRA scaling ratio = 1.0 at rank 32).
3. Stage2 LR 2.0e-5 -> 3.0e-5 and stage3 LR 5e-5 -> 3.0e-5
   (cosine, warmup_ratio=0.05, max_grad_norm=1.0) so all three stages share a
   single healthy optimiser regime.

`src/student/inference.py` now accepts `--official-eval` to mirror the official
Kaggle leaderboard vLLM config (do_sample=True, T=1.0, top_p=1.0,
max_new_tokens=3584, chat_thinking prompt).

`scripts/check_smoke_health.py` reads the HF Trainer `trainer_state.json` and
fails fast if grad_norm is non-finite or above 1e10, or if loss exceeds 10.0.
This is the tripwire that would have caught the stage2 overflow on day one.

`scripts/setup_new_machine.sh` is the single-command entrypoint that chains
bootstrap -> probe -> data prep -> smoke -> stage1 -> acceptance ->
stage2 -> stage3 -> select final -> validate + package submission.zip.

## Running on a fresh machine

```bash
# 1. SSH in, land in $WORKSPACE_DIR (example uses the H200 path; any workspace
#    volume works as long as KAGGLEHUB_CACHE points at it).
mkdir -p /workspace/nivida_h200_run && cd /workspace/nivida_h200_run

# 2. Clone + switch to this branch
git clone https://github.com/Jerry2003826/nivida.git .
git checkout codex/blackwell-managed-training

# 3. Export Kaggle credentials so kagglehub can download the base model
export KAGGLE_USERNAME="<your kaggle username>"
export KAGGLE_KEY="<your kaggle api key>"

# 4. Run the whole pipeline (bootstrap + smoke + stage1/2/3 + submission)
bash scripts/setup_new_machine.sh
```

The script is idempotent.  Re-running it after a crash resumes from the last
stage whose canonical artifact is missing.

## Expected artifacts

| Stage | Canonical output | Gate |
|-------|------------------|------|
| smoke | `artifacts/smoke/adapter_stage1_format/` + `health_report.json` | `check_smoke_health.py` (no inf grad, loss<10) |
| stage1 | `artifacts/adapter_stage1_format/` | `check_stage1_acceptance.py` (4 required files, chat_template sha16 match) |
| stage2 | `artifacts/adapter_stage2_bestproxy/` + `stage2_bestproxy_*_eval.json` | bestproxy all-family + hard-triad eval |
| stage3 | `artifacts/adapter_stage3_bestproxy/` + `stage3_bestproxy_*_eval.json` | bestproxy all-family + hard-triad eval |
| final | `artifacts/adapter_final/` + `final_adapter_selection.json` | stage2 vs stage3 decision rule |
| submission | `submission.zip` + `artifacts/submission_validation.json` | size <1 GB, adapter loadable |

## Rollback

If the Blackwell fixes cause a new regression, roll back by:

```bash
git checkout codex/h200-recovery-fixes
```

but be aware that branch still has the broken wide-target stage2/3 configs.
