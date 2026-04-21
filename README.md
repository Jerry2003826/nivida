# NVIDIA Nemotron Reasoning Challenge Repo

This repo is organized around a three-stage single-LoRA competition pipeline:

- harness-aligned `chat_thinking` prompts
- local competition-proxy evaluation with `competition_correct`
- staged fine-tuning: stage1 format align -> stage2 distill -> stage3 repair

## Canonical Training Order

Run these in order before packaging a submission:

```bash
python scripts/probe_chat_template.py
python scripts/inspect_target_modules.py --config configs/train_stage1_format.yaml

bash scripts/train_stage1_format_align.sh
bash scripts/train_stage2_distill.sh
bash scripts/train_stage3_repair.sh

python scripts/select_final_adapter.py \
  --stage2-hard-eval data/processed/stage2_bestproxy_hard_eval.json \
  --stage2-all-eval  data/processed/stage2_bestproxy_all_eval.json \
  --stage2-adapter-dir artifacts/adapter_stage2_bestproxy \
  --stage3-hard-eval data/processed/stage3_bestproxy_hard_eval.json \
  --stage3-all-eval  data/processed/stage3_bestproxy_all_eval.json \
  --stage3-adapter-dir artifacts/adapter_stage3_bestproxy \
  --output-adapter-dir artifacts/adapter_final_selected \
  --output-json        data/processed/final_adapter_selection.json

python scripts/validate_submission.py \
  --config configs/train_stage3_repair.yaml \
  --adapter-dir artifacts/adapter_final_selected \
  --smoke-input data/processed/official_train_tagged.jsonl \
  --labels data/processed/official_train_tagged.jsonl \
  --splits data/splits/official/splits.json \
  --max-new-tokens 2048 \
  --package-output submission.zip
```

For local post-training acceptance, the recommended shortcut is:

```bash
make final-acceptance
```

This runs the canonical local closeout chain
(`select_final_adapter -> trained probe -> validate_submission -> package`)
against already-downloaded stage2/stage3 bestproxy artifacts. It does not
download cloud outputs or run training.

## Data Preparation

Prepare the canonical official dataset and maintained splits:

```bash
python scripts/prepare_data.py --config configs/data_official.yaml
```

This writes:

- `data/processed/official_train_tagged.jsonl`
- `data/splits/official/splits.json`

Generate hard-triad synthetic data for stage2:

```bash
python -m src.teacher.synth_generator --config configs/synth_hard_triads.yaml
```

## SFT Dataset Builders

Stage1 build:

```bash
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/stage1_format_align_train.jsonl \
  --selection-profile stage1 \
  --prompt-mode chat_thinking \
  --tokenizer-path artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default \
  --split-file data/splits/official/splits.json \
  --split-name rule_novelty_all \
  --split-role train \
  --exclude-split-file data/splits/official/splits.json \
  --exclude-split-name hard_triad_rule_novelty \
  --exclude-split-role valid
```

Because `rule_novelty_all` and `hard_triad_rule_novelty` are built
independently (different seeds, different subsets), stage1 train explicitly
excludes `hard_triad_rule_novelty/valid` so later hard-triad validation stays
unseen across the full staged pipeline (`stage1 -> stage2 init_adapter_dir=stage1 -> stage3 init_adapter_dir=stage2`).

Stage2 build with family balancing and stronger teacher search:

For reproducibility, prefer the canonical shell script:

```bash
bash scripts/train_stage2_distill.sh
```

Manual builder commands are intentionally omitted here. The canonical stage2
script also materialises the leak-free official subset, applies hard-triad
oversampling, enables the silver official pool, runs second-pass rescue, and
finishes with proxy eval plus bestproxy selection. Use
`bash scripts/train_stage2_distill.sh` unless you are deliberately debugging a
single preprocessing step.

Stage3 build from stage2 model failures plus all-family replay:

```bash
python -m src.student.sft_dataset_builder \
  --input data/processed/official_train_tagged.jsonl \
  --output data/processed/stage3_repair_train.jsonl \
  --selection-profile stage3 \
  --prompt-mode chat_thinking \
  --tokenizer-path artifacts/_tokenizer_cache/metric_nemotron-3-nano-30b-a3b-bf16_transformers_default \
  --completion-style short_trace \
  --beam-width 8 \
  --max-depth 2 \
  --top-k 2 \
  --repair-artifact data/processed/stage2_model_failures_train.json \
  --replay-input data/processed/stage2_model_successes_all_train.json \
  --replay-ratio 0.25 \
  --report-output data/processed/stage3_repair_train_report.json
```

Stage3 failure / success buckets are produced by `scripts/train_stage3_repair.sh`:

- repair failures come from `hard_triad_rule_novelty/train` - the stage2 adapter
  predictions that missed on the hard triad, the only samples we want to correct
- replay successes come from `rule_novelty_all/train` - the stage2 adapter
  predictions that were correct across all six families, so replay can keep
  easy-triad families anchored and avoid catastrophic forgetting
- the builder receives `data/processed/official_train_tagged.jsonl` as input so
  `build_repair_set` can materialise replay records for ids that live outside
  the hard-triad subset

## Training Notes

- canonical train configs are:
  - `configs/train_stage1_format.yaml`
  - `configs/train_stage2_selected_trace.yaml`
  - `configs/train_stage3_repair.yaml`
- stage2 and stage3 continue from the previous stage adapter via `training.init_adapter_dir`
- `lora_train.py` supports `--force-train`; canonical stage scripts pass it explicitly
- `target_modules` are hard-validated before training starts
- `max_seq_length: auto` uses tokenizer-aware BPE accounting when a tokenizer path is configured
- stage2 / stage3 local inference defaults to `max_new_tokens: 2048` to avoid truncating `chat_thinking` generations
- `max_depth=3` is intentionally more expensive for stage2 selection; expect noticeably higher CPU time than `max_depth=2`
- `stage2_distill_valid.jsonl` is the SFT **loss monitor** (teacher-solvable subset); it is not the hard-triad headline metric. Trust the proxy eval artifacts below instead.
- after stage2 and stage3 training the canonical scripts run a per-stage **bestproxy selector** (`scripts/select_best_proxy_checkpoint.py`). It iterates over every ``checkpoint-*`` plus the final adapter, scores each against hard-triad and all-family proxies, and writes the winner to `artifacts/adapter_stage{2,3}_bestproxy/`. The canonical artifact pairs consumed by `select_final_adapter.py` are:
  - hard-triad proxy: `data/processed/stage{2,3}_bestproxy_hard_eval.json`
  - all-family proxy (leak-free, `rule_novelty_all/valid` minus `hard_triad_rule_novelty/train`): `data/processed/stage{2,3}_bestproxy_all_eval.json`
  The stage-root proxy eval artifacts still exist (`stage{2,3}_proxy_valid_eval.json` / `stage{2,3}_proxy_all_valid_eval.json`) as monitoring signals for comparing "final step" vs "bestproxy", but the submission path reads the bestproxy pair.
- final adapter selection is **automated** by `scripts/select_final_adapter.py`: it compares the two bestproxy pairs at half-sample tolerance (primary: all-family; tie-break: hard-triad; default on complete tie: stage2) and copies the winner into `artifacts/adapter_final_selected/`. It also writes `data/processed/final_adapter_selection.json` with the full decision trace. **Package submission.zip from `artifacts/adapter_final_selected/`, not from the per-stage adapter directories.**
- stage2 enables a **silver hard-triad official pool** by default (`--stage2-enable-silver-official`): official hard-triad (bit/cipher/equation) samples that fail the strict gate but satisfy weaker thresholds (teacher_confidence >= 0.65, support_coverage >= 0.67) are admitted to the train set with `trace_style=answer_only`, capped at `min(0.25 * (strict + synth), 800)` and sampled `equation -> cipher -> bit`. The stage2 build report now carries `selection_counts` and `official_rejection_diagnostics` so the next iteration can decide whether to tune the silver thresholds.
- stage2 train also runs a **hard-triad rescue search** (`--stage2-second-pass-hard-triad`, default opt-in only at the shell layer). For every rejected official sample whose family is in `--stage2-rescue-families` (default `equation`, the family most consistently missing `program_signature` in diagnostics), a second chain-search pass is run with `beam=12/depth=4/top-k=3`. The new annotation is promoted only if the quality tuple (`solver_verifiable`, `support_coverage`, `has_signature`, `teacher_confidence`, `top1_top2_margin`) strictly improves; otherwise the first-pass annotation is restored. Rescue cannot degrade any candidate. `stage2_distill_report.json` now carries `rescue_diagnostics.{rescue_attempted, rescue_promoted, rescue_families, rescue_settings}` so you can tell how many rejected equation samples were pulled back into strict-quality annotation.
- `configs/synth_hard_triads.yaml` (and the legacy `configs/synth.yaml`) set `hard_negative_ratio: 0.0` because the `hard_negative` / `negative_answer` metadata fields have no downstream consumer in the current training pipeline. Re-enable only when a real consumer (loss term, filter, etc.) lands.
- stage3 can **skip itself** when stage2 produced zero hard-triad train failures. In that case `scripts/train_stage3_repair.sh` copies the stage2 bestproxy adapter to `artifacts/adapter_stage3_repair/` and `artifacts/adapter_stage3_bestproxy/`, reuses the stage2 bestproxy eval JSONs, and writes `stage3_skipped.json` next to the weights so downstream packaging / validation does not need to branch. See `data/processed/stage3_decision.json` and `data/processed/stage3_best_checkpoint_selection.json` for the gate outcome.

### Submission size budget

`src/student/adapter_submission_budget.py` is the single source of truth
for canonical `target_modules` regexes. Three named variants exist; all
docs, tests and configs must import by name rather than hardcoding
strings:

- `DEMO_BASELINE_TARGET_REGEX` - `(in_proj|out_proj|up_proj|down_proj)`.
  Legacy demo default; retained only for reproducing old runs.
- `WIDE_WITH_IN_PROJ_TARGET_REGEX` - legacy "submission safe wide"
  target that includes `in_proj` + attention. Observed to overflow on
  Blackwell (RF-2 audit). Do NOT use in production.
- `BLACKWELL_NARROW_TARGET_REGEX` - production target shipped with
  `configs/train_stage{1,2,3}*.yaml`. Drops `in_proj` because the Mamba
  in_proj matmul produces non-finite gradients under bf16 at alpha=32,
  rank=32.

Production regex (`BLACKWELL_NARROW_TARGET_REGEX`):

```text
.*\.(out_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)$
```

This regex covers the real LoRA-compatible Nemotron-H Linear modules we
currently rely on:

- Mamba: `out_proj` (in_proj excluded on Blackwell)
- MoE experts: `up_proj`, `down_proj`
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`

The current HF Nemotron-H implementation does not expose a `gate_proj`
Linear. `NemotronHMOE.gate` is a router module, not a LoRA-compatible
Linear, so `gate_proj` is not a valid `target_modules` entry for this
budget model. If a future model revision introduces a real `gate_proj`
Linear, extend `src/student/adapter_submission_budget.py` only after
proving the module exists with `scripts/list_model_linear_modules.py`
against the actual model source.

The MoE part of the estimator intentionally distinguishes routed and
shared experts. Per MoE layer, Nemotron-3 Nano 30B has `128` routed
experts at `moe_intermediate_size=1856` plus `1` shared expert at
`moe_shared_expert_intermediate_size=3712`, so the canonical rank-32
projection is:

- projected adapter bytes: `3,535,495,168`
- projected submission zip: `888,873,792`
- remaining headroom to Kaggle's 1GB limit: about `111.1MB`

The physical artifact model matters: peft saves LoRA adapter weights as
new float32 trainable tensors (`lora_A` / `lora_B`), not at the base
model's bf16 dtype. The estimator therefore uses
`ADAPTER_WEIGHT_BYTES = 4` plus a conservative `ZIP_DEFLATED`
compression ratio of `0.25`.

The probe exposes three mutually-exclusive modes so the provenance of
any observed compression ratio is unambiguous:

1. `--tiny-mode` - self-contained formula smoke test against
   `TINY_NEMOTRON_LIKE_PAYLOAD`. The `lora_B` tensors are zero-filled
   numpy arrays. The probe writes `tiny_mode_is_not_calibration: True`
   and emits neither compression-ratio key. This is the cheapest mode
   and runs in CI.
2. Fresh PEFT structure (default, no flag) - materialises a real
   Nemotron base model, wraps it with `peft.get_peft_model`, runs one
   forward + backward pass to allocate shapes, and saves the artifact.
   `optimizer.step()` is **not** called, so `lora_B` remains at PEFT's
   standard zero init. The output key is
   `fresh_peft_structure_compression_ratio` plus
   `peft_weights_are_fresh: True` and
   `fresh_peft_structure_is_not_calibration: True`. This mode only
   validates key paths, dtypes, and PEFT serialisation overhead - its
   zip ratio is a lower bound, not a calibration.
3. `--adapter-dir artifacts/adapter_final_selected` - the only real
   calibration mode. The probe reads an already-trained adapter
   directory (`adapter_model.safetensors` + `adapter_config.json`),
   requires the artifact's `rank/target_modules` to match the requested
   `--config`, re-runs `ensure_submission_budget_safe` against those,
   and then hard-fails unless the measured safetensors bytes/module
   counts match the canonical formula exactly. The trained-mode payload
   records `requested_rank`, `requested_selected_suffixes`, and
   `artifact_matches_requested_config` alongside
   `real_trained_adapter_archive_ratio` plus
   `real_trained_adapter_weight_compression_ratio`; the legacy
   `real_trained_adapter_compression_ratio` is retained as a backwards-
   compatible alias of the archive ratio. Every probe run also emits
   `lora_b_zero_fraction` and `lora_b_likely_untrained` so the
   provenance check is machine-readable.

The headline `NEMOTRON_ZIP_COMPRESSION_RATIO = 0.25` must be reconciled
against a `--adapter-dir` run on the actual trained adapter before the
final submission is packaged. Fresh-structure and tiny modes are
intentionally insufficient for this reconciliation.

Three guards enforce the budget end to end:

1. `lora_train.validate_lora_config` rejects target modules whose
   projected zip exceeds 1GB before training starts.
2. `scripts/validate_submission.py` re-reads the trained
   `adapter_config.json` and runs the budget check again before
   packaging.
3. `src/student/package_submission.build_submission_zip` reads the
   final `submission.zip` size from disk and hard-fails if the physical
   archive exceeds 1GB.

To run each probe mode:

```bash
# Formula smoke (no GPU, no download):
make probe-submission-size
# Calibration against the trained adapter (must be run post-training):
make probe-submission-size-trained ADAPTER_DIR=artifacts/adapter_final_selected
```

Both commands write `artifacts/adapter_submission_probe.json`. Rerun
the `--adapter-dir` probe whenever any of the following changes:

- PEFT / Transformers / Safetensors serialization behavior
- Torch or dtype policy that could change saved LoRA tensor dtypes
- Model geometry or target-module selection, including routed/shared MoE widths
- Packaging behavior, such as extra files or zip compression settings

## Smoke and Validation

H100 smoke path:

```bash
bash scripts/train_stage1_smoke.sh
```

Legacy local smoke entry now forwards to the canonical stage1 smoke script:

```bash
bash scripts/train_smoke_local.sh
```

Inside each stage, `scripts/select_best_proxy_checkpoint.py` iterates over
every ``checkpoint-*`` directory plus the final adapter, runs the hard-triad
and all-family proxy evals, and materialises the winner at
``artifacts/adapter_stage{2,3}_bestproxy``. The selector uses the shared
rule from ``src/student/proxy_selection.py`` (all-family primary,
hard-triad tie-break, prefer the final checkpoint on complete tie) so the
same comparison logic is used at both the checkpoint level and the
stage2-vs-stage3 level. Expect roughly **1.5-2.5h extra H100 time per stage
for the selector passes**; budget 3-5h total on top of the 24-36h training
run.

The validator hard-fails when:

- `adapter_config.json` is missing or its rank cannot be parsed
- the parsed rank exceeds 32
- `--labels` is passed without `--smoke-input`, or `--package-output` is passed
  without both `--smoke-input` and `--labels`
- `--package-output` runs before a successful `local_eval`

`--max-new-tokens` is optional; it overrides the inference token budget from
the config (defaults to 2048 for `chat_thinking` in stage2 / stage3 configs).

## Tests

```bash
python -m pytest -q
```
