# Artifact Naming

This document is the naming contract for the current repository state. It is
not a migration plan and it does not rename the live canonical pipeline.

## Categories

| Category | Purpose | Examples |
| --- | --- | --- |
| `canonical` | The mainline stage1 -> stage2 -> stage3 training path. | `artifacts/adapter_stage1_format/`, `data/processed/stage2_distill_train.jsonl` |
| `bestproxy` | Stable adapter/eval copies chosen by the proxy selectors. | `artifacts/adapter_stage2_bestproxy/`, `data/processed/stage3_bestproxy_all_eval.json` |
| `final-selected` | The single adapter promoted for packaging. | `artifacts/adapter_final_selected/`, `data/processed/final_adapter_selection.json` |
| `branch` | Experimental sibling runs that must never overwrite canonical outputs. | `artifacts/adapter_stage2_subtype_rescue/`, `data/processed/stage2_subtype_rescue_train.jsonl` |
| `audit` | Read-only analysis helpers and checks. | `artifacts/chat_template_probe.json`, `artifacts/submission_validation.json` |
| `smoke` | Small-scope debugging or environment checks. | `scripts/train_stage1_smoke.sh`, `scripts/train_smoke_local.sh` |
| `archive` | Historical outputs that are intentionally not consumed by training. | `artifacts/_archive/` |

## Canonical Paths

### Stage1

- Adapter dir: `artifacts/adapter_stage1_format/`
- Train dataset: `data/processed/stage1_format_align_train.jsonl`
- Valid dataset: `data/processed/stage1_format_align_valid.jsonl`

### Stage2

- Training adapter dir: `artifacts/adapter_stage2_selected_trace/`
- Bestproxy adapter dir: `artifacts/adapter_stage2_bestproxy/`
- Train dataset: `data/processed/stage2_distill_train.jsonl`
- Valid dataset: `data/processed/stage2_distill_valid.jsonl`
- Proxy evals:
  - `data/processed/stage2_bestproxy_hard_eval.json`
  - `data/processed/stage2_bestproxy_all_eval.json`
- Checkpoint selection report: `data/processed/stage2_best_checkpoint_selection.json`

### Stage3

- Training adapter dir: `artifacts/adapter_stage3_repair/`
- Bestproxy adapter dir: `artifacts/adapter_stage3_bestproxy/`
- Train dataset: `data/processed/stage3_repair_train.jsonl`
- Valid dataset: `data/processed/stage3_repair_valid.jsonl`
- Proxy evals:
  - `data/processed/stage3_bestproxy_hard_eval.json`
  - `data/processed/stage3_bestproxy_all_eval.json`
- Gate / selection reports:
  - `data/processed/stage3_decision.json`
  - `data/processed/stage3_best_checkpoint_selection.json`

### Final Submission

- Selected adapter dir: `artifacts/adapter_final_selected/`
- Selection report: `data/processed/final_adapter_selection.json`
- Validation report: `artifacts/submission_validation.json`

## Branch Paths

### Stage2 Subtype Rescue

- Training adapter dir: `artifacts/adapter_stage2_subtype_rescue/`
- Bestproxy adapter dir: `artifacts/adapter_stage2_subtype_rescue_bestproxy/`
- Train dataset: `data/processed/stage2_subtype_rescue_train.jsonl`
- Valid dataset: `data/processed/stage2_subtype_rescue_valid.jsonl`
- Proxy evals:
  - `data/processed/stage2_subtype_rescue_bestproxy_hard_eval.json`
  - `data/processed/stage2_subtype_rescue_bestproxy_all_eval.json`
- Selector report: `data/processed/stage2_subtype_rescue_best_checkpoint_selection.json`

### Stage3 Subtype Rescue

This path is scaffolded but dormant until the branch promotion gate passes.

- Training adapter dir: `artifacts/adapter_stage3_subtype_rescue/`
- Bestproxy adapter dir: `artifacts/adapter_stage3_subtype_rescue_bestproxy/`
- Train dataset: `data/processed/stage3_subtype_rescue_train.jsonl`
- Valid dataset: `data/processed/stage3_subtype_rescue_valid.jsonl`
- Proxy evals:
  - `data/processed/stage3_subtype_rescue_bestproxy_hard_eval.json`
  - `data/processed/stage3_subtype_rescue_bestproxy_all_eval.json`
- Selector report: `data/processed/stage3_subtype_rescue_best_checkpoint_selection.json`

## Shared Upstream Inputs

These paths are intentionally shared between canonical and branch runs because
they are deterministic upstream inputs, not adapter outputs:

- `data/processed/official_train_tagged.jsonl`
- `data/processed/stage2_official_train_no_hard_valid.jsonl`
- `data/processed/stage2_official_valid_hard_triad.jsonl`
- `data/processed/proxy_all_family_valid.jsonl`
- `data/synthetic/synth_hard_triads.jsonl`

## Naming Rules

- Canonical and branch adapters must always differ at the directory level.
- Bestproxy artifacts must always use the `_bestproxy` suffix.
- Selector summaries live in `data/processed/*selection*.json`.
- Gate decisions live in `data/processed/*decision*.json`.
- Scripts may share deterministic inputs, but they must not share:
  - adapter output directories
  - stage train/valid dataset outputs
  - bestproxy eval outputs
  - selector reports

## Non-Goals

This document does not define:

- prompt/completion formatting contracts
- LoRA hyperparameters
- split-builder implementation details
- archive cleanup policy beyond the `_archive/` prefix
