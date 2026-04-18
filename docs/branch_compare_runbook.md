# Branch Compare Runbook

This document explains how to compare canonical stage2 against the
`subtype_rescue` branch and decide whether the branch earns a stage3.

## Baseline vs Branch Inputs

### Canonical baseline

- `data/processed/stage2_bestproxy_hard_eval.json`
- `data/processed/stage2_bestproxy_all_eval.json`
- `artifacts/adapter_stage2_bestproxy/`

### Subtype-rescue branch

- `data/processed/stage2_subtype_rescue_bestproxy_hard_eval.json`
- `data/processed/stage2_subtype_rescue_bestproxy_all_eval.json`
- `artifacts/adapter_stage2_subtype_rescue_bestproxy/`

By default, `scripts/train_stage2_subtype_rescue.sh` expects the canonical
stage2 inputs to already exist and copies them into branch-local paths before
training. It also writes:

- `data/processed/stage2_subtype_rescue_input_manifest.json`

Use that manifest to confirm the branch-local train subset, valid subset,
all-family proxy subset, and synth file came from the expected sources. Set
`ALLOW_SUBTYPE_RESCUE_REGENERATE_INPUTS=1` only for a deliberate standalone
experiment where canonical inputs are unavailable.

## Promotion Rule

The branch advances only when both conditions are true:

- `branch_all >= baseline_all - 0.5 / N_all`
- `branch_hard >= baseline_hard + 0.5 / N_hard`

Important: `N_all` and `N_hard` are read directly from each eval JSON's
`num_examples` field. You do not need to look them up in `split_report.json`.

## One-Command Decision

```bash
python scripts/decide_subtype_branch_promotion.py \
  --baseline-hard-eval data/processed/stage2_bestproxy_hard_eval.json \
  --baseline-all-eval  data/processed/stage2_bestproxy_all_eval.json \
  --branch-hard-eval   data/processed/stage2_subtype_rescue_bestproxy_hard_eval.json \
  --branch-all-eval    data/processed/stage2_subtype_rescue_bestproxy_all_eval.json \
  --output data/processed/stage2_subtype_rescue_promotion.json
```

By default this is a query tool:

- promotion success -> exit `0`
- promotion failure -> still exit `0`, but JSON says `"promote": false`

For CI / gating use:

```bash
python scripts/decide_subtype_branch_promotion.py \
  --exit-nonzero-on-no-promotion
```

## Coverage Rules

The decision script hard-fails when any compared eval has incomplete coverage.
That means any nonzero:

- `coverage.num_missing`
- `coverage.num_unexpected`
- `coverage.num_duplicate`

## Mid-Run Summary

If you want a single consolidated snapshot while only stage2 is available:

```bash
python scripts/analyze_proxy_results.py --allow-partial
```

That allows missing stage3 / branch groups, but it still hard-fails on
coverage problems in any eval file that is present.

## If the Branch Wins

Run the pre-wired stage3 scaffold:

```bash
bash scripts/train_stage3_subtype_rescue.sh
```

The scaffold reads `data/processed/stage2_subtype_rescue_promotion.json` by
default and refuses to proceed unless that JSON says `"promote": true`.
Set both `ALLOW_UNPROMOTED_SUBTYPE_STAGE3=1` and
`I_UNDERSTAND_SUBTYPE_STAGE3_WAS_NOT_PROMOTED=1` only for a deliberate manual
override.
All outputs stay on isolated branch paths and do not overwrite canonical
stage3.

## If the Branch Loses

Do not run branch stage3. Keep canonical stage3 as the only continuation path.
