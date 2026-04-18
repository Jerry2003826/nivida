# Submission Runbook

This runbook starts after canonical stage2 and stage3 have both produced their
bestproxy artifacts.

## Inputs

- `artifacts/adapter_stage2_bestproxy/`
- `data/processed/stage2_bestproxy_hard_eval.json`
- `data/processed/stage2_bestproxy_all_eval.json`
- `artifacts/adapter_stage3_bestproxy/`
- `data/processed/stage3_bestproxy_hard_eval.json`
- `data/processed/stage3_bestproxy_all_eval.json`

## 1. Select the Final Adapter

```bash
python scripts/select_final_adapter.py \
  --stage2-hard-eval data/processed/stage2_bestproxy_hard_eval.json \
  --stage2-all-eval  data/processed/stage2_bestproxy_all_eval.json \
  --stage2-adapter-dir artifacts/adapter_stage2_bestproxy \
  --stage3-hard-eval data/processed/stage3_bestproxy_hard_eval.json \
  --stage3-all-eval  data/processed/stage3_bestproxy_all_eval.json \
  --stage3-adapter-dir artifacts/adapter_stage3_bestproxy \
  --output-adapter-dir artifacts/adapter_final_selected \
  --output-json data/processed/final_adapter_selection.json
```

Result:

- weights copied to `artifacts/adapter_final_selected/`
- selection trace written to `data/processed/final_adapter_selection.json`

## 2. Validate Before Packaging

```bash
python scripts/validate_submission.py \
  --config configs/train_stage3_repair.yaml \
  --adapter-dir artifacts/adapter_final_selected \
  --smoke-input data/processed/official_train_tagged.jsonl \
  --labels data/processed/official_train_tagged.jsonl \
  --splits data/splits/official/splits.json \
  --max-new-tokens 2048 \
  --package-output submission.zip
```

This enforces:

- adapter rank is readable
- adapter rank is `<= 32`
- local smoke inference ran
- local eval had complete coverage
- package creation only happens after those gates pass

## 3. Final Outputs

- adapter: `artifacts/adapter_final_selected/`
- validation report: `artifacts/submission_validation.json`
- package: `submission.zip`

## Notes

- Always package from `artifacts/adapter_final_selected/`, never directly from
  stage2 or stage3 adapter directories.
- If stage3 was skipped by its gate, `stage3_bestproxy_*` still exists and is
  intentionally wired so final selection remains a two-way comparison.
