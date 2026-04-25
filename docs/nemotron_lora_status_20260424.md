# Nemotron LoRA Status 2026-04-24

## Leaderboard State

| adapter/submission | public score | interpretation |
| --- | ---: | --- |
| official-balanced thin continuation from B thin `20260424_161110Z` | 0.57 | current public baseline |
| B thin | 0.54 | prior best baseline |
| norm-shared top8 scale1 | 0.54 | no clear gain over B thin |
| routeweighted mixed shared | 0.53 | regression |
| A path thin | 0.51 | regression |

## Current Read

Training is useful: the official-balanced continuation moved public score from
`0.54` to `0.57`. The weak point is no longer "LoRA did not train"; it is
candidate selection and metric alignment.

The route/shared transplant line remains demoted. It has both a statistical
selection issue and a structural risk: routed expert deltas are conditional,
while `shared_experts` are dense global paths.

## Next Machine Use

Default next machine run should be inference/eval first, not training:

```bash
git pull
python scripts/build_local_eval_manifests.py
EVAL_INPUTS=smoke_6pf bash scripts/run_cloud_inference_only_v3.sh
bash scripts/run_cloud_inference_only_v3.sh
```

After predictions are copied back locally:

```bash
python scripts/score_local_eval_predictions.py
```

The scoring script now auto-selects the official-balanced adapter as baseline
when its predictions are present.

## Training Only After Ranking

If no existing checkpoint beats the official-balanced exact-eval baseline,
build answer-focused data and train one of these low-LR continuations:

```bash
python scripts/build_stage2_answer_focused_data.py
python -m src.student.lora_train --config configs/train_stage2_official_balanced_answer_only.yaml
python -m src.student.lora_train --config configs/train_stage2_official_balanced_short_trace.yaml
```

Submit only if local ranking passes the gate against official-balanced:

- higher overall official-verify accuracy
- no large family regression worse than one sample
- tie-break by boxed-valid rate, shorter output, and lower family/subtype variance
