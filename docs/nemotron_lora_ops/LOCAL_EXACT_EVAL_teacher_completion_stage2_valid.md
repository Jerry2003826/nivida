# Local Exact Eval Report

- predictions: `data\processed\stage2_distill_valid.jsonl`
- labels: `data\processed\stage2_distill_valid.jsonl`
- joined rows: `667`
- official-verify accuracy: `0.9715`
- local competition accuracy: `0.9430`
- boxed-valid rate: `0.9625`

## Family

| family | n | official_verify | local_competition | local_exact | boxed_valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| bit | 217 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| cipher | 236 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| equation | 214 | 0.9112 | 0.8224 | 0.8224 | 0.8832 |

## Answer Kind

| answer_kind | n | official_verify |
| --- | ---: | ---: |
| binary | 219 | 1.0000 |
| numeric | 19 | 1.0000 |
| text_phrase | 236 | 1.0000 |
| text_token | 193 | 0.9016 |
