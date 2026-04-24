# Exact Eval Comparison

Baseline: `teacher`

## Overall

| model | n | official_verify | delta | boxed_valid | local_competition |
| --- | ---: | ---: | ---: | ---: | ---: |
| teacher | 667 | 0.9715 | +0.0000 | 0.9625 | 0.9430 |

## Family

| family | model | n | official_verify | delta | boxed_valid |
| --- | --- | ---: | ---: | ---: | ---: |
| bit | teacher | 217 | 1.0000 | +0.0000 | 1.0000 |
| cipher | teacher | 236 | 1.0000 | +0.0000 | 1.0000 |
| equation | teacher | 214 | 0.9112 | +0.0000 | 0.8832 |
