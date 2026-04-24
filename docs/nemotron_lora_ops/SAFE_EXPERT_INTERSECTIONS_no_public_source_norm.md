# Safe Expert Intersections: no-public source-normalized route

This is a local-only diagnostic. It does not imply a submission should be made.

## Summary

- modules checked: `46`
- modules with norm-route intersection: `37`
- recommended modules after layer filters: `9`
- recommended expert slots: `14`

Layer filters:

- route top8 mass >= `0.35`
- route normalized entropy <= `0.87`
- public visible leverage <= `1`

## Recommended Rows

| layer | module | experts | top8_mass | entropy | public_leverage |
| ---: | --- | --- | ---: | ---: | ---: |
| 10 | down_proj | 63 | 0.357 | 0.850 | 1 |
| 10 | up_proj | 39, 37 | 0.357 | 0.850 | 1 |
| 20 | up_proj | 23 | 0.379 | 0.844 | 1 |
| 24 | down_proj | 87 | 0.451 | 0.803 | 1 |
| 24 | up_proj | 87, 123 | 0.451 | 0.803 | 1 |
| 27 | down_proj | 102 | 0.361 | 0.857 | 0 |
| 27 | up_proj | 0, 102, 21 | 0.361 | 0.857 | 0 |
| 29 | down_proj | 19 | 0.425 | 0.811 | 1 |
| 29 | up_proj | 19, 44 | 0.425 | 0.811 | 1 |

## Interpretation

Use this only as a candidate filter. The next required step is local parsed-exact inference eval.
If the recommended set is small, keep it small; do not force top8 per layer.
