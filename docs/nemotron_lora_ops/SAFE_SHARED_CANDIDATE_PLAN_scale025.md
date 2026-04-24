# Safe Shared Candidate Plan

This is a small candidate plan, not a submission recommendation.

- scale: `0.25`
- modules: `9`
- expert slots: `14`

| layer | module | experts | scale | top8_mass | entropy | public_leverage |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 10 | down_proj | 63 | 0.250 | 0.357 | 0.850 | 1 |
| 10 | up_proj | 39, 37 | 0.250 | 0.357 | 0.850 | 1 |
| 20 | up_proj | 23 | 0.250 | 0.379 | 0.844 | 1 |
| 24 | down_proj | 87 | 0.250 | 0.451 | 0.803 | 1 |
| 24 | up_proj | 87, 123 | 0.250 | 0.451 | 0.803 | 1 |
| 27 | down_proj | 102 | 0.250 | 0.361 | 0.857 | 0 |
| 27 | up_proj | 0, 102, 21 | 0.250 | 0.361 | 0.857 | 0 |
| 29 | down_proj | 19 | 0.250 | 0.425 | 0.811 | 1 |
| 29 | up_proj | 19, 44 | 0.250 | 0.425 | 0.811 | 1 |

Next step: build only if local per-example route v3 still supports these intersections, then run local parsed-exact inference eval before any submission.
