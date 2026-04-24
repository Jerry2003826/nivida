# Selection Stability Summary

This is a local-only route selection audit. It does not recommend a submission by itself.

## Effective Raw-Count Shares

| source | nominal_weight | raw_count_share | rows | max_new_tokens |
| --- | ---: | ---: | ---: | ---: |
| public_visible | 0.25 | 0.5165 | 3 | 192 |
| official_hard | 0.30 | 0.1782 | 256 | 0 |
| official_all | 0.30 | 0.1409 | 256 | 0 |
| stage2_train | 0.15 | 0.1645 | 512 | 0 |

## Pair Summaries

| comparison | mean_overlap | min | max | mean_jaccard | mean_jsd | overlap_hist |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| raw_mixed vs source_normalized_mixed | 6.83 / 8 | 4 | 8 | 0.758 | 0.0055 | `{"4": 1, "6": 5, "7": 13, "8": 4}` |
| raw_mixed vs no_public_source_normalized | 6.04 / 8 | 3 | 7 | 0.623 | 0.0212 | `{"3": 1, "4": 1, "5": 3, "6": 9, "7": 9}` |
| source_normalized_mixed vs no_public_source_normalized | 7.13 / 8 | 6 | 8 | 0.814 | 0.0054 | `{"6": 4, "7": 12, "8": 7}` |
| public_visible vs official_all | 3.48 / 8 | 1 | 6 | 0.297 | 0.1301 | `{"1": 2, "2": 5, "3": 5, "4": 5, "5": 3, "6": 3}` |
| public_visible vs official_hard | 5.65 / 8 | 2 | 7 | 0.564 | 0.0485 | `{"2": 1, "4": 2, "5": 5, "6": 10, "7": 5}` |
| official_hard vs official_all | 4.35 / 8 | 2 | 6 | 0.385 | 0.0732 | `{"2": 1, "3": 5, "4": 5, "5": 9, "6": 3}` |

## Source Leverage

Leverage is `8 - overlap(full_mix_top8, remove_source_top8)`.

| mix | removed_source | mean_leverage | max_leverage | leverage_hist |
| --- | --- | ---: | ---: | --- |
| raw_mixed | public_visible | 1.83 | 5 | `{"0": 1, "1": 9, "2": 9, "3": 2, "4": 1, "5": 1}` |
| raw_mixed | official_hard | 0.35 | 2 | `{"0": 16, "1": 6, "2": 1}` |
| raw_mixed | official_all | 0.48 | 2 | `{"0": 13, "1": 9, "2": 1}` |
| raw_mixed | stage2_train | 0.26 | 2 | `{"0": 18, "1": 4, "2": 1}` |
| source_normalized_mixed | public_visible | 0.87 | 2 | `{"0": 7, "1": 12, "2": 4}` |
| source_normalized_mixed | official_hard | 0.83 | 2 | `{"0": 8, "1": 11, "2": 4}` |
| source_normalized_mixed | official_all | 1.00 | 3 | `{"0": 4, "1": 16, "2": 2, "3": 1}` |
| source_normalized_mixed | stage2_train | 0.17 | 1 | `{"0": 19, "1": 4}` |

## Safe Intersection Filter

- modules checked: `46`
- modules with norm-route intersection: `37`
- recommended modules after filters: `9`
- recommended expert slots: `14`

Recommended rows are documented in `SAFE_EXPERT_INTERSECTIONS_no_public_source_norm.md`.

## Bottom Line

Raw-count mixing materially overweights visible public in route counts, but source-normalizing does not fully flip top8 selection. The safer interpretation is a combined problem: route-count weighting is misaligned with the intended source mixture, and routed-expert-to-shared-expert transplant remains structurally risky.
