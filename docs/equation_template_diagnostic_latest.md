# Equation Template Diagnostic

| manifest | risk_class | n | top1_acc | oracle_at_k | unseen_literal_rows |
| --- | --- | ---: | ---: | ---: | ---: |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | operator_gap_oracle_miss | 31 | 0.0323 | 0.0323 | 0 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | ranker_miss_oracle_hit | 1 | 0.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | unseen_literal_high_risk | 7 | 0.0000 | 0.0000 | 7 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | low_risk_support_stable | 5 | 1.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | operator_gap_oracle_miss | 156 | 0.0705 | 0.0705 | 0 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | ranker_miss_oracle_hit | 1 | 0.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | unseen_literal_high_risk | 28 | 0.0000 | 0.0000 | 28 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | low_risk_support_stable | 1 | 1.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | operator_gap_oracle_miss | 36 | 0.0833 | 0.0833 | 0 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | unseen_literal_high_risk | 9 | 0.0000 | 0.0000 | 9 |

## Top Misses

- `0b0a3643` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`""` top=`-)'`
- `1cce5949` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`[?}` top=``??`
- `2ba4b99f` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`@:%>` top=`{`
- `30e4b199` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`|:|` top=`)`
- `36557a2e` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`!{}?` top=`}`
- `38c7aca1` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`!!]{` top=`!||{`
- `398478f6` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`:[` top=`@|))`
- `3cb3fd89` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`[[&]` top=`]|{\`
- `4d8df95b` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`)` top=`-?"`
- `51174a9d` risk=`unseen_literal_high_risk` oracle_rank=`None` target=`[/<` top=`$|||`
- `563bf8f9` risk=`unseen_literal_high_risk` oracle_rank=`None` target=`-'/` top=`/"!$`
- `6beb3a1f` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`??` top=`[)`
- `6c7231ac` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`:/#` top=`#`
- `6f8261d9` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`\` top=`(`
- `771472d6` risk=`ranker_miss_oracle_hit` oracle_rank=`4` target=`-'?` top=`?'\\`
- `8753cdcc` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`'""` top=`)%%`
- `898bc85a` risk=`unseen_literal_high_risk` oracle_rank=`None` target=`^[[` top=`]>[`
- `90feb0c5` risk=`unseen_literal_high_risk` oracle_rank=`None` target=`:<]` top=`?```
- `948e5474` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`-&` top=`^%?%`
- `9a4f2f47` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`('^` top=`!(}:`