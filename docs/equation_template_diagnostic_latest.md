# Equation Template Diagnostic

| manifest | risk_class | n | top1_acc | oracle_at_k | target_expressible | unseen_literal_rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | expressible_oracle_miss | 4 | 0.5000 | 0.5000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | operator_gap_oracle_miss | 6 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | unseen_key_template_miss | 22 | 0.0000 | 0.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | unseen_literal_high_risk | 7 | 0.0000 | 0.0000 | 0.8571 | 7 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | expressible_oracle_miss | 20 | 0.6000 | 0.6000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | low_risk_support_stable | 5 | 1.0000 | 1.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | operator_gap_oracle_miss | 43 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | unseen_key_template_miss | 94 | 0.0000 | 0.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | unseen_literal_high_risk | 28 | 0.0000 | 0.0000 | 0.7500 | 28 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | expressible_oracle_miss | 8 | 0.3750 | 0.3750 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | low_risk_support_stable | 1 | 1.0000 | 1.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | operator_gap_oracle_miss | 9 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | unseen_key_template_miss | 19 | 0.0000 | 0.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | unseen_literal_high_risk | 9 | 0.0000 | 0.0000 | 0.7778 | 9 |

## Target Expressibility

- current ops can fit support+query target: `207 / 275`
- via `operator_template`: `192`
- via `operator_template` with query key seen in support: `22`
- via `operator_template` with query key unseen in support: `170`
- via `position_transducer`: `79`

## Top Misses

- `0b0a3643` risk=`unseen_key_template_miss` oracle_rank=`None` target=`""` top=`-)'`
- `1cce5949` risk=`unseen_key_template_miss` oracle_rank=`None` target=`[?}` top=``??`
- `2ba4b99f` risk=`unseen_key_template_miss` oracle_rank=`None` target=`@:%>` top=`{`
- `30e4b199` risk=`unseen_key_template_miss` oracle_rank=`None` target=`|:|` top=`)`
- `36557a2e` risk=`unseen_key_template_miss` oracle_rank=`None` target=`!{}?` top=`)}!}`
- `38c7aca1` risk=`unseen_key_template_miss` oracle_rank=`None` target=`!!]{` top=`!||{`
- `398478f6` risk=`unseen_key_template_miss` oracle_rank=`None` target=`:[` top=`@|))`
- `3cb3fd89` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`[[&]` top=`]|{\`
- `4d8df95b` risk=`unseen_key_template_miss` oracle_rank=`None` target=`)` top=`-?"`
- `51174a9d` risk=`unseen_literal_high_risk` oracle_rank=`None` target=`[/<` top=`$|||`
- `563bf8f9` risk=`unseen_literal_high_risk` oracle_rank=`None` target=`-'/` top=`/"!$`
- `6beb3a1f` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`??` top=`[)`
- `6c7231ac` risk=`unseen_key_template_miss` oracle_rank=`None` target=`:/#` top=`#`
- `6f8261d9` risk=`expressible_oracle_miss` oracle_rank=`None` target=`\` top=`(`
- `8753cdcc` risk=`unseen_key_template_miss` oracle_rank=`None` target=`'""` top=`)%%`
- `898bc85a` risk=`unseen_literal_high_risk` oracle_rank=`None` target=`^[[` top=`]>[`
- `90feb0c5` risk=`unseen_literal_high_risk` oracle_rank=`None` target=`:<]` top=`?```
- `948e5474` risk=`unseen_key_template_miss` oracle_rank=`None` target=`-&` top=`^%?%`
- `9a4f2f47` risk=`operator_gap_oracle_miss` oracle_rank=`None` target=`('^` top=`!(}:`
- `a3183159` risk=`unseen_key_template_miss` oracle_rank=`None` target=``{` top=`-]``