# Equation Template Diagnostic

| manifest | risk_class | n | top1_acc | oracle_at_k | target_expressible | unseen_literal_rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | expressible_oracle_miss | 6 | 0.5000 | 0.5000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | low_risk_support_stable | 1 | 1.0000 | 1.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | operator_gap_oracle_miss | 8 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | ranker_miss_oracle_hit | 1 | 0.0000 | 1.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | unseen_key_template_miss | 14 | 0.0000 | 0.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | unseen_literal_high_risk | 3 | 0.0000 | 0.0000 | 0.6667 | 3 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | expressible_oracle_miss | 10 | 0.8000 | 0.8000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | low_risk_support_stable | 4 | 1.0000 | 1.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | operator_gap_oracle_miss | 28 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | ranker_miss_oracle_hit | 2 | 0.0000 | 1.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | unseen_key_template_miss | 92 | 0.0000 | 0.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | unseen_literal_high_risk | 20 | 0.0000 | 0.0000 | 0.7500 | 20 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | expressible_oracle_miss | 8 | 0.3750 | 0.3750 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | low_risk_support_stable | 1 | 1.0000 | 1.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | operator_gap_oracle_miss | 9 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | unseen_key_template_miss | 19 | 0.0000 | 0.0000 | 1.0000 | 0 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | unseen_literal_high_risk | 9 | 0.0000 | 0.0000 | 0.7778 | 9 |

## Target Expressibility

- current ops can fit support+query target: `182 / 235`
- via `operator_template`: `172`
- via `operator_template` with query key seen in support: `21`
- via `operator_template` with query key unseen in support: `151`
- via `position_transducer`: `69`

## Top Misses

- `0dce4039` risk=`unseen_key_template_miss` oracle_rank=`None` query=`)!-#>` target=`-@/` top=`)!#>` target_expressible=`True`
- `196ff375` risk=`unseen_key_template_miss` oracle_rank=`None` query=`%^\#@` target=`@@@` top=`@^@` target_expressible=`True`
- `1a28140b` risk=`expressible_oracle_miss` oracle_rank=`None` query=`"[*#/` target=`%`&` top=```` target_expressible=`True`
- `20e6b2d1` risk=`operator_gap_oracle_miss` oracle_rank=`None` query=`@`-](` target=`''` top=`]%``` target_expressible=`False`
- `24e1f1d5` risk=`unseen_key_template_miss` oracle_rank=`None` query=`^#+(!` target=`^#(!` top=`^"#` target_expressible=`True`
- `25ee72c3` risk=`expressible_oracle_miss` oracle_rank=`None` query=`#"*<<` target=`#"<<` top=`#]` target_expressible=`True`
- `2ba4b99f` risk=`unseen_key_template_miss` oracle_rank=`None` query=`@|(\|` target=`@:%>` top=`{` target_expressible=`True`
- `2d89386e` risk=`expressible_oracle_miss` oracle_rank=`None` query=`%]-$]` target=`:` top=`%<` target_expressible=`True`
- `3d2cb38a` risk=`operator_gap_oracle_miss` oracle_rank=`None` query=`<}+#)` target=`}{` top=`//#}` target_expressible=`False`
- `5690981d` risk=`unseen_key_template_miss` oracle_rank=`None` query=`(`-&%` target=`-](` top=`||` target_expressible=`True`
- `685be3a7` risk=`operator_gap_oracle_miss` oracle_rank=`None` query=`!?+`:` target=`>$$` top=`$:&!` target_expressible=`False`
- `6c7f24b7` risk=`ranker_miss_oracle_hit` oracle_rank=`2` query=`$'-^$` target=`^` top=`$` target_expressible=`True`
- `802c3591` risk=`operator_gap_oracle_miss` oracle_rank=`None` query=`)?<#`` target=``"""` top=`)?'@` target_expressible=`False`
- `8326116b` risk=`unseen_literal_high_risk` oracle_rank=`None` query=`#:*#\` target=`{{@?` top=`{#` target_expressible=`False`
- `852a6f48` risk=`unseen_literal_high_risk` oracle_rank=`None` query=`]){&$` target=`\\` top=`&&')` target_expressible=`True`
- `94367b1d` risk=`unseen_key_template_miss` oracle_rank=`None` query=`[[+|<` target=`|'` top=`[{[` target_expressible=`True`
- `9925be81` risk=`unseen_literal_high_risk` oracle_rank=`None` query=`]#[^)` target=`@}])` top=`>` target_expressible=`True`
- `9bfca34c` risk=`unseen_key_template_miss` oracle_rank=`None` query=`<{$>^` target=`|>`` top=`||` target_expressible=`True`
- `9dae880f` risk=`operator_gap_oracle_miss` oracle_rank=`None` query=`:<}`\` target=`))[` top=`<}` target_expressible=`False`
- `a2b66927` risk=`unseen_key_template_miss` oracle_rank=`None` query=`)@-$|` target=`{]` top=`{@` target_expressible=`True`