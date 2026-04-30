# Solver Breakout v2

CPU-only upper-bound and ranker-gap report for weak families.

| family | n | top1_acc | oracle@k | safe_override | ranker_miss | operator_gap | gain_ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| equation_template | 37 | 0.1622 | 0.1622 | 0.0270 | 0 | 7 | 0.0000 |
| bit_permutation | 41 | 0.3902 | 0.4146 | 0.3902 | 1 | 23 | 0.0244 |

## equation_template

- `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl`: n=`10`, top1=`0.2000`, oracle@k=`0.2000`, gain_ceiling=`0.0000`
- `data\processed\local_eval_manifests\hard_triad_full.jsonl`: n=`17`, top1=`0.1765`, oracle@k=`0.1765`, gain_ceiling=`0.0000`
- `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl`: n=`10`, top1=`0.1000`, oracle@k=`0.1000`, gain_ceiling=`0.0000`

### Ranker Miss Examples


### Operator Gap Examples

- `20e6b2d1` target="''" top="]%``" risk=`operator_gap_oracle_miss`
- `0e2d6796` target="|)]" top=")!!'" risk=`operator_gap_oracle_miss`
- `16ddcf94` target="\"^" top="*]?" risk=`operator_gap_oracle_miss`
- `01ef1e3e` target="[](" top="!<" risk=`operator_gap_oracle_miss`
- `05bd2dab` target="[))" top="''$/" risk=`operator_gap_oracle_miss`
- `0625f633` target="@@//" top="($" risk=`operator_gap_oracle_miss`
- `0e2d6796` target="|)]" top=")!!'" risk=`operator_gap_oracle_miss`

### Operator Gap Clusters

- `support_key_coverage`: `{"0.01-0.50": 2, "0.50-0.75": 3, "0.75-0.99": 2}`
- `target_literal_provenance`: `{"target_literals_seen": 7}`
- `target_expressibility`: `{"target_not_expressible": 7}`
- `query_key_seen`: `{"seen_query_key": 7}`
- `literal_reuse`: `{"literal_reuse_risk": 4, "no_literal_reuse_risk": 3}`

## bit_permutation

- `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl`: n=`8`, top1=`0.2500`, oracle@k=`0.2500`, gain_ceiling=`0.0000`
- `data\processed\local_eval_manifests\hard_triad_full.jsonl`: n=`24`, top1=`0.3333`, oracle@k=`0.3750`, gain_ceiling=`0.0417`
- `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl`: n=`9`, top1=`0.6667`, oracle@k=`0.6667`, gain_ceiling=`0.0000`

### Ranker Miss Examples

- `0e7f299d` oracle_rank=`4` target="01010010" top="01010011" family=`boolean_template`

### Operator Gap Examples

- `07e8cf66` target="10010110" top="01010100" risk=`operator_gap_oracle_miss`
- `0cb88778` target="00110000" top="00111000" risk=`operator_gap_oracle_miss`
- `1f1292d6` target="10001000" top="10001100" risk=`operator_gap_oracle_miss`
- `28feff8e` target="00000000" top="00000010" risk=`operator_gap_oracle_miss`
- `322c6169` target="01100011" top="01000011" risk=`operator_gap_oracle_miss`
- `02a66bcb` target="00011110" top="00001010" risk=`operator_gap_oracle_miss`
- `053f87d3` target="11111111" top="11011101" risk=`operator_gap_oracle_miss`
- `1298c980` target="00011010" top="10011010" risk=`operator_gap_oracle_miss`
- `00fdc0be` target="11111111" top="10111111" risk=`operator_gap_oracle_miss`
- `01248b76` target="11000101" top="10000101" risk=`operator_gap_oracle_miss`

### Operator Gap Clusters

- `top_operator_family`: `{"affine_gf2": 1, "boolean_template": 20, "rotation": 2}`
- `oracle_operator_family`: `{"unknown": 23}`
- `oracle_rank_bucket`: `{"miss": 23}`
- `support_stability`: `{"ambiguous_support_fit": 17, "stable_low_complexity": 6}`
- `top_hamming_to_target`: `{"1": 11, "2-3": 12}`
- `top_oracle_hamming`: `{"unknown": 23}`
- `top_complexity_penalty`: `{"0-0.10": 23}`
- `top_expression_complexity`: `{"0": 3, "1-4": 1, "5-8": 2, "9+": 17}`
