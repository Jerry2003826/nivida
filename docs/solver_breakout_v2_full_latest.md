# Solver Breakout v2

CPU-only upper-bound and ranker-gap report for weak families.

| family | n | top1_acc | oracle@k | safe_override | ranker_miss | operator_gap | gain_ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| equation_template | 235 | 0.0936 | 0.0979 | 0.0255 | 1 | 45 | 0.0043 |
| bit_permutation | 345 | 0.4203 | 0.5159 | 0.4203 | 33 | 161 | 0.0957 |

## equation_template

- `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl`: n=`33`, top1=`0.1515`, oracle@k=`0.1515`, gain_ceiling=`0.0000`
- `data\processed\local_eval_manifests\hard_triad_full.jsonl`: n=`156`, top1=`0.0833`, oracle@k=`0.0897`, gain_ceiling=`0.0064`
- `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl`: n=`46`, top1=`0.0870`, oracle@k=`0.0870`, gain_ceiling=`0.0000`

### Ranker Miss Examples

- `8d9a062b` oracle_rank=`2` target="^!" top="^!^" family=``

### Operator Gap Examples

- `20e6b2d1` target="''" top="]%``" risk=`operator_gap_oracle_miss`
- `3d2cb38a` target="}{" top="//#}" risk=`operator_gap_oracle_miss`
- `685be3a7` target=">$$" top="$:&!" risk=`operator_gap_oracle_miss`
- `802c3591` target="`\"\"\"" top=")?'@" risk=`operator_gap_oracle_miss`
- `9dae880f` target="))[" top="<}" risk=`operator_gap_oracle_miss`
- `a2ca3aae` target="\\<?/" top=")[[" risk=`operator_gap_oracle_miss`
- `b0206bb7` target="]$!" top="&#$#" risk=`operator_gap_oracle_miss`
- `fbd5fe63` target="$[>" top="^<" risk=`operator_gap_oracle_miss`
- `0e2d6796` target="|)]" top=")!!'" risk=`operator_gap_oracle_miss`
- `16ddcf94` target="\"^" top="*]?" risk=`operator_gap_oracle_miss`

### Operator Gap Clusters

- `support_key_coverage`: `{"0.01-0.50": 11, "0.50-0.75": 19, "0.75-0.99": 14, "1.00": 1}`
- `target_literal_provenance`: `{"target_literals_seen": 45}`
- `target_expressibility`: `{"target_not_expressible": 45}`
- `query_key_seen`: `{"seen_query_key": 45}`
- `literal_reuse`: `{"literal_reuse_risk": 23, "no_literal_reuse_risk": 22}`

## bit_permutation

- `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl`: n=`48`, top1=`0.4375`, oracle@k=`0.5000`, gain_ceiling=`0.0625`
- `data\processed\local_eval_manifests\hard_triad_full.jsonl`: n=`240`, top1=`0.3958`, oracle@k=`0.5125`, gain_ceiling=`0.1167`
- `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl`: n=`57`, top1=`0.5088`, oracle@k=`0.5439`, gain_ceiling=`0.0351`

### Ranker Miss Examples

- `455b6b61` oracle_rank=`2` target="00111011" top="00101011" family=`boolean_template`
- `54ca9d57` oracle_rank=`2` target="01011111" top="11011111" family=`affine_gf2`
- `81323d52` oracle_rank=`2` target="10110001" top="10111001" family=`boolean_template`
- `3e847951` oracle_rank=`2` target="10110010" top="10110011" family=`boolean_template`
- `a897b8bc` oracle_rank=`4` target="01111101" top="01111111" family=`boolean_template`
- `0e7f299d` oracle_rank=`4` target="01010010" top="01010011" family=`boolean_template`
- `1d930d32` oracle_rank=`2` target="00000100" top="00000000" family=`boolean_template`
- `1fe9b923` oracle_rank=`5` target="00000001" top="10000001" family=`boolean_template`
- `3131bfb3` oracle_rank=`4` target="11111101" top="11111100" family=`boolean_template`
- `37c94738` oracle_rank=`4` target="00000000" top="00001000" family=`boolean_template`

### Operator Gap Examples

- `07e8cf66` target="10010110" top="01010100" risk=`operator_gap_oracle_miss`
- `0cb88778` target="00110000" top="00111000" risk=`operator_gap_oracle_miss`
- `1f1292d6` target="10001000" top="10001100" risk=`operator_gap_oracle_miss`
- `28feff8e` target="00000000" top="00000010" risk=`operator_gap_oracle_miss`
- `322c6169` target="01100011" top="01000011" risk=`operator_gap_oracle_miss`
- `431b5993` target="11010000" top="11011000" risk=`operator_gap_oracle_miss`
- `47a5c4f4` target="11100001" top="00100000" risk=`operator_gap_oracle_miss`
- `5570b0a1` target="10011101" top="11011101" risk=`operator_gap_oracle_miss`
- `59c78e51` target="00000000" top="00100000" risk=`operator_gap_oracle_miss`
- `75cd12f1` target="00011110" top="10001110" risk=`operator_gap_oracle_miss`

### Operator Gap Clusters

- `top_operator_family`: `{"affine_gf2": 33, "boolean_template": 124, "nibble_byte_transform": 1, "rotation": 3}`
- `oracle_operator_family`: `{"unknown": 161}`
- `oracle_rank_bucket`: `{"miss": 161}`
- `support_stability`: `{"ambiguous_support_fit": 113, "stable_low_complexity": 48}`
- `top_hamming_to_target`: `{"1": 76, "2-3": 80, "4+": 5}`
- `top_oracle_hamming`: `{"unknown": 161}`
- `top_complexity_penalty`: `{"0-0.10": 161}`
- `top_expression_complexity`: `{"0": 37, "1-4": 5, "5-8": 6, "9+": 113}`
