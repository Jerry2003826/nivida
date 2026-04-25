# Bit Permutation Diagnostic

| manifest | risk_class | n | top1_acc | oracle_at_k | support_full_top1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | low_risk_top1 | 17 | 1.0000 | 1.0000 | 1.0000 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | operator_gap_oracle_miss | 25 | 0.0000 | 0.0000 | 1.0000 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | ranker_miss_oracle_hit | 4 | 0.0000 | 1.0000 | 1.0000 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | support_incomplete | 2 | 0.0000 | 0.0000 | 0.0000 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | low_risk_top1 | 96 | 1.0000 | 1.0000 | 1.0000 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | operator_gap_oracle_miss | 117 | 0.0000 | 0.0000 | 1.0000 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | ranker_miss_oracle_hit | 23 | 0.0000 | 1.0000 | 1.0000 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | support_incomplete | 3 | 0.0000 | 0.0000 | 0.0000 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | low_risk_top1 | 28 | 1.0000 | 1.0000 | 1.0000 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | operator_gap_oracle_miss | 26 | 0.0000 | 0.0000 | 1.0000 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | ranker_miss_oracle_hit | 3 | 0.0000 | 1.0000 | 1.0000 |

## Top Step Buckets

- `binary_boolean_expr`: 321
- `binary_affine_transform`: 13
- `bitwise_or_constant>binary_rotate_left`: 2
- `binary_invert>binary_rotate_left>bitwise_or_constant`: 2
- `binary_invert>binary_boolean_expr`: 2
- `binary_invert>binary_rotate_left>binary_nibble_map`: 1
- `binary_nibble_map`: 1
- `binary_rotate_left>binary_nibble_map`: 1
- `reverse_bits>binary_rotate_left>bitwise_or_constant`: 1

## Ranker Misses

- `551e93e7` oracle_rank=`4` target=`00000000` top=`00000100` oracle=`00000000` top_oracle_hamming=`1` top_ops=`const,const,const,const,and,xor,and,and` oracle_ops=`const,const,const,const,and,and,and,and` steps=`binary_boolean_expr`
- `a365e304` oracle_rank=`5` target=`11011011` top=`11001111` oracle=`11011011` top_oracle_hamming=`2` top_ops=`copy,copy,copy,copy,copy,xor,xor,xor` oracle_ops=`copy,copy,copy,copy,copy,xor,xor,xor` steps=`binary_boolean_expr`
- `bf002000` oracle_rank=`5` target=`11001000` top=`11001001` oracle=`11001000` top_oracle_hamming=`1` top_ops=`copy,copy,copy,copy,copy,copy,copy,xor` oracle_ops=`copy,copy,copy,copy,copy,copy,copy,xor` steps=`binary_boolean_expr`
- `ea5def07` oracle_rank=`3` target=`11111111` top=`11111100` oracle=`11111111` top_oracle_hamming=`2` top_ops=`xnor,nand,xnor,xnor,not,const,choice,xnor` oracle_ops=`xnor,or,xnor,xnor,copy,const,choice,or` steps=`binary_boolean_expr`
- `3e847951` oracle_rank=`2` target=`10110010` top=`10110011` oracle=`10110010` top_oracle_hamming=`1` top_ops=`copy,copy,copy,xor,xor,xor,xor,or` oracle_ops=`` steps=`binary_boolean_expr`
- `a897b8bc` oracle_rank=`4` target=`01111101` top=`01111111` oracle=`01111101` top_oracle_hamming=`1` top_ops=`copy,or,or,or,or,or,or,or` oracle_ops=`copy,or,or,or,or,or,or,or` steps=`binary_boolean_expr`
- `cf9bebe9` oracle_rank=`2` target=`10111001` top=`10011001` oracle=`10111001` top_oracle_hamming=`1` top_ops=`xnor,xnor,and,xnor,xnor,and,not,const` oracle_ops=`` steps=`binary_boolean_expr`
- `0e7f299d` oracle_rank=`4` target=`01010010` top=`01010011` oracle=`01010010` top_oracle_hamming=`1` top_ops=`copy,copy,copy,copy,copy,copy,or,or` oracle_ops=`copy,copy,copy,copy,copy,copy,or,or` steps=`binary_boolean_expr`
- `16122fba` oracle_rank=`2` target=`11000111` top=`10000111` oracle=`11000111` top_oracle_hamming=`1` top_ops=`xnor,nand,xnor,xnor,xnor,const,const,const` oracle_ops=`` steps=`binary_boolean_expr`
- `1d930d32` oracle_rank=`2` target=`00000100` top=`00000000` oracle=`00000100` top_oracle_hamming=`1` top_ops=`copy,copy,copy,copy,xor,and,xor,copy` oracle_ops=`` steps=`binary_boolean_expr`
- `1fe9b923` oracle_rank=`5` target=`00000001` top=`10000001` oracle=`00000001` top_oracle_hamming=`1` top_ops=`and,and,and,const,const,const,and,and` oracle_ops=`and,and,and,const,const,const,and,and` steps=`binary_boolean_expr`
- `2facfaa4` oracle_rank=`5` target=`00111110` top=`00111100` oracle=`00111110` top_oracle_hamming=`1` top_ops=`copy,copy,copy,copy,copy,copy,xor,copy` oracle_ops=`copy,copy,copy,copy,copy,copy,xor,copy` steps=`binary_boolean_expr`
- `31df43f1` oracle_rank=`4` target=`11011100` top=`11111100` oracle=`11011100` top_oracle_hamming=`1` top_ops=`or,or,or,or,or,copy,copy,copy` oracle_ops=`or,or,or,or,or,copy,copy,copy` steps=`binary_boolean_expr`
- `3a61d8ad` oracle_rank=`2` target=`11111111` top=`11111110` oracle=`11111111` top_oracle_hamming=`1` top_ops=`const,const,const,const,const,const,const,nand` oracle_ops=`` steps=`binary_boolean_expr`
- `53b84650` oracle_rank=`2` target=`11111101` top=`11111111` oracle=`11111101` top_oracle_hamming=`1` top_ops=`copy,xor,xor,xor,xor,xor,or,xor` oracle_ops=`` steps=`binary_boolean_expr`
- `551e93e7` oracle_rank=`4` target=`00000000` top=`00000100` oracle=`00000000` top_oracle_hamming=`1` top_ops=`const,const,const,const,and,xor,and,and` oracle_ops=`const,const,const,const,and,and,and,and` steps=`binary_boolean_expr`
- `71b70d29` oracle_rank=`2` target=`00101001` top=`00111001` oracle=`00101001` top_oracle_hamming=`1` top_ops=`copy,xor,xor,or,xor,copy,copy,copy` oracle_ops=`` steps=`binary_boolean_expr`
- `783a1317` oracle_rank=`2` target=`00111110` top=`10111110` oracle=`00111110` top_oracle_hamming=`1` top_ops=`and,and,copy,copy,copy,copy,copy,copy` oracle_ops=`` steps=`binary_boolean_expr`
- `78d02fc5` oracle_rank=`2` target=`11001001` top=`01001000` oracle=`11001001` top_oracle_hamming=`2` top_ops=`and,xnor,not,not,not,not,not,nand` oracle_ops=`` steps=`binary_boolean_expr`
- `81323d52` oracle_rank=`2` target=`10110001` top=`10111001` oracle=`10110001` top_oracle_hamming=`1` top_ops=`copy,copy,copy,xor,or,xor,xor,copy` oracle_ops=`` steps=`binary_boolean_expr`