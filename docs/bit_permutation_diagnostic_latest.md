# Bit Permutation Diagnostic

| manifest | risk_class | n | top1_acc | oracle_at_k | support_full_top1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | low_risk_top1 | 15 | 1.0000 | 1.0000 | 1.0000 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | operator_gap_oracle_miss | 26 | 0.0000 | 0.0000 | 1.0000 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | ranker_miss_oracle_hit | 7 | 0.0000 | 1.0000 | 1.0000 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | low_risk_top1 | 109 | 1.0000 | 1.0000 | 1.0000 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | operator_gap_oracle_miss | 99 | 0.0000 | 0.0000 | 1.0000 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | ranker_miss_oracle_hit | 30 | 0.0000 | 1.0000 | 1.0000 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | support_incomplete | 1 | 0.0000 | 0.0000 | 0.0000 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | low_risk_top1 | 28 | 1.0000 | 1.0000 | 1.0000 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | operator_gap_oracle_miss | 26 | 0.0000 | 0.0000 | 1.0000 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | ranker_miss_oracle_hit | 3 | 0.0000 | 1.0000 | 1.0000 |

## Top Step Buckets

- `binary_boolean_expr`: 320
- `binary_affine_transform`: 16
- `binary_invert>binary_rotate_left>binary_nibble_map`: 2
- `binary_nibble_map`: 2
- `binary_invert>binary_boolean_expr`: 2
- `binary_rotate_left>binary_nibble_map`: 1
- `binary_invert>binary_rotate_left>bitwise_or_constant`: 1

## Ranker Misses

- `1fe9b923` oracle_rank=`5` target=`00000001` top=`10000001` oracle=`00000001` top_oracle_hamming=`1` top_ops=`and,and,and,const,const,const,and,and` oracle_ops=`and,and,and,const,const,const,and,and` steps=`binary_boolean_expr`
- `4e5df314` oracle_rank=`5` target=`00100010` top=`10100010` oracle=`00100010` top_oracle_hamming=`1` top_ops=`and,copy,copy,copy,copy,copy,copy,copy` oracle_ops=`and,copy,copy,copy,copy,copy,copy,copy` steps=`binary_boolean_expr`
- `52d72862` oracle_rank=`2` target=`01001000` top=`01000000` oracle=`01001000` top_oracle_hamming=`1` top_ops=`copy,copy,copy,xor,and,xor,copy,copy` oracle_ops=`` steps=`binary_boolean_expr`
- `68614b78` oracle_rank=`2` target=`00001110` top=`01001110` oracle=`00001110` top_oracle_hamming=`1` top_ops=`or,nor,xnor,or,xnor,copy,xor,or` oracle_ops=`` steps=`binary_boolean_expr`
- `7f73016f` oracle_rank=`2` target=`01010110` top=`01010010` oracle=`01010110` top_oracle_hamming=`1` top_ops=`` oracle_ops=`xnor,xnor,copy,copy,copy,choice,const,copy` steps=`binary_affine_transform`
- `a59e2ff7` oracle_rank=`5` target=`01011111` top=`01011110` oracle=`01011111` top_oracle_hamming=`1` top_ops=`copy,copy,copy,or,or,or,or,or` oracle_ops=`copy,copy,copy,or,or,or,or,or` steps=`binary_boolean_expr`
- `a897b8bc` oracle_rank=`4` target=`01111101` top=`01111111` oracle=`01111101` top_oracle_hamming=`1` top_ops=`copy,or,or,or,or,or,or,or` oracle_ops=`copy,or,or,or,or,or,or,or` steps=`binary_boolean_expr`
- `3e847951` oracle_rank=`2` target=`10110010` top=`10110011` oracle=`10110010` top_oracle_hamming=`1` top_ops=`copy,copy,copy,xor,xor,xor,xor,or` oracle_ops=`` steps=`binary_boolean_expr`
- `a897b8bc` oracle_rank=`4` target=`01111101` top=`01111111` oracle=`01111101` top_oracle_hamming=`1` top_ops=`copy,or,or,or,or,or,or,or` oracle_ops=`copy,or,or,or,or,or,or,or` steps=`binary_boolean_expr`
- `cf9bebe9` oracle_rank=`2` target=`10111001` top=`10011001` oracle=`10111001` top_oracle_hamming=`1` top_ops=`xnor,xnor,and,xnor,xnor,and,not,const` oracle_ops=`` steps=`binary_boolean_expr`
- `08df5363` oracle_rank=`2` target=`01011100` top=`11011100` oracle=`01011100` top_oracle_hamming=`1` top_ops=`nor,copy,copy,copy,copy,copy,copy,copy` oracle_ops=`` steps=`binary_boolean_expr`
- `0f0e199c` oracle_rank=`2` target=`00001111` top=`11001111` oracle=`00001111` top_oracle_hamming=`2` top_ops=`or,or,xor,xor,xor,or,copy,copy` oracle_ops=`` steps=`binary_boolean_expr`
- `16122fba` oracle_rank=`2` target=`11000111` top=`10000111` oracle=`11000111` top_oracle_hamming=`1` top_ops=`xnor,nand,xnor,xnor,xnor,const,const,const` oracle_ops=`` steps=`binary_boolean_expr`
- `1fe9b923` oracle_rank=`5` target=`00000001` top=`10000001` oracle=`00000001` top_oracle_hamming=`1` top_ops=`and,and,and,const,const,const,and,and` oracle_ops=`and,and,and,const,const,const,and,and` steps=`binary_boolean_expr`
- `2395d6df` oracle_rank=`2` target=`00011010` top=`10011010` oracle=`00011010` top_oracle_hamming=`1` top_ops=`choice,copy,copy,copy,copy,copy,copy,copy` oracle_ops=`` steps=`binary_boolean_expr`
- `3595c82c` oracle_rank=`5` target=`10011011` top=`10010111` oracle=`10011011` top_oracle_hamming=`2` top_ops=`xnor,and,xor,xnor,and,xor,xnor,copy` oracle_ops=`xnor,and,xor,xnor,and,xor,xnor,copy` steps=`binary_boolean_expr`
- `3e847951` oracle_rank=`2` target=`10110010` top=`10110011` oracle=`10110010` top_oracle_hamming=`1` top_ops=`copy,copy,copy,xor,xor,xor,xor,or` oracle_ops=`` steps=`binary_boolean_expr`
- `46ae00b4` oracle_rank=`5` target=`11100010` top=`11100011` oracle=`11100010` top_oracle_hamming=`1` top_ops=`copy,const,copy,const,const,and,and,and` oracle_ops=`copy,const,copy,const,const,and,and,and` steps=`binary_boolean_expr`
- `4e5df314` oracle_rank=`5` target=`00100010` top=`10100010` oracle=`00100010` top_oracle_hamming=`1` top_ops=`and,copy,copy,copy,copy,copy,copy,copy` oracle_ops=`and,copy,copy,copy,copy,copy,copy,copy` steps=`binary_boolean_expr`
- `52d72862` oracle_rank=`2` target=`01001000` top=`01000000` oracle=`01001000` top_oracle_hamming=`1` top_ops=`copy,copy,copy,xor,and,xor,copy,copy` oracle_ops=`` steps=`binary_boolean_expr`