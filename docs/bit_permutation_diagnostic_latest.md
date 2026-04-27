# Bit Permutation Diagnostic

| manifest | risk_class | n | top1_acc | oracle_at_k | support_full_top1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | low_risk_top1 | 21 | 1.0000 | 1.0000 | 1.0000 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | operator_gap_oracle_miss | 23 | 0.0000 | 0.0000 | 1.0000 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | ranker_miss_oracle_hit | 3 | 0.0000 | 1.0000 | 1.0000 |
| `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl` | support_incomplete | 1 | 0.0000 | 0.0000 | 0.0000 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | low_risk_top1 | 95 | 1.0000 | 1.0000 | 1.0000 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | operator_gap_oracle_miss | 112 | 0.0000 | 0.0000 | 1.0000 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | ranker_miss_oracle_hit | 28 | 0.0000 | 1.0000 | 1.0000 |
| `data\processed\local_eval_manifests\hard_triad_full.jsonl` | support_incomplete | 5 | 0.0000 | 0.0000 | 0.0000 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | low_risk_top1 | 29 | 1.0000 | 1.0000 | 1.0000 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | operator_gap_oracle_miss | 26 | 0.0000 | 0.0000 | 1.0000 |
| `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl` | ranker_miss_oracle_hit | 2 | 0.0000 | 1.0000 | 1.0000 |

## Top Step Buckets

- `binary_boolean_expr`: 279
- `binary_affine_transform`: 48
- `binary_invert>binary_boolean_expr`: 6
- `binary_rotate_left>binary_nibble_map`: 2
- `binary_rotate_left>bitwise_or_constant`: 2
- `swap_nibbles>binary_nibble_map`: 2
- `bitwise_or_constant>binary_rotate_left`: 2
- `binary_invert>binary_rotate_left>binary_nibble_map`: 1
- `binary_nibble_map`: 1
- `reverse_bits>binary_rotate_left>bitwise_or_constant`: 1
- `binary_rotate_left>binary_and_mask`: 1

## Ranker Misses

- `455b6b61` oracle_rank=`2` target=`00111011` top=`00101011` oracle=`00111011` top_oracle_hamming=`1` top_ops=`copy,copy,xor,and,xor,and,xor,xor` oracle_ops=`` steps=`binary_boolean_expr`
- `54ca9d57` oracle_rank=`2` target=`01011111` top=`11011111` oracle=`01011111` top_oracle_hamming=`1` top_ops=`` oracle_ops=`choice,xor,const,copy,copy,copy,copy,copy` steps=`binary_affine_transform`
- `81323d52` oracle_rank=`2` target=`10110001` top=`10111001` oracle=`10110001` top_oracle_hamming=`1` top_ops=`copy,copy,copy,xor,or,xor,xor,copy` oracle_ops=`` steps=`binary_boolean_expr`
- `3e847951` oracle_rank=`2` target=`10110010` top=`10110011` oracle=`10110010` top_oracle_hamming=`1` top_ops=`copy,copy,copy,xor,xor,xor,xor,or` oracle_ops=`` steps=`binary_boolean_expr`
- `a897b8bc` oracle_rank=`4` target=`01111101` top=`01111111` oracle=`01111101` top_oracle_hamming=`1` top_ops=`copy,or,or,or,or,or,or,or` oracle_ops=`copy,or,or,or,or,or,or,or` steps=`binary_boolean_expr`
- `0e7f299d` oracle_rank=`4` target=`01010010` top=`01010011` oracle=`01010010` top_oracle_hamming=`1` top_ops=`copy,copy,copy,copy,copy,copy,or,or` oracle_ops=`copy,copy,copy,copy,copy,copy,or,or` steps=`binary_boolean_expr`
- `1d930d32` oracle_rank=`2` target=`00000100` top=`00000000` oracle=`00000100` top_oracle_hamming=`1` top_ops=`copy,copy,copy,copy,xor,and,xor,copy` oracle_ops=`` steps=`binary_boolean_expr`
- `1fe9b923` oracle_rank=`5` target=`00000001` top=`10000001` oracle=`00000001` top_oracle_hamming=`1` top_ops=`and,and,and,const,const,const,and,and` oracle_ops=`and,and,and,const,const,const,and,and` steps=`binary_boolean_expr`
- `3131bfb3` oracle_rank=`4` target=`11111101` top=`11111100` oracle=`11111101` top_oracle_hamming=`1` top_ops=`copy,copy,copy,copy,copy,copy,or,or` oracle_ops=`copy,copy,copy,copy,copy,copy,or,or` steps=`binary_boolean_expr`
- `37c94738` oracle_rank=`4` target=`00000000` top=`00001000` oracle=`00000000` top_oracle_hamming=`1` top_ops=`const,const,and,and,and,and,const,and` oracle_ops=`const,const,and,and,and,and,const,and` steps=`binary_boolean_expr`
- `3e847951` oracle_rank=`2` target=`10110010` top=`10110011` oracle=`10110010` top_oracle_hamming=`1` top_ops=`copy,copy,copy,xor,xor,xor,xor,or` oracle_ops=`` steps=`binary_boolean_expr`
- `4195699e` oracle_rank=`4` target=`01100000` top=`00000000` oracle=`01100000` top_oracle_hamming=`2` top_ops=`and,and,xor,const,const,const,const,const` oracle_ops=`and,and,and,const,const,const,const,const` steps=`binary_boolean_expr`
- `455b6b61` oracle_rank=`2` target=`00111011` top=`00101011` oracle=`00111011` top_oracle_hamming=`1` top_ops=`copy,copy,xor,and,xor,and,xor,xor` oracle_ops=`` steps=`binary_boolean_expr`
- `49b5ead6` oracle_rank=`4` target=`10111110` top=`10101110` oracle=`10111110` top_oracle_hamming=`1` top_ops=`copy,copy,copy,or,or,or,or,or` oracle_ops=`copy,copy,copy,or,or,or,or,or` steps=`binary_boolean_expr`
- `4b52b575` oracle_rank=`2` target=`10000100` top=`10000000` oracle=`10000100` top_oracle_hamming=`1` top_ops=`copy,const,const,const,const,and,and,const` oracle_ops=`` steps=`binary_boolean_expr`
- `54ca9d57` oracle_rank=`2` target=`01011111` top=`11011111` oracle=`01011111` top_oracle_hamming=`1` top_ops=`` oracle_ops=`choice,xor,const,copy,copy,copy,copy,copy` steps=`binary_affine_transform`
- `5bd26372` oracle_rank=`2` target=`00101000` top=`00001000` oracle=`00101000` top_oracle_hamming=`1` top_ops=`copy,copy,and,xor,xor,copy,copy,copy` oracle_ops=`` steps=`binary_boolean_expr`
- `63d4557e` oracle_rank=`2` target=`10111100` top=`10111101` oracle=`10111100` top_oracle_hamming=`1` top_ops=`xor,xnor,or,not,choice,copy,not,or` oracle_ops=`or,xnor,xor,copy,choice,not,copy,or` steps=`binary_boolean_expr`
- `733a819b` oracle_rank=`2` target=`00111011` top=`10111111` oracle=`00111011` top_oracle_hamming=`2` top_ops=`xor,xor,or,xor,or,or,copy,copy` oracle_ops=`` steps=`binary_boolean_expr`
- `77804b32` oracle_rank=`4` target=`00010110` top=`00011110` oracle=`00010110` top_oracle_hamming=`1` top_ops=`copy,copy,copy,copy,or,or,or,or` oracle_ops=`copy,copy,copy,copy,or,or,or,or` steps=`binary_boolean_expr`