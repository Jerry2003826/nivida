# Solver Coverage Audit

This measures how much of the labeled local eval set is already explainable by the rule/search system.

## `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl`

- rows: `288`
- query accuracy: `0.6910`
- support-full rate: `0.9722`
- avg support accuracy: `0.9808`
- failure classes: `{"query_correct": 198, "query_wrong_after_support_fit": 82, "support_incomplete": 8}`

| family | n | query_acc | support_full | avg_support | failures | top signature buckets |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| bit | 48 | 0.1250 | 0.8750 | 0.9113 | `{"query_wrong_after_support_fit": 37, "support_incomplete": 6, "query_correct": 5}` | `{"affine": 38, "rotl>nibble": 3, "rotl>binary_and_mask": 2, "reverse_bits>binary_and_mask": 1, "rotl>bitwise_or_constant": 1, "rotl>bitwise_xor_constant>rotl": 1, "nibble": 1, "bitwise_or_constant>reverse_bits": 1}` |
| cipher | 48 | 0.8750 | 1.0000 | 1.0000 | `{"query_correct": 42, "query_wrong_after_support_fit": 6}` | `{"vocab_sub": 48}` |
| equation | 48 | 0.1458 | 0.9583 | 0.9736 | `{"query_wrong_after_support_fit": 39, "query_correct": 7, "support_incomplete": 2}` | `{"template": 25, "eq_rule": 7, "position": 3, "evaluate_expression>mul": 1, "template>]],[=[[pos,3],[lit,>],[lit,]]],`=[[pos,0],[lit,?],[lit,?]]": 1, "position>=,{=,%=,@=>,[=,\\=+,(={,+=,?=\\,%=>=/,[=/,": 1, "template>=[[pos,2],[lit,^],[pos,4]],\\=[[pos,0],[pos,1],[pos,3],[pos,4]],}=[[lit,@],[lit,": 1, "evaluate_expression>affine": 1}` |
| gravity | 48 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` | `{"inverse_square": 48}` |
| numeral | 48 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` | `{"dec2roman": 48}` |
| unit | 48 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` | `{"scale": 44, "scale>scale": 4}` |

### Weakest Subtypes

| subtype | n | query_acc | support_full | failures |
| --- | ---: | ---: | ---: | --- |
| equation_position | 39 | 0.0256 | 1.0000 | `{"query_wrong_after_support_fit": 38, "query_correct": 1}` |
| bit_permutation | 48 | 0.1250 | 0.8750 | `{"query_wrong_after_support_fit": 37, "support_incomplete": 6, "query_correct": 5}` |
| equation_numeric | 9 | 0.6667 | 0.7778 | `{"query_correct": 6, "support_incomplete": 2, "query_wrong_after_support_fit": 1}` |
| cipher_char_sub | 48 | 0.8750 | 1.0000 | `{"query_correct": 42, "query_wrong_after_support_fit": 6}` |
| gravity_inverse_square | 48 | 1.0000 | 1.0000 | `{"query_correct": 48}` |
| numeral_roman | 48 | 1.0000 | 1.0000 | `{"query_correct": 48}` |
| unit_scale | 48 | 1.0000 | 1.0000 | `{"query_correct": 48}` |

## `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl`

- rows: `352`
- query accuracy: `0.7045`
- support-full rate: `0.9545`
- avg support accuracy: `0.9681`
- failure classes: `{"query_correct": 247, "query_wrong_after_support_fit": 89, "support_incomplete": 16}`

| family | n | query_acc | support_full | avg_support | failures | top signature buckets |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| bit | 57 | 0.2632 | 0.8947 | 0.9317 | `{"query_wrong_after_support_fit": 37, "query_correct": 14, "support_incomplete": 6}` | `{"affine": 47, "rotl>binary_and_mask": 3, "rotl>nibble": 2, "binary_invert>rotl>nibble": 1, "nibble": 1, "reverse_bits>binary_and_mask": 1, "rotl>bitwise_xor_constant>rotl": 1, "rotl>reverse_bits": 1}` |
| cipher | 39 | 0.8205 | 1.0000 | 1.0000 | `{"query_correct": 32, "query_wrong_after_support_fit": 7}` | `{"vocab_sub": 39}` |
| equation | 64 | 0.1406 | 0.8438 | 0.8854 | `{"query_wrong_after_support_fit": 45, "support_incomplete": 10, "query_correct": 9}` | `{"template": 32, "evaluate_expression>add": 9, "eq_rule": 6, "position": 2, "template>]],]=[[lit,#],[lit,[],[pos,4]]": 1, "template>=[[lit,&],[lit,&],[pos,1]]": 1, "position>=,{=,%=,@=>,[=,\\=+,(={,+=,?=\\,%=>=/,[=/,": 1, "evaluate_expression>affine": 1}` |
| gravity | 64 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 64}` | `{"inverse_square": 64}` |
| numeral | 64 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 64}` | `{"dec2roman": 64}` |
| unit | 64 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 64}` | `{"scale": 60, "scale>scale": 4}` |

### Weakest Subtypes

| subtype | n | query_acc | support_full | failures |
| --- | ---: | ---: | ---: | --- |
| equation_position | 48 | 0.0833 | 1.0000 | `{"query_wrong_after_support_fit": 44, "query_correct": 4}` |
| bit_permutation | 57 | 0.2632 | 0.8947 | `{"query_wrong_after_support_fit": 37, "query_correct": 14, "support_incomplete": 6}` |
| equation_numeric | 16 | 0.3125 | 0.3750 | `{"support_incomplete": 10, "query_correct": 5, "query_wrong_after_support_fit": 1}` |
| cipher_char_sub | 39 | 0.8205 | 1.0000 | `{"query_correct": 32, "query_wrong_after_support_fit": 7}` |
| gravity_inverse_square | 64 | 1.0000 | 1.0000 | `{"query_correct": 64}` |
| numeral_roman | 64 | 1.0000 | 1.0000 | `{"query_correct": 64}` |
| unit_scale | 64 | 1.0000 | 1.0000 | `{"query_correct": 64}` |

## `data\processed\local_eval_manifests\hard_triad_full.jsonl`

- rows: `709`
- query accuracy: `0.4302`
- support-full rate: `0.9394`
- avg support accuracy: `0.9569`
- failure classes: `{"query_wrong_after_support_fit": 364, "query_correct": 302, "support_incomplete": 43}`

| family | n | query_acc | support_full | avg_support | failures | top signature buckets |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| bit | 240 | 0.2750 | 0.9000 | 0.9288 | `{"query_wrong_after_support_fit": 153, "query_correct": 63, "support_incomplete": 24}` | `{"affine": 202, "rotl>binary_and_mask": 13, "rotl>nibble": 6, "nibble": 4, "binary_invert>rotl>nibble": 2, "swap_nibbles>binary_and_mask": 2, "bitwise_or_constant>binary_invert": 1, "swap_nibbles>nibble": 1}` |
| cipher | 236 | 0.8729 | 1.0000 | 1.0000 | `{"query_correct": 206, "query_wrong_after_support_fit": 30}` | `{"vocab_sub": 236}` |
| equation | 233 | 0.1416 | 0.9185 | 0.9421 | `{"query_wrong_after_support_fit": 181, "query_correct": 33, "support_incomplete": 19}` | `{"template": 131, "eq_rule": 20, "evaluate_expression>add": 17, "position": 10, "template>]]": 2, "template>]],]=[[lit,#],[lit,[],[pos,4]]": 1, "template>],[lit,\"]],]=[[lit,'],[pos,0]]": 1, "template>],[pos,3]]": 1}` |

### Weakest Subtypes

| subtype | n | query_acc | support_full | failures |
| --- | ---: | ---: | ---: | --- |
| equation_position | 193 | 0.0777 | 1.0000 | `{"query_wrong_after_support_fit": 178, "query_correct": 15}` |
| bit_permutation | 239 | 0.2720 | 0.8996 | `{"query_wrong_after_support_fit": 153, "query_correct": 62, "support_incomplete": 24}` |
| equation_numeric | 40 | 0.4500 | 0.5250 | `{"support_incomplete": 19, "query_correct": 18, "query_wrong_after_support_fit": 3}` |
| cipher_char_sub | 236 | 0.8729 | 1.0000 | `{"query_correct": 206, "query_wrong_after_support_fit": 30}` |
| bit_rotate | 1 | 1.0000 | 1.0000 | `{"query_correct": 1}` |
