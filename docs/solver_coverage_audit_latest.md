# Solver Coverage Audit

This measures how much of the labeled local eval set is already explainable by the rule/search system.

## `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl`

- rows: `288`
- query accuracy: `0.6528`
- support-full rate: `0.9722`
- avg support accuracy: `0.9834`
- failure classes: `{"query_correct": 186, "query_wrong_after_support_fit": 94, "support_incomplete": 8}`

| family | n | query_acc | support_full | avg_support | failures | top signature buckets |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| bit | 48 | 0.3333 | 0.8750 | 0.9269 | `{"query_wrong_after_support_fit": 28, "query_correct": 14, "support_incomplete": 6}` | `{"affine": 38, "rotl>nibble": 3, "rotl>binary_and_mask": 2, "reverse_bits>binary_and_mask": 1, "rotl>bitwise_or_constant": 1, "rotl>bitwise_xor_constant>rotl": 1, "nibble": 1, "bitwise_or_constant>reverse_bits": 1}` |
| cipher | 48 | 0.4375 | 1.0000 | 1.0000 | `{"query_wrong_after_support_fit": 27, "query_correct": 21}` | `{"char_sub": 48}` |
| equation | 48 | 0.1458 | 0.9583 | 0.9736 | `{"query_wrong_after_support_fit": 39, "query_correct": 7, "support_incomplete": 2}` | `{"template": 25, "eq_rule": 7, "position": 3, "evaluate_expression>mul": 1, "template>]],[=[[pos,3],[lit,>],[lit,]]],`=[[pos,0],[lit,?],[lit,?]]": 1, "position>=,{=,%=,@=>,[=,\\=+,(={,+=,?=\\,%=>=/,[=/,": 1, "template>=[[pos,2],[lit,^],[pos,4]],\\=[[pos,0],[pos,1],[pos,3],[pos,4]],}=[[lit,@],[lit,": 1, "evaluate_expression>affine": 1}` |
| gravity | 48 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` | `{"inverse_square": 48}` |
| numeral | 48 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` | `{"dec2roman": 48}` |
| unit | 48 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` | `{"scale": 44, "scale>scale": 4}` |

### Weakest Subtypes

| subtype | n | query_acc | support_full | failures |
| --- | ---: | ---: | ---: | --- |
| equation_position | 39 | 0.0256 | 1.0000 | `{"query_wrong_after_support_fit": 38, "query_correct": 1}` |
| bit_permutation | 48 | 0.3333 | 0.8750 | `{"query_wrong_after_support_fit": 28, "query_correct": 14, "support_incomplete": 6}` |
| cipher_char_sub | 48 | 0.4375 | 1.0000 | `{"query_wrong_after_support_fit": 27, "query_correct": 21}` |
| equation_numeric | 9 | 0.6667 | 0.7778 | `{"query_correct": 6, "support_incomplete": 2, "query_wrong_after_support_fit": 1}` |
| gravity_inverse_square | 48 | 1.0000 | 1.0000 | `{"query_correct": 48}` |
| numeral_roman | 48 | 1.0000 | 1.0000 | `{"query_correct": 48}` |
| unit_scale | 48 | 1.0000 | 1.0000 | `{"query_correct": 48}` |

## `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl`

- rows: `352`
- query accuracy: `0.6960`
- support-full rate: `0.9545`
- avg support accuracy: `0.9696`
- failure classes: `{"query_correct": 244, "query_wrong_after_support_fit": 92, "support_incomplete": 16}`

| family | n | query_acc | support_full | avg_support | failures | top signature buckets |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| bit | 57 | 0.5965 | 0.8947 | 0.9407 | `{"query_correct": 33, "query_wrong_after_support_fit": 18, "support_incomplete": 6}` | `{"affine": 47, "rotl>binary_and_mask": 3, "rotl>nibble": 2, "binary_invert>rotl>nibble": 1, "nibble": 1, "reverse_bits>binary_and_mask": 1, "rotl>bitwise_xor_constant>rotl": 1, "rotl>reverse_bits": 1}` |
| cipher | 39 | 0.2564 | 1.0000 | 1.0000 | `{"query_wrong_after_support_fit": 29, "query_correct": 10}` | `{"char_sub": 39}` |
| equation | 64 | 0.1406 | 0.8438 | 0.8854 | `{"query_wrong_after_support_fit": 45, "support_incomplete": 10, "query_correct": 9}` | `{"template": 32, "evaluate_expression>add": 9, "eq_rule": 6, "position": 2, "template>]],]=[[lit,#],[lit,[],[pos,4]]": 1, "template>=[[lit,&],[lit,&],[pos,1]]": 1, "position>=,{=,%=,@=>,[=,\\=+,(={,+=,?=\\,%=>=/,[=/,": 1, "evaluate_expression>affine": 1}` |
| gravity | 64 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 64}` | `{"inverse_square": 64}` |
| numeral | 64 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 64}` | `{"dec2roman": 64}` |
| unit | 64 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 64}` | `{"scale": 60, "scale>scale": 4}` |

### Weakest Subtypes

| subtype | n | query_acc | support_full | failures |
| --- | ---: | ---: | ---: | --- |
| equation_position | 48 | 0.0833 | 1.0000 | `{"query_wrong_after_support_fit": 44, "query_correct": 4}` |
| cipher_char_sub | 39 | 0.2564 | 1.0000 | `{"query_wrong_after_support_fit": 29, "query_correct": 10}` |
| equation_numeric | 16 | 0.3125 | 0.3750 | `{"support_incomplete": 10, "query_correct": 5, "query_wrong_after_support_fit": 1}` |
| bit_permutation | 57 | 0.5965 | 0.8947 | `{"query_correct": 33, "query_wrong_after_support_fit": 18, "support_incomplete": 6}` |
| gravity_inverse_square | 64 | 1.0000 | 1.0000 | `{"query_correct": 64}` |
| numeral_roman | 64 | 1.0000 | 1.0000 | `{"query_correct": 64}` |
| unit_scale | 64 | 1.0000 | 1.0000 | `{"query_correct": 64}` |

## `data\processed\local_eval_manifests\hard_triad_full.jsonl`

- rows: `709`
- query accuracy: `0.3399`
- support-full rate: `0.9408`
- avg support accuracy: `0.9600`
- failure classes: `{"query_wrong_after_support_fit": 432, "query_correct": 235, "support_incomplete": 42}`

| family | n | query_acc | support_full | avg_support | failures | top signature buckets |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| bit | 240 | 0.5167 | 0.9042 | 0.9381 | `{"query_correct": 118, "query_wrong_after_support_fit": 99, "support_incomplete": 23}` | `{"affine": 202, "rotl>binary_and_mask": 13, "rotl>nibble": 7, "nibble": 4, "binary_invert>rotl>bitwise_or_constant": 2, "swap_nibbles>binary_and_mask": 2, "bitwise_or_constant>binary_invert": 1, "binary_invert>rotl>nibble": 1}` |
| cipher | 236 | 0.3517 | 1.0000 | 1.0000 | `{"query_wrong_after_support_fit": 153, "query_correct": 83}` | `{"char_sub": 236}` |
| equation | 233 | 0.1459 | 0.9185 | 0.9421 | `{"query_wrong_after_support_fit": 180, "query_correct": 34, "support_incomplete": 19}` | `{"template": 131, "eq_rule": 20, "evaluate_expression>add": 17, "position": 10, "template>]]": 2, "template>]],]=[[lit,#],[lit,[],[pos,4]]": 1, "template>],[lit,\"]],]=[[lit,'],[pos,0]]": 1, "template>],[pos,3]]": 1}` |

### Weakest Subtypes

| subtype | n | query_acc | support_full | failures |
| --- | ---: | ---: | ---: | --- |
| equation_position | 193 | 0.0777 | 1.0000 | `{"query_wrong_after_support_fit": 178, "query_correct": 15}` |
| cipher_char_sub | 236 | 0.3517 | 1.0000 | `{"query_wrong_after_support_fit": 153, "query_correct": 83}` |
| equation_numeric | 40 | 0.4750 | 0.5250 | `{"query_correct": 19, "support_incomplete": 19, "query_wrong_after_support_fit": 2}` |
| bit_permutation | 239 | 0.5146 | 0.9038 | `{"query_correct": 117, "query_wrong_after_support_fit": 99, "support_incomplete": 23}` |
| bit_rotate | 1 | 1.0000 | 1.0000 | `{"query_correct": 1}` |
