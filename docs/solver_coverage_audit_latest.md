# Solver Coverage Audit

This measures how much of the labeled local eval set is already explainable by the rule/search system.

## `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl`

- rows: `288`
- query accuracy: `0.7118`
- support-full rate: `0.9757`
- avg support accuracy: `0.9831`
- failure classes: `{"query_correct": 204, "query_wrong_after_support_fit": 77, "support_incomplete": 7}`

| family | n | query_acc | support_full | avg_support | failures | top signature buckets |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| bit | 48 | 0.2292 | 0.8750 | 0.9113 | `{"query_wrong_after_support_fit": 32, "query_correct": 10, "support_incomplete": 6}` | `{"affine": 38, "rotl>nibble": 3, "rotl>binary_and_mask": 2, "reverse_bits>binary_and_mask": 1, "rotl>bitwise_or_constant": 1, "rotl>bitwise_xor_constant>rotl": 1, "nibble": 1, "bitwise_or_constant>reverse_bits": 1}` |
| cipher | 48 | 0.8750 | 1.0000 | 1.0000 | `{"query_correct": 42, "query_wrong_after_support_fit": 6}` | `{"vocab_sub": 48}` |
| equation | 48 | 0.1667 | 0.9792 | 0.9875 | `{"query_wrong_after_support_fit": 39, "query_correct": 8, "support_incomplete": 1}` | `{"template": 24, "eq_rule": 8, "position": 4, "template>]],[=[[pos,3],[lit,>],[lit,]]],`=[[pos,0],[lit,?],[lit,?]]": 1, "position>=,{=,%=,@=>,[=,\\=+,(={,+=,?=\\,%=>=/,[=/,": 1, "template>=[[pos,2],[lit,^],[pos,4]],\\=[[pos,0],[pos,1],[pos,3],[pos,4]],}=[[lit,@],[lit,": 1, "evaluate_expression>affine": 1, "template>=[[lit,|],[lit,]]],{=[[lit,!],[lit,|],[lit,|],[pos,1]]": 1}` |
| gravity | 48 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` | `{"inverse_square": 48}` |
| numeral | 48 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` | `{"dec2roman": 48}` |
| unit | 48 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` | `{"scale": 44, "scale>scale": 4}` |

### Weakest Subtypes

| subtype | n | query_acc | support_full | failures |
| --- | ---: | ---: | ---: | --- |
| equation_template | 39 | 0.0256 | 1.0000 | `{"query_wrong_after_support_fit": 38, "query_correct": 1}` |
| bit_permutation | 48 | 0.2292 | 0.8750 | `{"query_wrong_after_support_fit": 32, "query_correct": 10, "support_incomplete": 6}` |
| equation_numeric | 9 | 0.7778 | 0.8889 | `{"query_correct": 7, "query_wrong_after_support_fit": 1, "support_incomplete": 1}` |
| cipher_char_sub | 48 | 0.8750 | 1.0000 | `{"query_correct": 42, "query_wrong_after_support_fit": 6}` |
| gravity_inverse_square | 48 | 1.0000 | 1.0000 | `{"query_correct": 48}` |
| numeral_roman | 48 | 1.0000 | 1.0000 | `{"query_correct": 48}` |
| unit_scale | 48 | 1.0000 | 1.0000 | `{"query_correct": 48}` |

## `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl`

- rows: `352`
- query accuracy: `0.7358`
- support-full rate: `0.9574`
- avg support accuracy: `0.9702`
- failure classes: `{"query_correct": 258, "query_wrong_after_support_fit": 79, "support_incomplete": 15}`

| family | n | query_acc | support_full | avg_support | failures | top signature buckets |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| bit | 57 | 0.4211 | 0.8947 | 0.9317 | `{"query_wrong_after_support_fit": 28, "query_correct": 23, "support_incomplete": 6}` | `{"affine": 47, "rotl>binary_and_mask": 3, "rotl>nibble": 2, "binary_invert>rotl>nibble": 1, "nibble": 1, "reverse_bits>binary_and_mask": 1, "rotl>bitwise_xor_constant>rotl": 1, "rotl>reverse_bits": 1}` |
| cipher | 39 | 0.8205 | 1.0000 | 1.0000 | `{"query_correct": 32, "query_wrong_after_support_fit": 7}` | `{"vocab_sub": 39}` |
| equation | 64 | 0.1719 | 0.8594 | 0.8971 | `{"query_wrong_after_support_fit": 44, "query_correct": 11, "support_incomplete": 9}` | `{"template": 31, "evaluate_expression>add": 8, "eq_rule": 7, "position": 2, "template>]],]=[[lit,#],[lit,[],[pos,4]]": 1, "template>=[[lit,&],[lit,&],[pos,1]]": 1, "position>=,{=,%=,@=>,[=,\\=+,(={,+=,?=\\,%=>=/,[=/,": 1, "template>=[[pos,3],[lit,\"],[lit,|],[lit,": 1}` |
| gravity | 64 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 64}` | `{"inverse_square": 64}` |
| numeral | 64 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 64}` | `{"dec2roman": 64}` |
| unit | 64 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 64}` | `{"scale": 60, "scale>scale": 4}` |

### Weakest Subtypes

| subtype | n | query_acc | support_full | failures |
| --- | ---: | ---: | ---: | --- |
| equation_template | 46 | 0.0870 | 1.0000 | `{"query_wrong_after_support_fit": 42, "query_correct": 4}` |
| equation_numeric | 16 | 0.3750 | 0.4375 | `{"support_incomplete": 9, "query_correct": 6, "query_wrong_after_support_fit": 1}` |
| bit_permutation | 57 | 0.4211 | 0.8947 | `{"query_wrong_after_support_fit": 28, "query_correct": 23, "support_incomplete": 6}` |
| equation_position | 2 | 0.5000 | 1.0000 | `{"query_correct": 1, "query_wrong_after_support_fit": 1}` |
| cipher_char_sub | 39 | 0.8205 | 1.0000 | `{"query_correct": 32, "query_wrong_after_support_fit": 7}` |
| gravity_inverse_square | 64 | 1.0000 | 1.0000 | `{"query_correct": 64}` |
| numeral_roman | 64 | 1.0000 | 1.0000 | `{"query_correct": 64}` |
| unit_scale | 64 | 1.0000 | 1.0000 | `{"query_correct": 64}` |

## `data\processed\local_eval_manifests\hard_triad_full.jsonl`

- rows: `709`
- query accuracy: `0.4824`
- support-full rate: `0.9478`
- avg support accuracy: `0.9630`
- failure classes: `{"query_correct": 339, "query_wrong_after_support_fit": 333, "support_incomplete": 37}`

| family | n | query_acc | support_full | avg_support | failures | top signature buckets |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| bit | 240 | 0.4000 | 0.9000 | 0.9288 | `{"query_wrong_after_support_fit": 123, "query_correct": 93, "support_incomplete": 24}` | `{"affine": 202, "rotl>binary_and_mask": 13, "rotl>nibble": 6, "nibble": 4, "binary_invert>rotl>nibble": 2, "swap_nibbles>binary_and_mask": 2, "bitwise_or_constant>binary_invert": 1, "swap_nibbles>nibble": 1}` |
| cipher | 236 | 0.8729 | 1.0000 | 1.0000 | `{"query_correct": 206, "query_wrong_after_support_fit": 30}` | `{"vocab_sub": 236}` |
| equation | 233 | 0.1717 | 0.9442 | 0.9608 | `{"query_wrong_after_support_fit": 180, "query_correct": 40, "support_incomplete": 13}` | `{"template": 129, "eq_rule": 26, "evaluate_expression>add": 12, "position": 11, "template>]]": 2, "template>]],]=[[lit,#],[lit,[],[pos,4]]": 1, "template>],[lit,\"]],]=[[lit,'],[pos,0]]": 1, "template>],[pos,3]]": 1}` |

### Weakest Subtypes

| subtype | n | query_acc | support_full | failures |
| --- | ---: | ---: | ---: | --- |
| equation_template | 190 | 0.0842 | 1.0000 | `{"query_wrong_after_support_fit": 174, "query_correct": 16}` |
| equation_position | 3 | 0.3333 | 1.0000 | `{"query_wrong_after_support_fit": 2, "query_correct": 1}` |
| bit_permutation | 239 | 0.3975 | 0.8996 | `{"query_wrong_after_support_fit": 123, "query_correct": 92, "support_incomplete": 24}` |
| equation_numeric | 40 | 0.5750 | 0.6750 | `{"query_correct": 23, "support_incomplete": 13, "query_wrong_after_support_fit": 4}` |
| cipher_char_sub | 236 | 0.8729 | 1.0000 | `{"query_correct": 206, "query_wrong_after_support_fit": 30}` |
| bit_rotate | 1 | 1.0000 | 1.0000 | `{"query_correct": 1}` |
