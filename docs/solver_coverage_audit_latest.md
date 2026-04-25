# Solver Coverage Audit

This measures how much of the labeled local eval set is already explainable by the rule/search system.

## `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl`

- rows: `288`
- query accuracy: `0.7257`
- oracle@k: `0.7361`
- support-full rate: `0.9965`
- avg support accuracy: `0.9979`
- failure classes: `{"query_correct": 209, "query_wrong_after_support_fit": 78, "support_incomplete": 1}`

| family | n | query_acc | oracle@k | support_full | avg_support | failures | top signature buckets |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bit | 48 | 0.3125 | 0.3750 | 1.0000 | 1.0000 | `{"query_wrong_after_support_fit": 33, "query_correct": 15}` | `{"binary_boolean_expr": 46, "affine": 2}` |
| cipher | 48 | 0.8750 | 0.8750 | 1.0000 | 1.0000 | `{"query_correct": 42, "query_wrong_after_support_fit": 6}` | `{"vocab_sub": 48}` |
| equation | 48 | 0.1667 | 0.1667 | 0.9792 | 0.9875 | `{"query_wrong_after_support_fit": 39, "query_correct": 8, "support_incomplete": 1}` | `{"template": 24, "eq_rule": 8, "position": 4, "template>]],[=[[pos,3],[lit,>],[lit,]]],`=[[pos,0],[lit,?],[lit,?]]": 1, "position>=,{=,%=,@=>,[=,\\=+,(={,+=,?=\\,%=>=/,[=/,": 1, "template>=[[pos,2],[lit,^],[pos,4]],\\=[[pos,0],[pos,1],[pos,3],[pos,4]],}=[[lit,@],[lit,": 1, "evaluate_expression>affine": 1, "template>=[[lit,|],[lit,]]],{=[[lit,!],[lit,|],[lit,|],[pos,1]]": 1}` |
| gravity | 48 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` | `{"inverse_square": 48}` |
| numeral | 48 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` | `{"dec2roman": 48}` |
| unit | 48 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` | `{"scale": 44, "scale>scale": 4}` |

### Weakest Subtypes

| subtype | n | query_acc | oracle@k | support_full | failures |
| --- | ---: | ---: | ---: | ---: | --- |
| equation_template | 39 | 0.0256 | 0.0256 | 1.0000 | `{"query_wrong_after_support_fit": 38, "query_correct": 1}` |
| bit_permutation | 48 | 0.3125 | 0.3750 | 1.0000 | `{"query_wrong_after_support_fit": 33, "query_correct": 15}` |
| equation_numeric | 9 | 0.7778 | 0.7778 | 0.8889 | `{"query_correct": 7, "query_wrong_after_support_fit": 1, "support_incomplete": 1}` |
| cipher_char_sub | 48 | 0.8750 | 0.8750 | 1.0000 | `{"query_correct": 42, "query_wrong_after_support_fit": 6}` |
| gravity_inverse_square | 48 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` |
| numeral_roman | 48 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` |
| unit_scale | 48 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` |

## `data\processed\local_eval_manifests\proxy_all_balanced_64pf.jsonl`

- rows: `352`
- query accuracy: `0.7472`
- oracle@k: `0.7528`
- support-full rate: `0.9744`
- avg support accuracy: `0.9813`
- failure classes: `{"query_correct": 263, "query_wrong_after_support_fit": 80, "support_incomplete": 9}`

| family | n | query_acc | oracle@k | support_full | avg_support | failures | top signature buckets |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bit | 57 | 0.4912 | 0.5263 | 1.0000 | 1.0000 | `{"query_wrong_after_support_fit": 29, "query_correct": 28}` | `{"binary_boolean_expr": 52, "affine": 3, "binary_invert>rotl>nibble": 1, "nibble": 1}` |
| cipher | 39 | 0.8205 | 0.8205 | 1.0000 | 1.0000 | `{"query_correct": 32, "query_wrong_after_support_fit": 7}` | `{"vocab_sub": 39}` |
| equation | 64 | 0.1719 | 0.1719 | 0.8594 | 0.8971 | `{"query_wrong_after_support_fit": 44, "query_correct": 11, "support_incomplete": 9}` | `{"template": 31, "evaluate_expression>add": 8, "eq_rule": 7, "position": 2, "template>]],]=[[lit,#],[lit,[],[pos,4]]": 1, "template>=[[lit,&],[lit,&],[pos,1]]": 1, "position>=,{=,%=,@=>,[=,\\=+,(={,+=,?=\\,%=>=/,[=/,": 1, "template>=[[pos,3],[lit,\"],[lit,|],[lit,": 1}` |
| gravity | 64 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 64}` | `{"inverse_square": 64}` |
| numeral | 64 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 64}` | `{"dec2roman": 64}` |
| unit | 64 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 64}` | `{"scale": 60, "scale>scale": 4}` |

### Weakest Subtypes

| subtype | n | query_acc | oracle@k | support_full | failures |
| --- | ---: | ---: | ---: | ---: | --- |
| equation_template | 46 | 0.0870 | 0.0870 | 1.0000 | `{"query_wrong_after_support_fit": 42, "query_correct": 4}` |
| equation_numeric | 16 | 0.3750 | 0.3750 | 0.4375 | `{"support_incomplete": 9, "query_correct": 6, "query_wrong_after_support_fit": 1}` |
| bit_permutation | 57 | 0.4912 | 0.5263 | 1.0000 | `{"query_wrong_after_support_fit": 29, "query_correct": 28}` |
| equation_position | 2 | 0.5000 | 0.5000 | 1.0000 | `{"query_correct": 1, "query_wrong_after_support_fit": 1}` |
| cipher_char_sub | 39 | 0.8205 | 0.8205 | 1.0000 | `{"query_correct": 32, "query_wrong_after_support_fit": 7}` |
| gravity_inverse_square | 64 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 64}` |
| numeral_roman | 64 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 64}` |
| unit_scale | 64 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 64}` |

## `data\processed\local_eval_manifests\hard_triad_full.jsonl`

- rows: `709`
- query accuracy: `0.5021`
- oracle@k: `0.5289`
- support-full rate: `0.9803`
- avg support accuracy: `0.9860`
- failure classes: `{"query_correct": 356, "query_wrong_after_support_fit": 339, "support_incomplete": 14}`

| family | n | query_acc | oracle@k | support_full | avg_support | failures | top signature buckets |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bit | 240 | 0.4583 | 0.5333 | 0.9958 | 0.9967 | `{"query_wrong_after_support_fit": 129, "query_correct": 110, "support_incomplete": 1}` | `{"binary_boolean_expr": 222, "affine": 11, "binary_invert>binary_boolean_expr": 2, "binary_invert>rotl>nibble": 1, "nibble": 1, "rotl>nibble": 1, "binary_invert>rotl>bitwise_or_constant": 1, "rotl": 1}` |
| cipher | 236 | 0.8729 | 0.8729 | 1.0000 | 1.0000 | `{"query_correct": 206, "query_wrong_after_support_fit": 30}` | `{"vocab_sub": 236}` |
| equation | 233 | 0.1717 | 0.1760 | 0.9442 | 0.9608 | `{"query_wrong_after_support_fit": 180, "query_correct": 40, "support_incomplete": 13}` | `{"template": 129, "eq_rule": 26, "evaluate_expression>add": 12, "position": 11, "template>]]": 2, "template>]],]=[[lit,#],[lit,[],[pos,4]]": 1, "template>],[lit,\"]],]=[[lit,'],[pos,0]]": 1, "template>],[pos,3]]": 1}` |

### Weakest Subtypes

| subtype | n | query_acc | oracle@k | support_full | failures |
| --- | ---: | ---: | ---: | ---: | --- |
| equation_template | 190 | 0.0842 | 0.0842 | 1.0000 | `{"query_wrong_after_support_fit": 174, "query_correct": 16}` |
| equation_position | 3 | 0.3333 | 0.3333 | 1.0000 | `{"query_wrong_after_support_fit": 2, "query_correct": 1}` |
| bit_permutation | 239 | 0.4561 | 0.5314 | 0.9958 | `{"query_wrong_after_support_fit": 129, "query_correct": 109, "support_incomplete": 1}` |
| equation_numeric | 40 | 0.5750 | 0.6000 | 0.6750 | `{"query_correct": 23, "support_incomplete": 13, "query_wrong_after_support_fit": 4}` |
| cipher_char_sub | 236 | 0.8729 | 0.8729 | 1.0000 | `{"query_correct": 206, "query_wrong_after_support_fit": 30}` |
| bit_rotate | 1 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 1}` |
