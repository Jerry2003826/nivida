# Solver Coverage Audit

This measures how much of the labeled local eval set is already explainable by the rule/search system.

## `data\processed\local_eval_manifests\combined_balanced_48pf.jsonl`

- rows: `288`
- query accuracy: `0.7604`
- oracle@k: `0.7674`
- support-full rate: `0.9896`
- avg support accuracy: `0.9915`
- failure classes: `{"query_correct": 219, "query_wrong_after_support_fit": 66, "support_incomplete": 3}`

| family | n | query_acc | oracle@k | support_full | avg_support | failures | top signature buckets |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bit | 48 | 0.3542 | 0.3750 | 0.9583 | 0.9646 | `{"query_wrong_after_support_fit": 29, "query_correct": 17, "support_incomplete": 2}` | `{"binary_boolean_expr": 45, "bitwise_or_constant>rotl": 1, "binary_invert>rotl>bitwise_or_constant": 1, "affine": 1}` |
| cipher | 48 | 0.9583 | 0.9583 | 1.0000 | 1.0000 | `{"query_correct": 46, "query_wrong_after_support_fit": 2}` | `{"vocab_sub": 48}` |
| equation | 48 | 0.2500 | 0.2708 | 0.9792 | 0.9844 | `{"query_wrong_after_support_fit": 35, "query_correct": 12, "support_incomplete": 1}` | `{"template": 24, "eq_rule": 13, "template>=[[lit,&],[lit,&],[pos,1]]": 1, "position>=,{=,%=,@=>,[=,\\=+,(={,+=,?=\\,%=>=/,[=/,": 1, "template>=[[pos,2],[lit,^],[pos,4]],\\=[[pos,0],[pos,1],[pos,3],[pos,4]],}=[[lit,@],[lit,": 1, "template>],[lit,[],[lit,?]]": 1, "evaluate_expression>add": 1, "template>],[pos,0],[lit,$]],`=[[lit,$],[pos,4],[lit,&],[lit,!]],|=[[lit,[],[lit,$],[pos,1]]": 1}` |
| gravity | 48 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` | `{"inverse_square": 48}` |
| numeral | 48 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` | `{"dec2roman": 48}` |
| unit | 48 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` | `{"scale": 48}` |

### Weakest Subtypes

| subtype | n | query_acc | oracle@k | support_full | failures |
| --- | ---: | ---: | ---: | ---: | --- |
| equation_template | 33 | 0.1212 | 0.1515 | 1.0000 | `{"query_wrong_after_support_fit": 29, "query_correct": 4}` |
| bit_permutation | 48 | 0.3542 | 0.3750 | 0.9583 | `{"query_wrong_after_support_fit": 29, "query_correct": 17, "support_incomplete": 2}` |
| equation_numeric | 14 | 0.5000 | 0.5000 | 0.9286 | `{"query_correct": 7, "query_wrong_after_support_fit": 6, "support_incomplete": 1}` |
| cipher_char_sub | 48 | 0.9583 | 0.9583 | 1.0000 | `{"query_correct": 46, "query_wrong_after_support_fit": 2}` |
| gravity_inverse_square | 48 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` |
| numeral_roman | 48 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` |
| unit_scale | 48 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 48}` |
| equation_position | 1 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 1}` |

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
| equation | 64 | 0.1719 | 0.1719 | 0.8594 | 0.8971 | `{"query_wrong_after_support_fit": 44, "query_correct": 11, "support_incomplete": 9}` | `{"template": 32, "evaluate_expression>add": 8, "eq_rule": 7, "template>]],]=[[lit,#],[lit,[],[pos,4]]": 1, "template>=[[lit,&],[lit,&],[pos,1]]": 1, "position>=,{=,%=,@=>,[=,\\=+,(={,+=,?=\\,%=>=/,[=/,": 1, "template>=[[pos,3],[lit,\"],[lit,|],[lit,": 1, "evaluate_expression>affine": 1}` |
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
- query accuracy: `0.5049`
- oracle@k: `0.5275`
- support-full rate: `0.9803`
- avg support accuracy: `0.9863`
- failure classes: `{"query_correct": 358, "query_wrong_after_support_fit": 337, "support_incomplete": 14}`

| family | n | query_acc | oracle@k | support_full | avg_support | failures | top signature buckets |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bit | 240 | 0.4042 | 0.4583 | 0.9875 | 0.9892 | `{"query_wrong_after_support_fit": 140, "query_correct": 97, "support_incomplete": 3}` | `{"binary_boolean_expr": 224, "affine": 9, "binary_invert>binary_boolean_expr": 2, "rotl>nibble": 1, "reverse_bits>rotl>bitwise_or_constant": 1, "bitwise_or_constant>rotl": 1, "binary_invert>rotl>bitwise_or_constant": 1, "rotl": 1}` |
| cipher | 236 | 0.8729 | 0.8729 | 1.0000 | 1.0000 | `{"query_correct": 206, "query_wrong_after_support_fit": 30}` | `{"vocab_sub": 236}` |
| equation | 233 | 0.2361 | 0.2489 | 0.9528 | 0.9694 | `{"query_wrong_after_support_fit": 167, "query_correct": 55, "support_incomplete": 11}` | `{"template": 103, "eq_rule": 63, "position": 11, "evaluate_expression>add": 9, "template>]]": 4, "evaluate_expression>affine": 2, "template>],[pos,4]],$=[[lit,}],[pos,3],[pos,4],[lit,}]],%=[[lit,$],[lit,)]]": 1, "template>],[pos,3]]": 1}` |

### Weakest Subtypes

| subtype | n | query_acc | oracle@k | support_full | failures |
| --- | ---: | ---: | ---: | ---: | --- |
| equation_template | 156 | 0.0769 | 0.0897 | 1.0000 | `{"query_wrong_after_support_fit": 144, "query_correct": 12}` |
| bit_permutation | 239 | 0.4017 | 0.4561 | 0.9874 | `{"query_wrong_after_support_fit": 140, "query_correct": 96, "support_incomplete": 3}` |
| equation_numeric | 77 | 0.5584 | 0.5714 | 0.8571 | `{"query_correct": 43, "query_wrong_after_support_fit": 23, "support_incomplete": 11}` |
| cipher_char_sub | 236 | 0.8729 | 0.8729 | 1.0000 | `{"query_correct": 206, "query_wrong_after_support_fit": 30}` |
| bit_rotate | 1 | 1.0000 | 1.0000 | 1.0000 | `{"query_correct": 1}` |
