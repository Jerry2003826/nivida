# Data Family Analysis

This is local-only. It summarizes distribution, not model quality.

## `data\official_kaggle\train.csv`

- rows: `9500`
- empty answers: `0`
- answer kinds: `{"numeric": 3847, "binary": 1634, "text_phrase": 1576, "roman": 1576, "text_token": 867}`

| family | n | share | prompt_mean | prompt_p90 | answer_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| bit | 1602 | 0.169 | 478.6 | 510 | 8.0 |
| gravity | 1597 | 0.168 | 320.3 | 353 | 5.0 |
| unit | 1594 | 0.168 | 222.2 | 245 | 4.9 |
| cipher | 1576 | 0.166 | 370.1 | 437 | 25.5 |
| numeral | 1576 | 0.166 | 219.1 | 233 | 4.0 |
| equation | 1555 | 0.164 | 195.1 | 208 | 2.9 |

Top subtypes:

`{"bit:unknown": 1602, "gravity:unknown": 1597, "unit:unknown": 1594, "cipher:unknown": 1576, "numeral:unknown": 1576, "equation:unknown": 1555}`

## `data\official_kaggle\test.csv`

- rows: `3`
- empty answers: `3`
- answer kinds: `{"missing": 3}`

| family | n | share | prompt_mean | prompt_p90 | answer_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| bit | 2 | 0.667 | 478.5 | 489 | 0.0 |
| cipher | 1 | 0.333 | 458.0 | 458 | 0.0 |

Top subtypes:

`{"bit:unknown": 2, "cipher:unknown": 1}`

## `data\processed\stage2_official_valid_hard_triad.jsonl`

- rows: `709`
- empty answers: `0`
- answer kinds: `{"binary": 242, "text_phrase": 236, "text_token": 193, "numeric": 38}`

| family | n | share | prompt_mean | prompt_p90 | answer_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| bit | 240 | 0.339 | 471.2 | 510 | 8.0 |
| cipher | 236 | 0.333 | 373.4 | 438 | 25.2 |
| equation | 233 | 0.329 | 194.2 | 208 | 2.9 |

Top subtypes:

`{"bit:bit_permutation": 239, "cipher:cipher_char_sub": 236, "equation:equation_position": 193, "equation:equation_numeric": 40, "bit:bit_rotate": 1}`

## `data\processed\proxy_all_family_valid.jsonl`

- rows: `907`
- empty answers: `0`
- answer kinds: `{"numeric": 499, "roman": 236, "text_token": 75, "binary": 58, "text_phrase": 39}`

| family | n | share | prompt_mean | prompt_p90 | answer_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| gravity | 239 | 0.264 | 322.7 | 354 | 5.0 |
| unit | 239 | 0.264 | 221.4 | 245 | 4.9 |
| numeral | 236 | 0.260 | 219.4 | 233 | 4.2 |
| equation | 97 | 0.107 | 192.5 | 206 | 2.8 |
| bit | 57 | 0.063 | 472.8 | 489 | 8.0 |
| cipher | 39 | 0.043 | 380.5 | 448 | 25.7 |

Top subtypes:

`{"gravity:gravity_inverse_square": 239, "unit:unit_scale": 239, "numeral:numeral_roman": 236, "equation:equation_position": 75, "bit:bit_permutation": 57, "cipher:cipher_char_sub": 39, "equation:equation_numeric": 22}`

## `data\processed\stage2_distill_train.jsonl`

- rows: `9683`
- empty answers: `2`
- answer kinds: `{"numeric": 3159, "text_phrase": 2720, "binary": 2548, "text_token": 1172, "roman": 82, "missing": 2}`

| family | n | share | prompt_mean | prompt_p90 | answer_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| cipher | 3168 | 0.327 | 500.4 | 600 | 20.1 |
| bit | 2534 | 0.262 | 585.2 | 681 | 7.8 |
| gravity | 1530 | 0.158 | 492.9 | 524 | 5.3 |
| unit | 1431 | 0.148 | 394.7 | 417 | 4.9 |
| equation | 956 | 0.099 | 389.9 | 425 | 2.9 |
| numeral | 64 | 0.007 | 389.8 | 404 | 4.2 |

Top subtypes:

`{"cipher:cipher_char_sub": 2974, "bit:bit_permutation": 1854, "gravity:gravity_inverse_square": 1530, "unit:unit_scale": 1401, "equation:equation_position": 624, "bit:bit_xor_mask": 260, "bit:bit_rotate": 258, "bit:bit_nibble": 162, "equation:equation_delete": 132, "cipher:cipher_token_sub": 126, "equation:equation_numeric": 112, "equation:equation_symbolic": 88}`

## `data\processed\stage2_distill_valid.jsonl`

- rows: `667`
- empty answers: `0`
- answer kinds: `{"text_phrase": 236, "binary": 219, "text_token": 193, "numeric": 19}`

| family | n | share | prompt_mean | prompt_p90 | answer_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| cipher | 236 | 0.354 | 544.4 | 609 | 25.2 |
| bit | 217 | 0.325 | 639.4 | 660 | 8.0 |
| equation | 214 | 0.321 | 365.1 | 379 | 2.9 |

Top subtypes:

`{"cipher:cipher_char_sub": 236, "bit:bit_permutation": 216, "equation:equation_position": 193, "equation:equation_numeric": 21, "bit:bit_rotate": 1}`
