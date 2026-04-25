# Nemotron 0.87 Strategy

## Decision

The objective is no longer "make the LoRA loss go down" or "squeeze another
public point from continuation training." The objective is now:

> Build a task-system solver/verifier for each official family, use it to
> generate and validate high-confidence teacher data, then train LoRA to
> internalize that system.

This is the path that plausibly explains `0.87`: not a slightly better adapter,
but stronger coverage of the puzzle generators.

## Current Evidence

Public leaderboard:

- official-balanced continuation: `0.57`
- prior B thin / norm-shared: `0.54`
- routeweighted shared: `0.53`

Solver coverage audit on local manifests:

| manifest | rows | query accuracy | support-full rate | read |
| --- | ---: | ---: | ---: | --- |
| `combined_balanced_48pf` | 288 | 0.6528 | 0.9722 | easy families solved; hard triad weak |
| `proxy_all_balanced_64pf` | 352 | 0.6960 | 0.9545 | same pattern |
| `hard_triad_full` | 709 | 0.3399 | 0.9408 | hard triad is the gap |

The important signal is the high support-full rate with low query accuracy.
The rule search often fits all demonstrations but extrapolates the held-out
query incorrectly. That means the missing capability is not "parse examples"
but "identify the true generator among many demonstration-consistent programs."

## Family Priorities

### Tier 0: Do Not Spend Main Effort

These families are already nearly solved by the existing rule system:

- `gravity_inverse_square`
- `numeral_roman`
- `unit_scale`

Use them for regression checks and format stability, not as the main source of
future gains.

### Tier 1: Highest Value

`equation_position`

- `combined_balanced_48pf`: 0.0256 query accuracy
- `proxy_all_balanced_64pf`: 0.0833
- `hard_triad_full`: 0.0777
- support-full is near 1.0, so the search is overfitting demonstrations.

This is the clearest system-cracking target.

The first fix now prefers lower-risk operator templates: fewer literals,
fewer repeated source positions, and more monotonic source-position reuse. It
raised equation-position coverage without changing bit/cipher outcomes, but
most support-consistent ambiguity remains.

### Tier 2

`cipher_char_sub`

- support-full is 1.0, but query accuracy is only 0.25-0.44.
- Existing `char_sub` hypothesis is too underdetermined for the query.
- Need better map completion and ambiguity handling.

### Tier 3

`bit_permutation`

- query accuracy is 0.33-0.60 depending on manifest.
- support failures still exist, but many failures are also wrong extrapolation.
- Need a stronger permutation/affine disambiguation strategy.

## New Work Loop

1. Run `scripts/audit_solver_coverage.py`.
2. Pick the weakest subtype by query accuracy and support-full rate.
3. Inspect wrong-after-support-fit records.
4. Add or refine a deterministic operator/verifier.
5. Regenerate high-confidence teacher data.
6. Train only after solver coverage improves.
7. Submit only after local exact ranking beats the official-balanced baseline.

## Next Local Actions

Use the audit output:

```bash
python scripts/audit_solver_coverage.py
```

Then open:

- `docs/solver_coverage_audit_latest.md`
- `data/processed/solver_coverage_records.csv`

Start with:

```text
failure_class = query_wrong_after_support_fit
subtype = equation_position
```

The next engineering target is a stricter `equation_position` verifier/ranker
that detects underconstrained query paths, not just perfect support fit. Labeled
examples now record whether the solver's query prediction matches the known
target; strict stage2 trace selection rejects query-mismatch traces so wrong
support-fitting programs do not become training rationales.

## Training Policy

Training remains useful, as proven by `0.57`, but it is downstream of solver
quality now. The preferred training source is no longer raw teacher traces; it
is solver-verified, family-balanced, ambiguity-filtered data.

Do not spend GPU time on more route/shared transplant experiments unless solver
coverage or exact-eval gives a clear reason.
