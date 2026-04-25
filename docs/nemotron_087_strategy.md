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

| manifest | rows | query accuracy | oracle@k | support-full rate | read |
| --- | ---: | ---: | ---: | ---: | --- |
| `combined_balanced_48pf` | 288 | 0.7292 | 0.7396 | 0.9965 | cipher mostly solved; bit/equation weak |
| `proxy_all_balanced_64pf` | 352 | 0.7472 | 0.7528 | 0.9744 | same pattern |
| `hard_triad_full` | 709 | 0.5035 | 0.5303 | 0.9803 | hard triad is still the gap |

The important signal is the high support-full rate with low query accuracy.
The rule search often fits all demonstrations but extrapolates the held-out
query incorrectly. That means the missing capability is not "parse examples"
but "identify the true generator among many demonstration-consistent programs."

The local metric now mirrors the official binary guard: pure `0/1` answers are
strict strings, not numeric values with 1% tolerance. This intentionally lowers
reported bit accuracy compared with older audits; those older bit scores were
partly false positives.

## Family Priorities

### Tier 0: Do Not Spend Main Effort

These families are already nearly solved by the existing rule system:

- `gravity_inverse_square`
- `numeral_roman`
- `unit_scale`

Use them for regression checks and format stability, not as the main source of
future gains.

### Tier 1: Highest Value

Symbolic equation templates previously mislabeled as `equation_position`

- `combined_balanced_48pf`: 0.0256 query accuracy
- `proxy_all_balanced_64pf`: 0.0870
- `hard_triad_full`: 0.0842
- support-full is near 1.0, so the search is overfitting demonstrations.

This is the clearest system-cracking target.

The first fix now prefers lower-risk operator templates: fewer literals,
fewer repeated source positions, and more monotonic source-position reuse. It
raised equation-position coverage without changing bit/cipher outcomes, but
most support-consistent ambiguity remains.

The subtype label itself was also too broad: any symbolic equation whose output
was not longer than the input was tagged `equation_position`. The corrected
tagger only uses `equation_position` when output characters can be supplied by
the paired input. Literal-introducing outputs are now `equation_template`; on
the current hard-triad manifest, audit retagging moves 190 / 193 old position
rows to template-like.

The latest equation-template diagnostic makes the next bottleneck clearer:
across the three local manifests, only 23 / 275 rows are top-1 correct and
oracle-at-10. The hard-triad slice is similar: 17 / 190 top-1 and oracle-at-10.
Keeping support-equivalent literal alternatives, adding a small query-copy
tie-break, and preferring support-modal output length in symbolic ties helped a
few ambiguous cases, but most misses remain `operator_gap_oracle_miss`, so pure
reranking cannot solve the bulk of this subtype; the operator/template search
space itself needs expansion.

The target-expressibility split tightens that diagnosis. If the labeled
query-target pair is added as one more support example, current operators can
fit 207 / 275 rows, but only 22 rows have an `operator_template` fit where the
query key was already seen in the support set. The dominant failure is now
`unseen_key_template_miss` (135 / 275): the answer is compatible with a
template, but it requires a key character whose template was not demonstrated.
That is unsafe trace supervision, not a normal ranker miss. The remaining
operator-space gap is 58 / 275.

Strict stage2 trace selection now rejects `equation_template` traces labeled as
`ranker_miss_oracle_hit`, `operator_gap_oracle_miss`, or
`unseen_key_template_miss`, or `unseen_literal_high_risk`. Those samples can
still be used as answer-only silver supervision, but not as
chain-of-thought/rationale traces.

Numeric equations are a smaller but cleaner target. Query-aware operator priors
and lookup fallbacks lifted `hard_triad_full` `equation_numeric` from `0.4500`
to `0.5750` without hurting the other families.

### Tier 2

`cipher_char_sub`

- support-full is 1.0, and query accuracy is now 0.82-0.87 after prioritizing
  `vocabulary_cipher` over raw `fixed_substitution` for char-substitution
  prompts.
- Remaining misses are true vocabulary/map-completion ambiguity cases.

### Tier 3

`bit_permutation`

- query accuracy is now 0.3125-0.4561 after adding a per-output-bit
  `binary_boolean_expr` operator for constants, copy, NOT, AND/OR/XOR/XNOR,
  NAND/NOR, three-input AND/OR/XOR/XNOR, majority, and choice.
- `hard_triad_full` bit accuracy rose from 0.4000 to 0.4583, and
  `bit_permutation` rose from 0.3975 to 0.4561. The oracle@k gap is now
  positive (`hard_triad_full`: 0.5314 oracle@k vs 0.4561 top1), so there is
  some remaining ranker upside after candidate generation.
- support failures are mostly gone, but many failures are still wrong
  extrapolation from support-perfect boolean expressions.
- Need a stronger permutation/affine disambiguation strategy. The current
  `binary_affine_transform` is still too expressive and often support-fits a
  simpler hidden generator. The boolean-expression operator captures nonlinear
  official prompt language, but it is wider and makes audits slower, so future
  bit work should either narrow this operator by subtype or add verifier
  features before more training.
- `scripts/diagnose_bit_permutation.py` now breaks this down directly: across
  the three local manifests, `bit_permutation` has 152 low-risk top1 hits, 40
  ranker-miss/oracle-hit rows, 151 operator-gap rows, and 1 support-incomplete
  row. This makes the next bit target concrete: ranker/verifier features can
  recover a bounded set, but more operator coverage is still the larger gap.

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
python scripts/diagnose_equation_template.py
python scripts/diagnose_bit_permutation.py
```

Then open:

- `docs/solver_coverage_audit_latest.md`
- `docs/equation_template_diagnostic_latest.md`
- `docs/bit_permutation_diagnostic_latest.md`
- `data/processed/solver_coverage_records.csv`

Start with:

```text
failure_class = query_wrong_after_support_fit
subtype = equation_template
```

The next engineering target is a stricter symbolic-equation template
operator expansion plus verifier/ranker that detects underconstrained query
paths, not just perfect support fit. Labeled examples now record whether the
solver's query prediction matches the known target; strict stage2 trace
selection rejects query-mismatch and high-risk template traces so wrong
support-fitting programs do not become training rationales.

## Training Policy

Training remains useful, as proven by `0.57`, but it is downstream of solver
quality now. The preferred training source is no longer raw teacher traces; it
is solver-verified, family-balanced, ambiguity-filtered data.

Do not spend GPU time on more route/shared transplant experiments unless solver
coverage or exact-eval gives a clear reason.
