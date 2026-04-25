# Equation Template Ranker Design

## Purpose

The next local work item is to improve symbolic equation-template selection
before spending more GPU time. The current solver often finds programs that
fit every support example, but it chooses the wrong support-consistent program
for the held-out query.

This spec defines a local-only diagnostic and ranking loop for
`equation_template`. It is intended to raise local parsed-exact proxy quality
and to prevent wrong solver traces from entering strict training data.

## Current Evidence

Latest audit after retagging with current code:

| manifest | rows | overall query accuracy | equation_template accuracy | equation_template support-full |
| --- | ---: | ---: | ---: | ---: |
| `combined_balanced_48pf` | 288 | 0.7118 | 0.0256 | 1.0000 |
| `proxy_all_balanced_64pf` | 352 | 0.7330 | 0.0652 | 1.0000 |
| `hard_triad_full` | 709 | 0.4795 | 0.0737 | 1.0000 |

On `hard_triad_full`, `equation_template` has 190 rows. Of those, 176 are
`query_wrong_after_support_fit`. This makes the failure mode precise:

- support fitting is solved for this subtype;
- query extrapolation is not solved;
- the immediate bottleneck is candidate selection, not basic parsing.

The retagging fix also changed the interpretation of the old
`equation_position` gap. Most of those rows introduce literal symbols or
symbol counts not available from the paired input, so they are template-like
rather than pure position transductions.

## Goals

1. Add a repeatable local diagnostic for `equation_template` ambiguity.
2. Improve top-1 template candidate selection without broadening the operator
   search space first.
3. Mark high-risk symbolic-template rows so strict trace selection does not
   learn unreliable rationales.
4. Keep bit, cipher, numeric equation, unit, numeral, and gravity behavior
   unchanged except for incidental audit metadata.

## Non-Goals

- Do not train a new adapter as part of this work.
- Do not change Kaggle submission packaging.
- Do not add a neural reranker.
- Do not add a broad symbolic-program DSL until oracle-at-K proves the current
  candidate set is insufficient.

## Recommended Approach

Use a ranker/verifier layer before expanding `OperatorTemplateOp`.

The current search already produces support-perfect symbolic programs. Adding
more template expressivity now would likely increase the number of wrong
support-perfect candidates. A better next move is to measure ambiguity, rank
existing candidates more intelligently, and gate high-risk traces out of strict
training data.

## Alternatives Considered

### A. Ranker and Verifier First

Add diagnostics and scoring features for existing template candidates.

Pros:

- Lowest risk to existing solver coverage.
- Directly targets the observed support-fit/query-miss failure.
- Produces useful metadata for training-data gates.

Cons:

- Cannot solve examples whose true generator is not in the current candidate
  set.

### B. Expand Template Expressivity First

Generate more templates, such as multi-key templates or richer literal-source
templates.

Pros:

- May raise oracle upper bound.
- Useful if current candidates rarely contain the target query output.

Cons:

- Increases ambiguity and may reduce top-1 accuracy without a ranker.
- More likely to add brittle search behavior.

### C. Training-Data Filtering Only

Do not improve the solver. Only move ambiguous template rows out of strict
trace supervision.

Pros:

- Very safe for training.
- Reduces bad rationale contamination.

Cons:

- Does not improve local solver proxy.
- Slower path toward a 0.87-style task-system solution.

Decision: implement A, include the trace-gating metadata needed by C, and only
move to B if oracle-at-K is low.

## Design

### 1. Diagnostic Report

Add a local diagnostic script for symbolic equation templates. It should read
the labeled local manifests and emit JSON, CSV, and Markdown reports.

For each `equation_template` row, record:

- `id`, manifest path, support pairs, query, target, current prediction;
- top-K candidate predictions and step signatures;
- whether any top-K candidate matches the target query answer;
- support exact ratio for each candidate;
- template ambiguity count;
- query key status: seen key, unseen key, or no key;
- output length delta between support outputs and query prediction;
- target character provenance:
  - from query input;
  - from support inputs;
  - from support outputs;
  - unseen in all local row text;
- risk class:
  - `low_risk_support_stable`;
  - `ranker_miss_oracle_hit`;
  - `operator_gap_oracle_miss`;
  - `unseen_literal_high_risk`.

The main purpose is to separate ranker failures from operator-coverage
failures.

### 2. Template Candidate Ranking

Add a template-specific ranking score used after support exactness. This score
should not replace support correctness; it should break ties among candidates
that already fit support well.

Candidate features:

- lower literal count is better;
- fewer repeated source positions is better;
- fewer backward source-position jumps is better;
- less skipped source span is better;
- stable output length across matching support cases is better;
- query key seen in support is safer than unseen;
- literals observed in support outputs are safer than row-global unseen
  literals;
- leave-one-out support stability is better;
- a larger top1-top2 rank margin is better for strict trace use.

The ranking should be deterministic and exposed in candidate debug metadata so
the audit can explain why a candidate won.

### 3. Trace-Gating Metadata

Extend annotation metadata for symbolic templates:

- `template_oracle_rank`: rank of a query-correct candidate if present;
- `template_ambiguity_count`: number of support-perfect template candidates;
- `template_risk_class`: one of the diagnostic risk classes;
- `template_ranker_features`: compact feature summary for the selected top
  candidate.

Strict stage2 trace selection should accept only low-risk or query-verified
template examples when labels are available. High-risk rows may still enter an
answer-only silver pool, but they should not contribute generated reasoning
traces.

### 4. Audit Integration

`scripts/audit_solver_coverage.py` should remain the headline coverage audit.
The new diagnostic should complement it, not replace it.

After ranker changes:

1. run the template diagnostic;
2. run `scripts/audit_solver_coverage.py`;
3. compare `equation_template` accuracy, oracle-at-K, and risk-class counts;
4. verify unchanged or improved family-level results.

## Data Flow

```text
local eval manifests
  -> retag with current family_tagger
  -> ChainSearchEngine top-K candidates
  -> template diagnostic features
  -> template ranker tie-breaks
  -> solver coverage audit
  -> stage2 annotation metadata
  -> strict trace gate / answer-only silver gate
```

## Error Handling

- If a row has no candidates, classify it as `operator_gap_oracle_miss`.
- If top-K candidates cannot be evaluated on query, record the exception text
  in diagnostics and keep the row out of strict trace selection.
- If labels are absent, do not compute oracle rank or query-correct risk
  classes; use conservative support-only risk classes.
- If a manifest has stale subtype labels, diagnostics should retag by default
  with the current tagger.

## Tests

Add focused tests for:

- diagnostic classification on a tiny fixture with one top-1 hit, one
  oracle-at-2 hit, and one oracle miss;
- ranker preference for lower-risk templates when support predictions tie;
- query-seen-key versus query-unseen-key behavior;
- stage2 strict gate rejection of high-risk template traces;
- answer-only silver admission for high-risk but labeled official hard-triad
  samples;
- regression coverage for bit, cipher, numeric equation, and family retagging.

Run before committing implementation:

```bash
python -m pytest tests/test_atomic_ops.py tests/test_chain_search.py tests/test_family_tagger.py tests/test_stage2_silver_selection.py tests/test_stage2_subtype_rescue.py
python scripts/audit_solver_coverage.py
```

## Success Criteria

Primary:

- `hard_triad_full` `equation_template` query accuracy improves from `0.0737`
  to at least `0.1200`.
- Overall `hard_triad_full` query accuracy improves from `0.4795`.
- `combined_balanced_48pf` and `proxy_all_balanced_64pf` do not regress.

Secondary:

- Oracle-at-K is reported for `equation_template`.
- At least 80% of strict accepted template traces are low-risk or
  query-verified on labeled local data.
- No family-level regression greater than one sample on the balanced
  manifests.

Decision rule:

- If oracle-at-K is high but top-1 stays low, continue ranker work.
- If oracle-at-K is low, expand `OperatorTemplateOp` in a separate design and
  implementation plan.

## Implementation Boundaries

Expected write areas for the implementation phase:

- `scripts/diagnose_equation_template.py`
- `src/teacher/atomic_ops.py`
- `src/teacher/chain_search.py`
- `src/teacher/program_signature.py`
- `src/student/sft_dataset_builder.py`
- targeted tests under `tests/`
- coverage docs under `docs/`

The implementation should be done in small commits so each local proxy change
can be attributed to a specific solver/ranker behavior.
