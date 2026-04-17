# Nemotron Reasoning Challenge — CoE Takeover Plan for Codex

## 0. Mission

Build a **competition-oriented offline teacher + final LoRA student** system for the **NVIDIA Nemotron Model Reasoning Challenge**.

The final submitted artifact must be a **LoRA adapter** for **NVIDIA Nemotron-3-Nano-30B**, packaged as `submission.zip`, with **LoRA rank <= 32**. The competition encourages **prompting, data pipelines, and lightweight fine-tuning** on a benchmark of **logical reasoning puzzles based on transformation rules**. Each team can submit **up to 5 times per day** and mark **up to 2 final submissions**. Use those constraints as hard design requirements. Do not build a system that depends on external runtime tools at inference time.

## 1. Core strategic decision

Do **not** try to turn the published Chain-of-Event repository into the final competition model.

Instead:

- Use the **paper’s abstraction** as inspiration:
  - an **overall event-causal graph**,
  - an **incident-specific graph**,
  - **edge weights = causal likelihood**,
  - **node weights = importance / starting prior**,
  - graph reasoning used as an **interpretable teacher**.
- Translate that abstraction into a **reasoning puzzle domain**:
  - “event” -> **atomic transformation operation**,
  - “incident” -> **one puzzle instance**,
  - “event-causal graph” -> **rule-transition graph / chain-of-transformation graph**.
- Use the graph system **offline only** for:
  - rule-family discovery,
  - solver search,
  - pseudo-label filtering,
  - synthetic data generation,
  - curriculum construction,
  - hard-case mining.
- Train a **Nemotron LoRA student** on the resulting data.
- Final submission = **LoRA only**.

## 2. Why we are not “repairing the repo” as the main path

The public Chain-of-Event repo is a lightweight research prototype, not a competition-ready foundation:

- the repo is minimal, with only a small folder set and **1 commit**;
- the README is extremely thin;
- the current scripts are tightly coupled to a specific incident dataset layout and internal service topology;
- the config references internal endpoints and domain-specific pool lists;
- the dependency declaration is incomplete relative to imported modules;
- some code paths appear inconsistent.

Therefore, Codex should **salvage the idea**, not preserve the implementation.

## 3. Hard constraints to encode everywhere

1. Final artifact must be a **LoRA adapter**.
2. **Rank must not exceed 32**.
3. Assume final inference environment is fixed; do not rely on external APIs, custom retrieval, or solver services at submission time.
4. Optimize for **final boxed answer accuracy**, not verbose reasoning quality.
5. Public leaderboard is useful but should not dominate design; maintain strong offline validation.
6. Since only **5 submissions/day** and **2 final picks** are allowed, the local evaluator must be treated as first-class infra.

## 4. Competition interpretation

Treat the task as:

> learn to infer the hidden transformation rule from examples, then apply it to a held-out query instance, and produce a stable final answer format.

This means we need three layers:

1. **rule understanding** — identify the latent operation family,
2. **rule composition** — combine multiple atomic operations if needed,
3. **format reliability** — always emit one valid final answer.

## 5. Translation of CoE into this competition

### 5.1 Concept mapping

- **Overall event-causal graph** -> **global operation-transition graph**
- **Incident-specific event-causal graph** -> **puzzle-specific candidate rule graph**
- **Event importance** -> **prior probability of an operation being the start of a valid chain**
- **Causal edge weight** -> **probability that operation B follows operation A in a valid transformation chain**
- **Graph ranking / propagation** -> **search/ranking over candidate solution chains**
- **Human knowledge integration** -> **hand-authored rule priors, constraints, and curricula**

### 5.2 What survives from the paper

Keep the following ideas:

- the distinction between **global graph** and **instance-specific graph**,
- **continuous weights** rather than only hard binary rules,
- graph-guided interpretability,
- graph-guided ranking over plausible causes / rules,
- optional human modification of priors.

### 5.3 What we drop

Drop or ignore:

- eBay-specific configs,
- incident ticket parsing,
- microservice pool topology assumptions,
- GAT/BERT plumbing from the public repo,
- anything depending on internal monitoring formats.

## 6. New architecture to implement

Create a new repo or new top-level package inside the repo called `nemotron_reasoning/`.

### 6.1 Directory layout

```text
nemotron_reasoning/
  configs/
    default.yaml
    data.yaml
    train.yaml
    synth.yaml
    eval.yaml
  src/
    data/
      parse_kaggle.py
      normalize_answer.py
      split_builder.py
      feature_extractors.py
      schema.py
    rules/
      atomic_ops.py
      composed_ops.py
      rule_families.py
      rule_graph.py
      rule_search.py
      solver_interface.py
      confidence.py
    synth/
      synth_templates.py
      synth_generator.py
      hard_negative_generator.py
      curriculum.py
    labeling/
      pseudo_label.py
      rationale_filter.py
      teacher_distill.py
    training/
      build_sft_dataset.py
      train_lora.py
      train_format_only.py
      pack_submission.py
    eval/
      boxed_parser.py
      metric.py
      offline_eval.py
      family_eval.py
      composition_eval.py
      ablation.py
    utils/
      io.py
      logging.py
      seeds.py
      registry.py
  scripts/
    run_parse.sh
    run_synth.sh
    run_train_stage_a.sh
    run_train_stage_b.sh
    run_eval.sh
  tests/
    test_parser.py
    test_atomic_ops.py
    test_boxed_parser.py
    test_metric.py
    test_rule_search.py
  README.md
```

## 7. Data model

### 7.1 Canonical puzzle object

Define a canonical schema for one training/test row:

```python
PuzzleExample = {
  "id": str,
  "raw_prompt": str,
  "train_pairs": [
    {"input": str, "output": str},
    ...
  ],
  "query_input": str,
  "target_answer": str | None,
  "metadata": {
    "source": str,
    "split": str,
    "family_tags": list[str],
    "difficulty": float | None,
  }
}
```

### 7.2 Required parser behavior

`parse_kaggle.py` must:

- ingest official train files,
- robustly extract example pairs and query input,
- normalize whitespace and punctuation while preserving exact semantic content,
- preserve raw prompt for later formatting ablations,
- emit canonical JSONL.

### 7.3 Feature extraction

`feature_extractors.py` should derive cheap structural features:

- input/output length deltas,
- character-set overlap,
- digit/letter/symbol ratios,
- positional displacement signatures,
- repeated-token patterns,
- numeric transform hints,
- base-conversion hints,
- bitwise hints,
- permutation/reversal indicators.

These features are for routing, clustering, rule-family tagging, and hard-case mining.

## 8. Rule library

### 8.1 Atomic operations to implement first

Implement deterministic operators with forward application and eligibility checks.

#### String / sequence ops
- identity
- reverse
- rotate left/right by k
- chunk reverse
- interleave
- deinterleave
- sort characters
- stable unique
- substring extraction
- mirror by alphabet index
- Caesar shift
- substitution cipher by learned map
- category-wise replace (digit->digit, vowel->vowel, etc.)
- pairwise swap
- index-based reordering

#### Numeric ops
- add/subtract/multiply/divide by constant
- affine transform
- modulo
- digit sum/product
- parity mapping
- absolute difference chain
- recurrence with short order

#### Representation ops
- decimal/binary/octal/hex conversion
- Roman or custom numeral conversion if seen
- ASCII/Unicode char-code transform
- bitwise xor/and/or/not with constant
- bit shift
- nibble/byte reversal

#### Equation / symbolic ops
- move term across equality
- substitute variable
- simplify constant side
- expand / factor only if deterministic and validated

### 8.2 Operator interface

Every op must implement:

```python
class AtomicOp(Protocol):
    name: str
    arity: int = 1
    def fit(self, examples: list[tuple[str, str]]) -> FitResult: ...
    def apply(self, x: str, params: dict) -> str: ...
    def score(self, examples: list[tuple[str, str]], params: dict) -> float: ...
```

Where `fit()` returns learned params plus a confidence score.

### 8.3 Composed ops

Support composition lengths 2 to 4 first. That is enough to be useful without exploding search.

## 9. Rule graph system

### 9.1 Global rule graph

`rule_graph.py` should maintain:

- node set = atomic ops,
- node prior = estimated start probability,
- directed edges = op transition probability,
- optional edge-type metadata = same-domain, representation shift, arithmetic-to-symbolic, etc.

### 9.2 Initialization

Initialize from:

- uniform prior,
- simple human priors,
- parser-derived clustering,
- early solver success statistics.

### 9.3 Updating

Update graph weights from:

- successful solver traces on official training data,
- successful synthetic generations,
- teacher rationales that pass consistency checks.

### 9.4 Puzzle-specific candidate graph

For each puzzle:

1. derive rough family candidates,
2. activate a small subset of nodes,
3. activate only compatible transitions,
4. run bounded beam search,
5. keep top candidate chains with confidence.

### 9.5 Why this matters

The graph is not the final model; it is the **teacher** that decides:

- what synthetic data to generate,
- what pseudo labels to trust,
- what mistakes to repair next,
- what curriculum order to use.

## 10. Solver search

`rule_search.py` should implement a hybrid search:

### 10.1 Stage 1: cheap routing

Use features to select 3–10 likely rule families.

### 10.2 Stage 2: exact / near-exact fit search

Try atomic ops and short compositions on training example pairs.

### 10.3 Stage 3: graph-guided beam search

Beam search over op chains using score:

```text
score = fit_score
      + lambda_start * start_prior
      + lambda_edge  * transition_logprob
      + lambda_short * length_penalty
      + lambda_cons  * answer_consistency
```

### 10.4 Stage 4: confidence estimation

Return:

- best chain,
- alternative chains,
- margin over second best,
- confidence bucket.

## 11. Synthetic data pipeline

### 11.1 Goals

Generate data that teaches the LoRA student:

- atomic rules,
- common 2-step compositions,
- rare but plausible 3–4 step compositions,
- hard negatives that expose confusions.

### 11.2 Types of synthetic data

#### Type A: atomic warmup
Large balanced set across high-confidence atomic ops.

#### Type B: graph-sampled compositions
Sample op chains from the global rule graph instead of uniform random composition.

#### Type C: confusion pairs
Generate pairs where two rule families produce superficially similar outputs.

#### Type D: format-only data
Tiny examples teaching consistent final answer emission.

### 11.3 Synthetic sample format

Each synthetic sample should contain:

- prompt text in competition-like style,
- target answer,
- optional short rationale,
- provenance metadata:
  - generating rule chain,
  - family tag,
  - difficulty,
  - confidence.

### 11.4 Filtering

Do not keep all synthetic data.

Keep only samples that pass:

- exact internal replay,
- parser round-trip,
- solver self-consistency,
- no ambiguous multiple-valid-answer detection,
- distribution sanity checks.

## 12. Pseudo-labeling and teacher distillation

### 12.1 Teacher sources

Allow two teacher sources:

- programmatic solver traces,
- optional stronger LLM-generated rationales.

### 12.2 Acceptance criteria for a pseudo label

A pseudo rationale is kept only if:

1. the final answer is correct,
2. the rationale is consistent with a valid rule chain or can be reduced to one,
3. no formatting pathology,
4. confidence exceeds threshold.

### 12.3 Distillation target

Do not distill long chain-of-thought by default.

Preferred target format:

- compact reasoning sketch,
- then a single final boxed answer.

## 13. Training plan for the LoRA student

Use staged training.

### Stage A — format alignment

Objective: near-perfect final answer formatting.

- train on small format dataset,
- enforce one final answer only,
- minimal reasoning length,
- early stop when format error rate is near zero.

### Stage B — atomic rule warmup

Objective: learn core operators.

- use high-confidence atomic synthetic data,
- add some official train examples,
- prioritize coverage over difficulty.

### Stage C — mixed real + synthetic

Objective: match real distribution.

- blend official train with filtered synthetic,
- upweight official data,
- downweight noisy pseudo labels,
- monitor family-wise generalization.

### Stage D — hard-case repair

Objective: fix systematic failure modes.

Mine errors into buckets:

- wrong family,
- right family wrong parameter,
- right chain wrong order,
- right reasoning wrong final answer,
- correct answer not boxed correctly,
- too long / truncation.

Then fine-tune on targeted repair set.

## 14. Output format guardrails

Implement `boxed_parser.py` and `normalize_answer.py` together.

### Requirements

- exactly one final answer box in generated target format,
- no nested braces inside the final answer,
- no extra boxed alternatives,
- normalization for whitespace and trivial punctuation only in local metrics,
- preserve exact submission behavior in a separate strict metric mode.

### Training-side prompt pattern

Prefer prompts like:

```text
Solve the puzzle. Think briefly and output only one final answer in the form \boxed{...}.
```

Avoid encouraging long uncontrolled reasoning.

## 15. Offline evaluation

Create 3 validation regimes.

### 15.1 IID holdout
Random split from official train.

### 15.2 Family holdout
Hold out entire rule families.

### 15.3 Composition holdout
Hold out unseen combinations of seen atomic ops.

### 15.4 Metrics
Track at least:

- strict exact match,
- numeric tolerance match if needed by local replica,
- boxed-format success rate,
- average output length,
- family-wise accuracy,
- composition depth accuracy.

### 15.5 Required reports
Every training run should emit:

- overall metrics,
- per-family metrics,
- confusion matrix over family tags,
- top failing examples,
- format failures,
- ablation delta from prior best run.

## 16. Minimal viable implementation order

### Milestone 1 — infrastructure
Implement first:

1. parser
2. answer normalizer
3. strict boxed parser
4. local metric
5. split builder
6. experiment config system

### Milestone 2 — deterministic teacher
Implement:

1. atomic op library,
2. family tagger,
3. short composition search,
4. confidence scores,
5. graph skeleton.

### Milestone 3 — synthetic pipeline
Implement:

1. synthetic generator,
2. filtering,
3. curriculum builder,
4. JSONL exporters.

### Milestone 4 — LoRA path
Implement:

1. format-only trainer,
2. main LoRA trainer,
3. offline evaluator,
4. submission packager.

### Milestone 5 — hard-case repair
Implement:

1. error bucketing,
2. repair-set builder,
3. ablation harness.

## 17. What to reuse from the existing Chain-of-Event repo

Reuse as reference only:

- the idea of a graph with node/edge weights,
- the distinction between global and instance-specific graphs,
- graph-based ranking intuition,
- optional human-editable priors.

Do **not** depend on old repo code paths in production.

## 18. Explicit engineering decisions

1. Prefer **deterministic programmatic solvers** over black-box teacher outputs when possible.
2. Keep the graph system **small and inspectable**.
3. Keep the final model path **submission-compatible from day 1**.
4. Avoid RL first; use it only after the SFT pipeline is stable.
5. Prefer **short outputs**.
6. Do not overfit to the public leaderboard; prioritize holdout stability.

## 19. First-pass ablation matrix

Run these in order:

1. format-only vs no format-only,
2. official-only SFT vs official + atomic synthetic,
3. uniform synthetic vs graph-sampled synthetic,
4. without family routing vs with family routing,
5. without hard-negative repair vs with repair,
6. rank 8 vs 16 vs 32,
7. short rationale vs no rationale.

## 20. Success criteria for Codex

The implementation is acceptable only if:

- `parse_kaggle.py` converts raw data to canonical JSONL,
- `offline_eval.py` runs end-to-end on a small split,
- `atomic_ops.py` covers the priority rule families,
- `rule_search.py` solves a nontrivial subset of train examples exactly,
- `synth_generator.py` produces validated JSONL samples,
- `train_lora.py` runs with PEFT LoRA and emits adapter files,
- `pack_submission.py` builds a competition-style `submission.zip`,
- tests pass for parser, ops, boxed parsing, and metric.

## 21. Codex execution instructions

Use this as the direct tasking prompt:

---

You are taking over a repo to build a competition-ready system for the NVIDIA Nemotron Model Reasoning Challenge.

Hard constraints:
- final artifact must be a LoRA adapter for NVIDIA Nemotron-3-Nano-30B,
- LoRA rank must be <= 32,
- final submission must be packable into submission.zip,
- no external APIs or external runtime tools at inference time,
- optimize for final boxed answer accuracy and format reliability.

Strategic decision:
- do NOT repair the original Chain-of-Event repo as the main solution,
- do reuse the paper's abstraction: global weighted graph + instance-specific graph,
- reinterpret "event" as an atomic transformation operation for reasoning puzzles,
- use the graph only as an offline teacher for rule discovery, solver search, pseudo-label filtering, synthetic data generation, curriculum learning, and hard-case mining,
- train a Nemotron LoRA student as the final competition model.

Required deliverables:
1. canonical parser for official Kaggle puzzle data,
2. answer normalizer and strict boxed parser,
3. deterministic atomic operation library,
4. short composition search engine,
5. global rule graph and puzzle-specific candidate graph,
6. synthetic data generator with filtering,
7. pseudo-label and rationale filtering pipeline,
8. staged LoRA training pipeline,
9. offline evaluator with IID/family/composition splits,
10. submission packager,
11. tests for parser, ops, boxed parser, metric, and rule search.

Create this structure:
- nemotron_reasoning/configs
- nemotron_reasoning/src/data
- nemotron_reasoning/src/rules
- nemotron_reasoning/src/synth
- nemotron_reasoning/src/labeling
- nemotron_reasoning/src/training
- nemotron_reasoning/src/eval
- nemotron_reasoning/src/utils
- nemotron_reasoning/scripts
- nemotron_reasoning/tests

Priority implementation order:
1. parser + local metric + boxed parser,
2. atomic ops + rule-family tagging + rule search,
3. synthetic data + filtering,
4. format-only LoRA training + mixed-data LoRA training,
5. error bucketing + hard-case repair,
6. submission packaging.

Engineering requirements:
- Python 3.10+,
- typed code where reasonable,
- modular design,
- clear docstrings,
- config-driven entrypoints,
- deterministic seeds,
- tests runnable with pytest,
- no dead eBay-specific assumptions.

Do not stop at scaffolding. Implement usable code with reasonable defaults, tests, and CLI entrypoints. Start with Milestone 1 and Milestone 2, then continue until there is an executable end-to-end baseline.

---

## 22. Immediate next run order

After implementation, run in this order:

1. `python -m nemotron_reasoning.src.data.parse_kaggle --config nemotron_reasoning/configs/data.yaml`
2. `python -m nemotron_reasoning.src.eval.offline_eval --sanity-check`
3. `python -m nemotron_reasoning.src.rules.rule_search --smoke-test`
4. `python -m nemotron_reasoning.src.synth.synth_generator --config nemotron_reasoning/configs/synth.yaml`
5. `python -m nemotron_reasoning.src.training.train_format_only --config nemotron_reasoning/configs/train.yaml`
6. `python -m nemotron_reasoning.src.training.train_lora --config nemotron_reasoning/configs/train.yaml`
7. `python -m nemotron_reasoning.src.eval.offline_eval --config nemotron_reasoning/configs/eval.yaml`
8. `python -m nemotron_reasoning.src.training.pack_submission --config nemotron_reasoning/configs/train.yaml`

## 23. Final note

The best use of Chain-of-Event here is as a **teacher-side graph prior**, not as the final submitted architecture. Build for submission compatibility first, then use the graph system to improve data quality and curriculum.
