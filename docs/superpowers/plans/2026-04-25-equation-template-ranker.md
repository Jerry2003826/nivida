# Equation Template Ranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local `equation_template` diagnostic and ranker/verifier loop that improves support-perfect template extrapolation and prevents high-risk template traces from entering strict training data.

**Architecture:** Add a standalone diagnostic script first, then expose deterministic template candidate features from `OperatorTemplateOp`, then pass those features through chain-search and annotation metadata into stage2 selection. Keep the headline audit in `scripts/audit_solver_coverage.py` as the final regression check.

**Tech Stack:** Python 3.14, existing `src.competition.schema`, `src.teacher.chain_search`, `src.teacher.atomic_ops`, pytest, JSONL/CSV/Markdown reports.

---

## File Structure

- Create `scripts/diagnose_equation_template.py`: local-only diagnostic CLI for `equation_template` oracle-at-K, risk classes, and feature summaries.
- Modify `src/teacher/atomic_ops.py`: add template feature extraction helpers and attach feature summaries to `operator_template` params.
- Modify `src/teacher/chain_search.py`: preserve candidate debug metadata emitted by atomic ops.
- Modify `src/teacher/program_signature.py`: copy template diagnostic metadata into example extras during annotation.
- Modify `src/student/sft_dataset_builder.py`: reject high-risk template traces from strict stage2 while allowing answer-only silver.
- Add tests in `tests/test_equation_template_diagnostic.py`, `tests/test_atomic_ops.py`, `tests/test_chain_search.py`, and `tests/test_stage2_silver_selection.py`.
- Update `docs/solver_coverage_audit_latest.md`, `docs/nemotron_087_strategy.md`, and `docs/nemotron_lora_next_round.md` after verification.

---

### Task 1: Add Equation Template Diagnostic CLI

**Files:**
- Create: `scripts/diagnose_equation_template.py`
- Create: `tests/test_equation_template_diagnostic.py`

- [ ] **Step 1: Write failing diagnostic tests**

Create `tests/test_equation_template_diagnostic.py` with fixtures that cover top-1 hit, oracle-at-K hit, and oracle miss:

```python
from __future__ import annotations

from scripts.diagnose_equation_template import (
    classify_template_risk,
    target_char_provenance,
)


def test_target_char_provenance_counts_sources() -> None:
    row_text = {
        "support_inputs": ["ab*c", "de*f"],
        "support_outputs": ["aX", "dY"],
        "query": "gh*i",
        "target": "gYZ",
    }

    provenance = target_char_provenance(row_text)

    assert provenance == {
        "from_query": 1,
        "from_support_inputs": 0,
        "from_support_outputs": 1,
        "unseen": 1,
    }


def test_classify_template_risk_top1_hit() -> None:
    risk = classify_template_risk(
        oracle_rank=1,
        ambiguity_count=3,
        has_unseen_literal=False,
        support_full=True,
    )

    assert risk == "low_risk_support_stable"


def test_classify_template_risk_oracle_at_k_hit() -> None:
    risk = classify_template_risk(
        oracle_rank=2,
        ambiguity_count=5,
        has_unseen_literal=False,
        support_full=True,
    )

    assert risk == "ranker_miss_oracle_hit"


def test_classify_template_risk_unseen_literal() -> None:
    risk = classify_template_risk(
        oracle_rank=None,
        ambiguity_count=4,
        has_unseen_literal=True,
        support_full=True,
    )

    assert risk == "unseen_literal_high_risk"
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
python -m pytest tests/test_equation_template_diagnostic.py -v
```

Expected: import failure because `scripts/diagnose_equation_template.py` does not exist.

- [ ] **Step 3: Implement diagnostic helpers and CLI**

Create `scripts/diagnose_equation_template.py` with:

```python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition.metrics import competition_correct  # noqa: E402
from src.competition.schema import PuzzleExample  # noqa: E402
from src.teacher.chain_search import ChainSearchEngine  # noqa: E402
from src.teacher.family_tagger import apply_family_tags  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def target_char_provenance(row_text: dict[str, Any]) -> dict[str, int]:
    query = str(row_text.get("query", ""))
    support_inputs = "".join(str(value) for value in row_text.get("support_inputs", []))
    support_outputs = "".join(str(value) for value in row_text.get("support_outputs", []))
    target = str(row_text.get("target", ""))
    counts = {
        "from_query": 0,
        "from_support_inputs": 0,
        "from_support_outputs": 0,
        "unseen": 0,
    }
    for char in target:
        if char in query:
            counts["from_query"] += 1
        elif char in support_inputs:
            counts["from_support_inputs"] += 1
        elif char in support_outputs:
            counts["from_support_outputs"] += 1
        else:
            counts["unseen"] += 1
    return counts


def classify_template_risk(
    *,
    oracle_rank: int | None,
    ambiguity_count: int,
    has_unseen_literal: bool,
    support_full: bool,
) -> str:
    if not support_full:
        return "operator_gap_oracle_miss"
    if oracle_rank == 1 and ambiguity_count <= 3 and not has_unseen_literal:
        return "low_risk_support_stable"
    if oracle_rank is not None and oracle_rank > 1:
        return "ranker_miss_oracle_hit"
    if has_unseen_literal:
        return "unseen_literal_high_risk"
    return "operator_gap_oracle_miss"


def _support_full(candidate: Any, example: PuzzleExample) -> bool:
    if candidate is None or len(candidate.predictions) != len(example.parsed_examples):
        return False
    return all(
        competition_correct(prediction, pair.output)
        for prediction, pair in zip(candidate.predictions, example.parsed_examples)
    )


def diagnose_example(engine: ChainSearchEngine, row: dict[str, Any], *, top_k: int) -> dict[str, Any] | None:
    example = PuzzleExample.from_dict(row)
    apply_family_tags([example])
    if example.metadata.official_family != "equation" or example.metadata.subtype != "equation_template":
        return None
    candidates = engine.solve_example(example, top_k=top_k)
    target = "" if example.target_answer is None else str(example.target_answer)
    oracle_rank = None
    candidate_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        prediction = "" if candidate.query_prediction is None else str(candidate.query_prediction)
        query_correct = bool(target) and competition_correct(prediction, target)
        if query_correct and oracle_rank is None:
            oracle_rank = index
        candidate_rows.append(
            {
                "rank": index,
                "prediction": prediction,
                "query_correct": query_correct,
                "score": float(candidate.score),
                "exact_ratio": float(candidate.exact_ratio),
                "steps": ">".join(step.op_name for step in candidate.steps),
                "debug": candidate.debug,
            }
        )
    provenance = target_char_provenance(
        {
            "support_inputs": [pair.input for pair in example.parsed_examples],
            "support_outputs": [pair.output for pair in example.parsed_examples],
            "query": example.query,
            "target": target,
        }
    )
    support_full = bool(candidates) and _support_full(candidates[0], example)
    risk = classify_template_risk(
        oracle_rank=oracle_rank,
        ambiguity_count=len([candidate for candidate in candidates if _support_full(candidate, example)]),
        has_unseen_literal=provenance["unseen"] > 0,
        support_full=support_full,
    )
    return {
        "id": example.id,
        "family": example.metadata.official_family,
        "subtype": example.metadata.subtype,
        "target": target,
        "query": example.query,
        "top_prediction": "" if not candidates or candidates[0].query_prediction is None else str(candidates[0].query_prediction),
        "top_query_correct": oracle_rank == 1,
        "oracle_rank": oracle_rank,
        "ambiguity_count": len([candidate for candidate in candidates if _support_full(candidate, example)]),
        "risk_class": risk,
        **{f"provenance_{key}": value for key, value in provenance.items()},
        "candidates": candidate_rows,
    }
```

Then add this CLI code below `diagnose_example()`:

```python
def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat_rows = [{key: value for key, value in row.items() if key != "candidates"} for row in rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Equation Template Diagnostic",
        "",
        "| risk_class | n | top1_acc | oracle_at_k | unseen_literal_rows |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    risk_classes = sorted({str(row["risk_class"]) for row in rows})
    for risk_class in risk_classes:
        group = [row for row in rows if row["risk_class"] == risk_class]
        top1 = _rate([bool(row["top_query_correct"]) for row in group])
        oracle = _rate([row["oracle_rank"] is not None for row in group])
        unseen = sum(int(row["provenance_unseen"]) > 0 for row in group)
        lines.append(f"| {risk_class} | {len(group)} | {top1:.4f} | {oracle:.4f} | {unseen} |")
    lines.extend(["", "## Top Misses", ""])
    misses = [row for row in rows if not row["top_query_correct"]][:20]
    for row in misses:
        lines.append(
            f"- `{row['id']}` risk=`{row['risk_class']}` oracle_rank=`{row['oracle_rank']}` "
            f"target=`{row['target']}` top=`{row['top_prediction']}`"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_diagnostic(inputs: list[Path], *, top_k: int) -> list[dict[str, Any]]:
    engine = ChainSearchEngine(beam_width=24, max_depth=4)
    diagnostics: list[dict[str, Any]] = []
    for path in inputs:
        for row in _load_jsonl(path):
            diagnostic = diagnose_example(engine, row, top_k=top_k)
            if diagnostic is None:
                continue
            diagnostic["path"] = str(path)
            diagnostics.append(diagnostic)
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose equation_template ambiguity and oracle-at-K.")
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=Path("data/processed/equation_template_diagnostic.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/equation_template_diagnostic.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/equation_template_diagnostic_latest.md"))
    args = parser.parse_args()

    inputs = args.input or [
        Path("data/processed/local_eval_manifests/combined_balanced_48pf.jsonl"),
        Path("data/processed/local_eval_manifests/proxy_all_balanced_64pf.jsonl"),
        Path("data/processed/local_eval_manifests/hard_triad_full.jsonl"),
    ]
    rows = run_diagnostic(inputs, top_k=args.top_k)
    _write_json(args.output_json, {"settings": {"top_k": args.top_k}, "rows": rows})
    _write_csv(args.output_csv, rows)
    _write_markdown(args.output_md, rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "output_md": str(args.output_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run diagnostic tests**

Run:

```bash
python -m pytest tests/test_equation_template_diagnostic.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run diagnostic on local manifests**

Run:

```bash
python scripts/diagnose_equation_template.py
```

Expected: report files are written under `data/processed/equation_template_diagnostic.*` and `docs/equation_template_diagnostic_latest.md`.

- [ ] **Step 6: Commit Task 1**

```bash
git add scripts/diagnose_equation_template.py tests/test_equation_template_diagnostic.py data/processed/equation_template_diagnostic.json data/processed/equation_template_diagnostic.csv docs/equation_template_diagnostic_latest.md
git commit -m "Add equation template diagnostic report"
```

---

### Task 2: Expose Template Feature Metadata

**Files:**
- Modify: `src/teacher/atomic_ops.py`
- Modify: `src/teacher/chain_search.py`
- Test: `tests/test_atomic_ops.py`
- Test: `tests/test_chain_search.py`

- [ ] **Step 1: Add failing atomic-op feature test**

Append to `tests/test_atomic_ops.py`:

```python
def test_operator_template_params_include_rank_features() -> None:
    op = OperatorTemplateOp()
    params = op.candidate_params(
        [
            ("#/-\\@", "-@#"),
            ("\"\"+#)", ")/"),
            ("'#+/#", "%\""),
            ("\\)-)@", "-'\""),
        ]
    )

    assert params
    features = params[0]["template_rank_features"]
    assert set(features) >= {
        "literal_count",
        "repeated_positions",
        "backward_edges",
        "skipped_unique_positions",
        "template_count",
    }
```

- [ ] **Step 2: Run failing test**

Run:

```bash
python -m pytest tests/test_atomic_ops.py::test_operator_template_params_include_rank_features -v
```

Expected: fails with missing `template_rank_features`.

- [ ] **Step 3: Implement feature extraction**

In `src/teacher/atomic_ops.py`, add a helper near `_template_generalisation_key`:

```python
def _template_rank_features(templates: dict[str, list[tuple[str, int | str]]]) -> dict[str, int]:
    literal_count = 0
    repeated_positions = 0
    backward_edges = 0
    skipped_unique_positions = 0
    for template in templates.values():
        positions = [int(value) for kind, value in template if kind == "pos"]
        literal_count += sum(1 for kind, _ in template if kind == "lit")
        repeated_positions += len(positions) - len(set(positions))
        backward_edges += sum(1 for left, right in zip(positions, positions[1:]) if right < left)
        if positions:
            skipped_unique_positions += max(positions) - min(positions) + 1 - len(set(positions))
    return {
        "literal_count": literal_count,
        "repeated_positions": repeated_positions,
        "backward_edges": backward_edges,
        "skipped_unique_positions": skipped_unique_positions,
        "template_count": len(templates),
    }
```

When `OperatorTemplateOp.candidate_params()` appends a candidate, include:

```python
"template_rank_features": _template_rank_features(templates)
```

- [ ] **Step 4: Preserve metadata in chain debug**

In `src/teacher/chain_search.py`, when constructing `ChainStep`, no structural change is needed because `params` are already stored. Add `template_rank_features` into the candidate `debug` object for the final state if any step has it:

```python
"template_rank_features": [
    step.params.get("template_rank_features")
    for step in state.steps
    if isinstance(step.params, dict) and step.params.get("template_rank_features") is not None
],
```

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest tests/test_atomic_ops.py tests/test_chain_search.py -v
```

Expected: pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/teacher/atomic_ops.py src/teacher/chain_search.py tests/test_atomic_ops.py tests/test_chain_search.py
git commit -m "Expose equation template rank features"
```

---

### Task 3: Add Template Risk Metadata To Annotation And Gates

**Files:**
- Modify: `src/teacher/program_signature.py`
- Modify: `src/student/sft_dataset_builder.py`
- Test: `tests/test_stage2_silver_selection.py`

- [ ] **Step 1: Add failing gate test**

Append to `tests/test_stage2_silver_selection.py`:

```python
def test_strict_gate_rejects_high_risk_template_trace() -> None:
    example = _example("template_risk", family="equation", target="X", query="a*b")
    example.metadata.subtype = "equation_template"
    example.metadata.extras["template_risk_class"] = "unseen_literal_high_risk"

    ok, reason = _select_official_stage2_strict(example)

    assert ok is False
    assert reason == "high_risk_template_trace"
```

- [ ] **Step 2: Run failing test**

Run:

```bash
python -m pytest tests/test_stage2_silver_selection.py::test_strict_gate_rejects_high_risk_template_trace -v
```

Expected: fails because the new rejection reason is not implemented.

- [ ] **Step 3: Add rejection reason and strict gate**

In `src/student/sft_dataset_builder.py`, add `"high_risk_template_trace"` to `STAGE2_REJECTION_REASONS`.

In `_select_official_stage2_strict()`, after `query_prediction_mismatch` and before `missing_program_signature`, add:

```python
    if (
        example.metadata.subtype == "equation_template"
        and example.metadata.extras.get("template_risk_class")
        in {"ranker_miss_oracle_hit", "operator_gap_oracle_miss", "unseen_literal_high_risk"}
    ):
        return False, "high_risk_template_trace"
```

- [ ] **Step 4: Add metadata propagation**

In `src/teacher/program_signature.py`, inside `annotate_example_from_candidates()`, after `query_solver_correct`, compute:

```python
    template_rank_features = []
    if top is not None:
        template_rank_features = top.debug.get("template_rank_features", [])
```

Then include in `example.metadata.extras`:

```python
        "template_ranker_features": template_rank_features,
```

Do not compute `template_risk_class` here unless the diagnostic script supplies labels. Keep risk-class generation in the diagnostic/report path for now.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest tests/test_stage2_silver_selection.py tests/test_stage2_subtype_rescue.py -v
```

Expected: pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/teacher/program_signature.py src/student/sft_dataset_builder.py tests/test_stage2_silver_selection.py
git commit -m "Gate high-risk equation template traces"
```

---

### Task 4: Run Full Local Verification And Update Docs

**Files:**
- Modify: `docs/solver_coverage_audit_latest.md`
- Modify: `docs/nemotron_087_strategy.md`
- Modify: `docs/nemotron_lora_next_round.md`

- [ ] **Step 1: Run targeted tests**

Run:

```bash
python -m pytest tests/test_equation_template_diagnostic.py tests/test_atomic_ops.py tests/test_chain_search.py tests/test_family_tagger.py tests/test_stage2_silver_selection.py tests/test_stage2_subtype_rescue.py
```

Expected: all pass.

- [ ] **Step 2: Compile changed Python files**

Run:

```bash
python -m py_compile scripts/diagnose_equation_template.py scripts/audit_solver_coverage.py src/teacher/atomic_ops.py src/teacher/chain_search.py src/teacher/program_signature.py src/student/sft_dataset_builder.py
```

Expected: exit code 0.

- [ ] **Step 3: Run diagnostics**

Run:

```bash
python scripts/diagnose_equation_template.py
python scripts/audit_solver_coverage.py
```

Expected: both scripts complete and write reports.

- [ ] **Step 4: Update docs**

Update `docs/nemotron_087_strategy.md` and `docs/nemotron_lora_next_round.md` with:

- latest overall audit scores;
- `equation_template` accuracy;
- diagnostic oracle-at-K summary;
- new gate behavior for high-risk template traces.

- [ ] **Step 5: Check diff**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files changed.

- [ ] **Step 6: Commit and push**

```bash
git add docs data scripts src tests
git commit -m "Add equation template diagnostic and trace gates"
git push origin codex/blackwell-managed-training
```

Expected: push succeeds.

---

## Self-Review

Spec coverage:

- Diagnostic report is covered by Task 1.
- Template feature/ranker metadata is covered by Task 2.
- Trace-gating metadata and strict rejection are covered by Task 3.
- Audit/docs verification is covered by Task 4.

Placeholder scan:

- No TBD/TODO/FIXME placeholders are intentionally left in this plan.

Scope check:

- This plan intentionally implements diagnostic metadata and trace gating first.
  It does not yet implement a broad `OperatorTemplateOp` search expansion. That
  expansion should wait until oracle-at-K from Task 1 shows the current
  candidate set is insufficient.
