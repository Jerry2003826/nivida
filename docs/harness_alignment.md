# Harness-Aligned Prompt & Completion Contract

**Status**: Draft — PR0 design note
**Scope**: Prompt / completion / answer-extract contract only. Adapter packaging,
naming hygiene, and training hyper-parameters live in separate notes.
**Ground truth sources** (checked into `官方资料/` via `kaggle CLI`):

- `官方资料/kaggle_demo/nvidia-nemotron-submission-demo.ipynb`
  (`ryanholbrook/nvidia-nemotron-submission-demo`, upstream demo)
- `官方资料/kaggle_metric/nvidia-nemotron-metric.ipynb`
  (`metric/nvidia-nemotron-metric`, the **actual evaluator**)

Any behaviour documented below that conflicts with these two kernels is a bug in
this note, not in the harness.

---

## 0. Why this note exists

Before pulling the metric kernel we only had the demo notebook, which describes
how to *save* an adapter but not how it gets *invoked*. With the evaluator in
hand, three repository assumptions turn out to be wrong:

1. Training prompt is a plain string with our own `\boxed{}` guard, but the
   harness wraps the prompt through `tokenizer.apply_chat_template(...,
   enable_thinking=True)` and attaches its own guard text.
2. Training completion contains no `<think>...</think>` markers, but the
   harness's assistant seed begins inside a thinking segment (Nemotron
   thinking mode).
3. Training sequence length is governed by `max_seq_length=1024..2048`, which
   describes *prompt+completion* during training. The harness caps
   `max_model_len=4096` with `max_tokens=3584`, leaving **prompt ≤ 512 BPE
   tokens** at inference time.

All three gaps are silent — nothing in the current pipeline raises — but they
translate directly into distribution shift at submission time.

This note proposes the canonical contract and an end-state for the repository
such that training text and inference text agree on every byte that the model
actually sees.

---

## 1. Harness behaviour (ground truth)

### 1.1 Prompt construction

From `官方资料/kaggle_metric/nvidia-nemotron-metric.ipynb` cell 0,
`generate_predictions`:

```python
user_content = (
    item.prompt
    + '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
)
prompt = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': user_content}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
```

- `item.prompt` is the raw `prompt` column of `test.csv` — no pre-wrapping, no
  boxed hint.
- The guard text appended by the harness is **not** the one currently used in
  the repository (see §2).
- `enable_thinking=True` is the Nemotron-specific flag. Its exact textual
  effect is tokenizer-dependent and must be verified with a real Nemotron
  tokenizer (see §6).

### 1.2 Decoding parameters

From `generate_predictions` and `score` defaults:

| Parameter | Value |
| --- | --- |
| `max_model_len` | 4096 |
| `max_tokens` | 3584 |
| `temperature` | 1.0 |
| `top_p` | 1.0 |
| `max_num_seqs` | 128 |
| `gpu_memory_utilization` | 0.85 |
| `max_lora_rank` | 32 |
| `tensor_parallel_size` | 1 |
| `dtype` | `auto` |
| `enable_prefix_caching` | True |
| `enable_chunked_prefill` | True |
| engine | vLLM with `LoRARequest("adapter", 1, lora_path)` |

**Derived hard constraint**:
`prompt_bpe_length ≤ max_model_len − max_tokens = 512`.

### 1.3 Answer extraction

From `extract_final_answer`:

```python
# Priority 1: ALL \boxed{...} occurrences, pick the last non-empty match.
matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
# Priority 2: "The final answer is:" / "Final answer is:" / "Final answer:"
# Priority 3: last regex match of -?\d+(\.\d+)?
# Priority 4: last non-empty line of output
```

Implication: any content that appears *before* the last `\boxed{...}` is
ignored by the extractor. Trace prefixes like `family=...; sub=...; sig=...`
are safe as long as a `\boxed{}` wraps the final answer at the tail.

### 1.4 Verification

From `verify`:

- Binary strings (`^[01]+$`): strict lowercase string equality.
- Otherwise try `float()` on both sides: `math.isclose(rel_tol=1e-2,
  abs_tol=1e-5)`.
- Otherwise case-insensitive string equality.

This matches `src/competition/metrics.py` (`competition_numeric_match`,
tolerance `1e-2 / 1e-5`), so no change is needed on the metric side.

---

## 2. Current repository behaviour

### 2.1 Prompt construction

`src/competition/prompt_templates.py` exposes
`PROMPT_MODE_RAW_WITH_GUARD` / `PROMPT_MODE_GENERIC`. The raw path, used by
every training pipeline, does this:

```24:27:src/competition/prompt_templates.py
def build_raw_prompt_with_guard(example: PuzzleExample) -> str:
    if example.raw_prompt.strip():
        return _append_single_guard(example.raw_prompt)
    return build_generic_prompt(example)
```

The guard text is `ANSWER_CONTRACT` in `src/competition/official_prompts.py`:

```8:8:src/competition/official_prompts.py
ANSWER_CONTRACT = r"Return exactly one final answer as \boxed{...}."
```

Neither the raw prompt nor the guard is wrapped by `apply_chat_template`. The
string that the model sees during training is therefore `user_raw_prompt +
"\nReturn exactly one final answer as \boxed{...}."`, without any Nemotron
role markers.

### 2.2 Completion construction

`src/teacher/trace_compiler.py` emits three styles:

```8:25:src/teacher/trace_compiler.py
def render_answer_only(target_answer: str) -> str:
    return wrap_boxed(target_answer or "")


def render_short_trace(signature: ProgramSignature, target_answer: str) -> str:
    answer = render_answer_only(target_answer)
    parts = [f"family={signature.official_family}"]
    if signature.subtype:
        parts.append(f"sub={signature.subtype}")
    parts.append(f"sig={signature.signature}")
    parts.append(f"answer={answer}")
    return "; ".join(parts)


def render_token_trace(signature: ProgramSignature, target_answer: str) -> str:
    answer = render_answer_only(target_answer)
    subtype = signature.subtype or "unknown"
    return f"fam={signature.official_family}|sub={subtype}|sig={signature.signature}|{answer}"
```

None of the three includes a `<think>...</think>` segment. The completion is a
flat string that the trainer treats as the supervised target after the raw
prompt.

### 2.3 Sequence-length accounting

`src/student/lora_train.py` estimates lengths via whitespace split:

```78:79:src/student/lora_train.py
def _simple_token_count(text: str) -> int:
    return len(text.split())
```

The `dry_run_manifest` output, the `max_seq_length=auto` branch, and every
percentile in `summarise_supervised_records` all inherit this approximation.
None of them relate to true BPE length under the Nemotron tokenizer.

---

## 3. Three misalignments that affect score

### 3.1 Chat-template wrapping (high severity)

The harness runs the prompt through `apply_chat_template(...,
enable_thinking=True)`. Training data skips this, so the model sees

| Phase | String structure |
| --- | --- |
| Training input | `{raw}\nReturn exactly one final answer as \boxed{...}.` |
| Inference input | `<role_markers>{raw}\nPlease put your final answer inside \`\\boxed{}\`. ...<role_end><assistant_seed_including_think_open>` |

The role markers and assistant seed are not optional — they are the first
tokens the model conditions on. Without matching them in training, the adapter
learns to produce the answer from a distribution it will never see at scoring
time.

### 3.2 Guard text drift (low severity)

`ANSWER_CONTRACT` vs. the harness guard differs in both phrasing and placement
relative to the prompt body. This is a local token-level drift, survivable if
§3.1 is fixed, but should still be aligned to remove an unnecessary degree of
freedom.

### 3.3 Thinking segment absence (high severity)

Nemotron chat template with `enable_thinking=True` typically seeds the
assistant turn with an opening `<think>\n` token (the exact sequence is
tokenizer-dependent — see §6). At inference the model therefore generates:

```
<think>
... reasoning ...
</think>
... final answer containing \boxed{...} ...
```

If training completions never terminate a `<think>` segment, the model has no
signal for when to leave thinking and emit `\boxed{}`. In practice this leads
to either (a) the model running out of `max_tokens` inside the thinking
segment, or (b) an unstable boundary where the boxed answer never appears.

---

## 4. Desired end state

### 4.1 Prompt format (harness-equivalent)

The canonical training prompt mode becomes:

```python
HARNESS_GUARD = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

def build_chat_thinking_prompt(example: PuzzleExample, tokenizer) -> str:
    user_content = example.raw_prompt + HARNESS_GUARD
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
```

Notes:

- The function requires a live tokenizer. Tests can inject a `FakeTokenizer`
  that mimics the template structure (see §7).
- The existing `PROMPT_MODE_RAW_WITH_GUARD` stays in the code, guarded by a
  `deprecated` warning, for backward compatibility of older dry-runs.

### 4.2 Completion format (trace-as-thinking)

Recommended: **use the existing trace body as the thinking segment**. The
completion becomes

```
{trace_body}
</think>
{boxed_answer}
```

For `short_trace` specifically:

```
family=bit; sub=bit_xor_mask; sig_bucket=xor>rotate
</think>
\boxed{11001010}
```

Why this works:

- The chat template's assistant seed already supplies an open `<think>\n`
  (contingent on §6.1). The completion therefore continues *inside* thinking,
  closes it with `</think>\n`, and emits the boxed answer.
- Zero additional reasoning data is needed; the existing `family/sub/sig`
  trace doubles as a compact justification and was already the stage-2/3
  supervised target.
- `extract_final_answer` ignores everything before the last `\boxed{}`, so the
  trace prefix has no effect on scoring.

Backup option: **empty thinking**. Completion is
`</think>\n{boxed_answer}`. Shortest possible; safest if §6.1 confirms the
assistant seed is in fact inside `<think>`; drops the trace signal.

Decision will be made after §6.1 empirical check.

### 4.3 Prompt BPE length SLA

All training prompts, measured with the real Nemotron tokenizer *after*
`apply_chat_template`, must satisfy:

- `len(tokenizer(prompt)["input_ids"]) ≤ 512`
- Samples above the threshold are **filtered**, not truncated (truncation
  destroys role markers and is indistinguishable at inference from a prompt
  injection).
- Current `max_seq_length=auto_floor_seq_length=1024` still governs
  prompt+completion total length for gradient checkpointing and padding
  decisions; it is *not* the submission constraint.

---

## 5. Implementation plan

### 5.1 New files

| Path | Purpose |
| --- | --- |
| `src/competition/harness_prompt.py` | Pure functions: `HARNESS_GUARD`, `wrap_as_thinking(body, answer)`, `build_chat_thinking_prompt(example, tokenizer)`. No tokenizer import at module top. |
| `scripts/audit_prompt_lengths.py` | Load a real tokenizer, run it over `stage{1,2,3}_*_train.jsonl`, write `artifacts/prompt_length_audit.json` with p50/p95/p100 BPE lengths and a `samples_over_512` field. |
| `tests/test_harness_prompt.py` | FakeTokenizer-based tests for `wrap_as_thinking` and the chat-template wrapping shape. |

### 5.2 Modified files

| Path | Change |
| --- | --- |
| `src/competition/prompt_templates.py` | Add `PROMPT_MODE_CHAT_THINKING = "chat_thinking"`; keep `PROMPT_MODE_RAW_WITH_GUARD` but mark as legacy. |
| `src/competition/official_prompts.py` | Add `HARNESS_ANSWER_CONTRACT` constant; keep `ANSWER_CONTRACT` for the legacy path. |
| `src/teacher/trace_compiler.py` | Add `wrap_as_thinking(body, answer)`; rename `render_short_trace` / `render_token_trace` to emit *body only* (no `\boxed{}`); let the caller attach `</think>\n{boxed}`. |
| `src/student/sft_dataset_builder.py` | Default `--prompt-mode` to `chat_thinking` for profiles `stage1/stage2/stage3`; wire tokenizer from config. |
| `src/student/lora_train.py` | `summarise_supervised_records` gains an optional `tokenizer` argument; `_simple_token_count` stays as a fallback only. |
| `src/student/inference.py` | Local inference path applies the same chat-template wrapping; this keeps the local smoke path distribution-aligned with vLLM. |
| `scripts/train_stage{1,2,3}_*.sh` | Pass `--prompt-mode chat_thinking`; ensure dataset builder receives the tokenizer through config. |

### 5.3 Unchanged (explicitly)

- `src/student/format_guard.py::wrap_boxed` — still the canonical boxed
  wrapper.
- `src/competition/metrics.py` — tolerance already matches `verify`.
- `src/competition/answer_extract.py` — already semantically equivalent to
  `extract_final_answer` (priority order verified against §1.3).
- `src/student/package_submission.py` — packaging contract is independent of
  prompt/completion format.

---

## 6. Questions that require a real Nemotron tokenizer (H100 day-1)

The three experiments below are **blocking** for freezing §4.2 and the trace
body shape. Each is one line and runs on any machine with the model handle
accessible — no training needed.

### 6.1 Assistant seed inspection

Run via `scripts/probe_chat_template.py`. The script's default mode uses the
per-file kagglehub API (`model_download(handle, path=<filename>)`) to pull
**only tokenizer files**, keeping disk usage under ~200 MB rather than the
~60 GB the bundle as a whole would require. Use `--download-full-model` if
the same machine will run the weights afterwards.

Core logic the probe executes:

```python
from transformers import AutoTokenizer
# tokenizer-only download path — see scripts/probe_chat_template.py
tok = AutoTokenizer.from_pretrained(local_tokenizer_dir, trust_remote_code=True)
print(repr(tok.apply_chat_template(
    [{"role": "user", "content": "HELLO"}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)))
```

Outputs of interest:

- Does the string end in `<think>\n`, `<think>` (no newline), or neither?
- Are there role markers like `<|im_start|>assistant` surrounding the seed?
- Is the user block terminated explicitly, e.g. `<|im_end|>\n`?

This determines whether §4.2's recommended completion (`body\n</think>\n{box}`)
is valid or needs an explicit opening `<think>\n`.

### 6.2 `enable_thinking` support check

Some tokenizer builds silently ignore `enable_thinking`. Inspect
`tok.chat_template` for a Jinja block referencing `enable_thinking`. If
absent, repeat 6.1 with `enable_thinking=False` and diff.

### 6.3 Real length on a public sample

Run the chat-template wrapping on the first row of
`官方资料/test.csv` (public) and report `len(tok(prompt)["input_ids"])`. A
headroom test: if this number exceeds ~300, the 512-token budget is tight and
any multi-example prompt (which training data frequently has) risks blowing
the budget.

---

## 7. Acceptance criteria for PR0

1. `PROMPT_MODE_CHAT_THINKING` is the default across
   `scripts/train_stage{1,2,3}*.sh` and `src/student/sft_dataset_builder.py`;
   `PROMPT_MODE_RAW_WITH_GUARD` remains importable but emits a
   `DeprecationWarning` when selected.
2. `wrap_as_thinking` produces a completion containing exactly one `</think>`
   and exactly one `\boxed{...}` closing the answer. Covered by
   `tests/test_harness_prompt.py` against a FakeTokenizer whose chat template
   mirrors the real one (placeholder until §6.1 validates).
3. `scripts/audit_prompt_lengths.py` produces `artifacts/prompt_length_audit.json`
   with `p100 ≤ 512` for every active training dataset; any over-budget samples
   are listed for filtering.
4. `pytest -q` passes with new tests. In particular:
   - `test_sft_dataset_builder_chat_thinking_default`
   - `test_harness_prompt_roundtrip`
   - `test_wrap_as_thinking_single_boxed`
5. Legacy `raw_with_guard` still builds its dataset (no hard break for
   reproducibility of prior dry-runs), but is not wired to any `train_*.sh`
   entry point.

---

## 8. Interaction with other in-flight work

| PR | Relationship to PR0 |
| --- | --- |
| PR1 — default entry / README / smoke convergence | Depends on PR0. Smoke script must emit the new default prompt mode. Merge PR0 first. |
| PR2 — artifact naming + schema guard | Independent. Can merge in parallel. |
| PR-adapter — stage2→stage3 continue training | Depends on PR0. The stage3 resume dataset must share the chat-thinking contract. Schedule after PR0 merges. |
| PR-H100 — real environment verification | Depends on PR0 for `audit_prompt_lengths.py`; §6 experiments feed back into PR0 if the assistant seed or `enable_thinking` behaviour differs from the assumption. |

---

## Appendix A — Harness extraction priority, for reference

```
1. All regex matches of  \\boxed\{([^}]*)(?:\}|$)  → last non-empty
2. regex  The final answer is:\s*([^\n]+)
3. regex  Final answer is:\s*([^\n]+)
4. regex  Final answer\s*[:：]\s*([^\n]+)
5. regex  final answer\s*[:：]\s*([^\n]+)
6. last regex match of  -?\d+(?:\.\d+)?
7. last non-empty stripped line
8. "NOT_FOUND"
```

Only (1) and (6) are reliable under a LoRA adapter's free-form output; training
must guarantee (1) fires.

## Appendix B — Single-source-of-truth file inventory

These files, once §6 completes, become the locked contract:

- `src/competition/harness_prompt.py` — all string constants, no imports other
  than `typing`.
- `docs/harness_alignment.md` (this file) — design authority.
- `tests/test_harness_prompt.py` — byte-level contract tests.

Any future change to the harness guard text, the `enable_thinking` flag, or
`max_model_len / max_tokens` must update all three in the same PR.
