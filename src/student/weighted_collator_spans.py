"""weighted_collator_spans.py
================================
Per-token weighted SFT loss collator for Round 1+.

Turns the GCD-distilled JSONL rows (from
``scripts/round1_gcd_teacher_distill.py``) into a batch whose labels
carry an extra ``weights`` tensor, so the training loop can evaluate:

    L_i = sum_t w_{i,t} * CE(logits_{i,t}, labels_{i,t+1}) * label_mask_{i,t}
          / max(sum_t w_{i,t} * label_mask_{i,t}, eps)
    L   = mean_i (sample_weight_i * L_i)

This matches the recipe that GPT-5.4 Pro Round 4 recommended:

* ``rationale`` (inside ``<think>...</think>``) gets weight 0.5
* ``\\boxed{`` wrapper and closing ``}`` get weight 1.0
* the payload inside the **last** ``\\boxed{...}`` gets weight 3.0
* EOS / stop tokens get weight 1.5 (optional; defaults to 1.0)
* prompt / system / user tokens have ``label_mask=0`` and ``weight=0``

Key numerical pitfalls handled here (from Round 4 guidance):

1. ``labels[i]`` is shifted one left of ``input_ids[i]`` in the trainer;
   ``weights`` is shifted the same way so it aligns with ``labels``.
2. ``offset_mapping`` on a **fast tokenizer** is used for span lookups;
   we always operate on the rendered chat-template string, never on
   the raw assistant content (template prefixes/suffixes shift offsets).
3. Brace-depth parser finds the last ``\\boxed{...}``, correctly
   handling nested braces (e.g. ``\\boxed{\\frac{1}{2}}``).
4. Per-example normalisation prevents long rationales from dominating
   the batch.
5. Packing is OFF — we rely on per-sample masks.

This module is CPU-only and free of torch at import time; torch is
imported lazily in :func:`collate_weighted_sft`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Default weight recipe (keep in sync with
# scripts.round1_gcd_teacher_distill.TOKEN_WEIGHT_RECIPE).
DEFAULT_TOKEN_WEIGHT_RECIPE: dict[str, float] = {
    "final_answer":  3.0,
    "boxed_wrapper": 1.0,
    "rationale":     0.5,
    "other":         1.0,
    "eos":           1.0,
}

ASSISTANT_HEADER_RE = re.compile(
    r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*",
)
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
BOXED_TOKEN = "\\boxed{"


# ---------------------------------------------------------------------------
# Span detection (pure-Python; no torch / no tokenizer needed)
# ---------------------------------------------------------------------------

@dataclass
class BoxedSpan:
    """Character offsets inside the rendered chat-template string."""

    open_start: int   # index of '\'  in '\\boxed{'
    payload_start: int   # index right after '{'
    payload_end: int     # index of the matching '}'
    close_end: int       # index right after matching '}'

    @property
    def wrapper_spans(self) -> list[tuple[int, int]]:
        """Character spans for the ``\\boxed{`` prefix and the closing ``}``."""
        return [
            (self.open_start, self.payload_start),   # '\\boxed{'
            (self.payload_end, self.close_end),      # '}'
        ]

    @property
    def payload_span(self) -> tuple[int, int]:
        return (self.payload_start, self.payload_end)


def find_last_boxed_span(text: str, search_start: int = 0) -> BoxedSpan | None:
    """Locate the **last** ``\\boxed{...}`` in ``text[search_start:]`` with
    brace-depth matching, so nested braces (e.g. ``\\frac{1}{2}``) are
    handled.  Returns ``None`` when no well-formed boxed span exists.
    """
    cursor = search_start
    last: BoxedSpan | None = None
    while True:
        idx = text.find(BOXED_TOKEN, cursor)
        if idx == -1:
            break
        payload_start = idx + len(BOXED_TOKEN)
        depth = 1
        j = payload_start
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth == 0 and j < len(text):
            last = BoxedSpan(
                open_start=idx,
                payload_start=payload_start,
                payload_end=j,
                close_end=j + 1,
            )
            cursor = j + 1
        else:
            break  # unmatched — give up
    return last


def find_assistant_span(rendered: str) -> tuple[int, int] | None:
    """Return the (start, end) character span of the last assistant turn
    in the rendered chat-template string.  ``end`` is the end of the
    string if no closing header follows.
    """
    matches = list(ASSISTANT_HEADER_RE.finditer(rendered))
    if not matches:
        return None
    last = matches[-1]
    start = last.end()
    # Look for the next turn delimiter (``<|eot_id|>`` or the next header).
    eot = rendered.find("<|eot_id|>", start)
    end = eot if eot != -1 else len(rendered)
    return (start, end)


def find_think_span(
    rendered: str, assistant_span: tuple[int, int]
) -> tuple[int, int] | None:
    """Character span of the **last** ``<think>...</think>`` block inside
    the assistant turn, if present.  Returns ``None`` for no-thinking
    outputs.
    """
    a_start, a_end = assistant_span
    sub = rendered[a_start:a_end]
    open_idx = sub.rfind(THINK_OPEN)
    if open_idx == -1:
        return None
    close_idx = sub.rfind(THINK_CLOSE)
    if close_idx == -1 or close_idx < open_idx:
        return None
    return (a_start + open_idx, a_start + close_idx + len(THINK_CLOSE))


# ---------------------------------------------------------------------------
# Weight assignment
# ---------------------------------------------------------------------------

def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def compute_token_weights(
    rendered: str,
    offset_mapping: list[tuple[int, int]],
    recipe: dict[str, float] | None = None,
) -> tuple[list[float], list[int]]:
    """Return (weights, label_mask) lists of length ``len(offset_mapping)``.

    * ``label_mask[i] == 0`` when the token is outside the assistant turn
      (prompt / system / user) — the trainer MUST replace ``labels[i]``
      with ``-100`` for those positions.
    * Weights are assigned by precedence (highest wins):
        final_answer > boxed_wrapper > rationale > other
      with an EOS override left for the caller.
    """
    rec = recipe or DEFAULT_TOKEN_WEIGHT_RECIPE
    n = len(offset_mapping)
    weights = [0.0] * n
    label_mask = [0] * n

    assistant_span = find_assistant_span(rendered)
    if assistant_span is None:
        return weights, label_mask

    think_span = find_think_span(rendered, assistant_span)
    # Boxed search only within assistant turn
    boxed = find_last_boxed_span(rendered, search_start=assistant_span[0])

    for i, (s, e) in enumerate(offset_mapping):
        if e <= s:
            continue
        tok_span = (s, e)

        if not _overlaps(tok_span, assistant_span):
            # Prompt / system / user — weight 0, masked out.
            continue
        label_mask[i] = 1

        # Default assistant weight
        w = float(rec["other"])

        # Rationale (inside <think>...</think>) outranks "other" *only*
        # downward (it's lighter, so we apply it first then let boxed
        # override).
        if think_span is not None and _overlaps(tok_span, think_span):
            w = float(rec["rationale"])

        # Boxed wrapper / payload overrides rationale + other.
        if boxed is not None:
            if _overlaps(tok_span, boxed.payload_span):
                w = max(w, float(rec["final_answer"]))
            else:
                for ws in boxed.wrapper_spans:
                    if _overlaps(tok_span, ws):
                        w = max(w, float(rec["boxed_wrapper"]))
                        break

        weights[i] = w
    return weights, label_mask


# ---------------------------------------------------------------------------
# Collator (torch-dependent; lazy import)
# ---------------------------------------------------------------------------

def collate_weighted_sft(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    recipe: dict[str, float] | None = None,
    sample_weight_key: str = "sample_weight",
) -> dict[str, Any]:
    """Collate ``examples`` (each with ``rendered`` + ``sample_weight``) into
    a batch dict suitable for a HF CausalLM trainer with a custom weighted
    loss.

    Each example MUST provide:
      * ``rendered``: str  — the FULL chat-templated text already with
        the teacher trajectory appended.
      * ``sample_weight``: float — GCD support weight.

    Output keys:
      * ``input_ids``:   [B, T]
      * ``attention_mask``: [B, T]
      * ``labels``:      [B, T]   (== input_ids with prompt positions set to -100)
      * ``token_weights``: [B, T] (float32; aligned with ``labels`` AFTER
                                   the HF trainer's internal left-shift)
      * ``sample_weights``: [B]
    """
    import torch  # local import

    B = len(examples)
    # 1) Tokenise each rendered string with offset_mapping
    enc = tokenizer(
        [ex["rendered"] for ex in examples],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_attention_mask=True,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    offsets = enc["offset_mapping"]

    batch_weights_tensor = torch.zeros((B, max_length), dtype=torch.float32)
    batch_label_mask = torch.zeros((B, max_length), dtype=torch.long)

    for i in range(B):
        w, m = compute_token_weights(examples[i]["rendered"], offsets[i], recipe)
        # Pad to max_length
        pad = max_length - len(w)
        if pad > 0:
            w = w + [0.0] * pad
            m = m + [0] * pad
        batch_weights_tensor[i] = torch.tensor(w[:max_length], dtype=torch.float32)
        batch_label_mask[i]     = torch.tensor(m[:max_length], dtype=torch.long)

    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    attn_t = torch.tensor(attn, dtype=torch.long)
    labels_t = input_ids_t.clone()
    labels_t[batch_label_mask == 0] = -100

    # --- loss-shift alignment (Round 4 pitfall #4) ---
    # HF CausalLM computes CE on logits[..., :-1, :] vs labels[..., 1:].
    # We precompute the "shifted" weights so the loss function can do:
    #     loss = (ce * shifted_weights).sum(-1) / shifted_weights.sum(-1)
    # without another shift.  We slice off weights[0] per row and pad with
    # a single 0 at the end; label_mask is shifted identically.
    shifted_weights = torch.zeros_like(batch_weights_tensor)
    shifted_weights[:, :-1] = batch_weights_tensor[:, 1:]
    # Zero out weights where the (shifted) label is -100 anyway
    shifted_labels = torch.full_like(labels_t, -100)
    shifted_labels[:, :-1] = labels_t[:, 1:]
    shifted_weights[shifted_labels == -100] = 0.0

    sample_weights_t = torch.tensor(
        [float(ex.get(sample_weight_key, 1.0)) for ex in examples],
        dtype=torch.float32,
    )

    return {
        "input_ids": input_ids_t,
        "attention_mask": attn_t,
        "labels": labels_t,
        "token_weights": shifted_weights,
        "sample_weights": sample_weights_t,
    }


def weighted_causal_lm_loss(
    logits,       # [B, T, V]
    labels,       # [B, T], -100 at masked positions
    token_weights,   # [B, T] pre-shifted
    sample_weights,  # [B]
    eps: float = 1e-8,
):
    """Per-example normalised weighted CE.  Intended to be called inside a
    subclassed ``Trainer.compute_loss``.  The shift is already baked into
    ``token_weights`` / ``labels`` by :func:`collate_weighted_sft`, so here
    we only trim the leading/trailing positions the standard HF trainer
    would trim.

    Returns a scalar tensor.
    """
    import torch  # local import
    import torch.nn.functional as F  # local import

    # Standard HF shift
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = token_weights[..., :-1].contiguous()

    B, T1, V = shift_logits.shape
    ce = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(B, T1)
    # mask out -100 positions in weights too (safety)
    active = (shift_labels != -100).to(ce.dtype)
    wt = shift_weights * active
    num = (ce * wt).sum(dim=1)
    den = wt.sum(dim=1).clamp_min(eps)
    per_ex = num / den
    return (per_ex * sample_weights.to(per_ex.dtype)).mean()


# ---------------------------------------------------------------------------
# Pure-Python self-test (no torch, no tokenizer)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    rendered = (
        "<|start_header_id|>user<|end_header_id|>Solve 1+1<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
        "<think>It's basic arithmetic.</think>"
        "The answer is \\boxed{\\frac{1}{2}}<|eot_id|>"
    )
    # Fake offset_mapping: each "char" is a token (simplest model)
    offsets = [(i, i + 1) for i in range(len(rendered))]
    w, m = compute_token_weights(rendered, offsets)

    # Assistant turn must have label_mask == 1 somewhere
    assert sum(m) > 0, "no tokens got label_mask=1"
    # Prompt tokens must be masked
    prompt_end = rendered.find("<|start_header_id|>assistant")
    assert all(mi == 0 for mi in m[:prompt_end]), "prompt tokens were not masked"

    # Boxed payload tokens must have weight == 3.0
    boxed_start = rendered.find("\\boxed{") + len("\\boxed{")
    boxed_end = rendered.rfind("}", 0, rendered.find("<|eot_id|>", boxed_start))
    payload_weights = w[boxed_start:boxed_end]
    assert all(abs(pw - 3.0) < 1e-6 for pw in payload_weights), (
        f"final-answer payload weights wrong: got {payload_weights[:5]}"
    )

    # Wrapper tokens should be weight 1.0
    assert abs(w[boxed_start - 1] - 1.0) < 1e-6, (
        f"wrapper '{{' weight wrong: {w[boxed_start - 1]}"
    )
    assert abs(w[boxed_end] - 1.0) < 1e-6, (
        f"wrapper '}}' weight wrong: {w[boxed_end]}"
    )

    # Rationale inside <think>...</think> must be 0.5
    think_open = rendered.find(THINK_OPEN)
    mid = think_open + len(THINK_OPEN) + 3
    assert abs(w[mid] - 0.5) < 1e-6, f"rationale weight wrong: {w[mid]}"

    print("[weighted_collator_spans] self-test OK "
          f"(assistant tokens={sum(m)}, payload chars={boxed_end - boxed_start})")


if __name__ == "__main__":
    _self_test()
