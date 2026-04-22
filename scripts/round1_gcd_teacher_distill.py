"""round1_gcd_teacher_distill.py
================================
Greedy Consensus Distillation (GCD): generate N teacher trajectories per
prompt, cluster by answer equivalence, and retain 1–2 verified concise
trajectories from the majority cluster as student SFT targets, with
support-weighted per-sample weights.

Round 3 updates (GPT-5.4 Pro, 2026-04-22)
-----------------------------------------
The original GCD design sampled a fixed N=16 and kept only the single
shortest verified trajectory per prompt.  Evidence-grounded Round 3
revision changes three things:

  1. **Adaptive N**  — N=4 → 8 → 16 tiered.  We first sample N=4; if a
     strict majority is already established (>50% agreement) AND the
     shortest verified trajectory is ≤2048 chars, we stop.  Otherwise we
     bump to N=8, then N=16.  This saves ~40–60% of teacher tokens on
     easy prompts.
  2. **Multi-trace retention** — keep 1 or 2 verified concise trajectories
     per prompt (not only the shortest).  The second retained trace
     must be the SECOND-shortest verified trajectory AND within 1.35×
     the length of the shortest.  This diversifies reasoning styles
     without pulling in noisy long chains.
  3. **Support-weighted sample weight** — each SFT row carries
     ``sample_weight = 1.0 + 2.0 * support / N``, where ``support`` is
     the size of the majority cluster (how many of N samples agreed).
     High-support prompts (unanimous or near-unanimous) get up to 3x
     the gradient contribution of noisy ones.  Token-level weights are
     3.0 on the final answer literal inside \\boxed{} and 0.5 on the
     rationale tokens inside <think>...</think>.

Theory
------
Standard best-of-N distillation selects any correct trajectory, which biases
the student toward longer, noisier reasoning chains.  GCD tightens this:

  1. Sample N completions from the teacher at temperature=0.7 top_p=0.95
     (Round 4 defaults — diverse but not degenerate) — adaptive
     N=4 → 8 → 16 per Round 3.  Round 4 also optionally gates the full
     run behind a 300-prompt majority-vs-greedy uplift check.
  2. Cluster the completions by answer equivalence using the competition's
     ``verify()`` function.  Each cluster's "canonical answer" is the
     extracted answer of its first member.
  3. Identify the majority cluster (most samples, tie broken by
     alphabetically-first canonical answer for reproducibility).
  4. Within the majority cluster, keep only trajectories whose extracted
     answer ``verify()``-matches the ground-truth answer (or, if no ground
     truth is available, the majority-cluster canonical answer).
  5. From those verified trajectories, select 1–2 concise trajectories
     as SFT targets.  Shorter chains are preferred because they reduce the
     student's tendency to over-generate during greedy inference (see PCPO,
     ACL Findings 2025).

This approach is grounded in:
  * ScPO — Self-Consistency Preference Optimisation (arXiv 2411.04109):
    uses majority-vote consistency as a proxy reward signal to build
    preference pairs for online DPO/PPO.  GCD adapts the clustering step
    from ScPO but selects a single SFT target instead of building pairs.
  * PCPO — Process-level Consistency Preference Optimisation (ACL Findings
    2025): establishes that trajectory length is a strong proxy for
    generalisation quality when accuracy is controlled.

--dry-run mode
--------------
When --dry-run is passed, the vLLM engine is NEVER instantiated.  Three
synthetic prompts are processed through mock teacher output so that the full
clustering + selection + serialisation path can be validated on a CPU-only
machine.

Teacher vLLM integration points
---------------------------------
The script reuses helpers from ``scripts.eval_official_vllm_proxy``:
  * ``_instantiate_vllm`` — wraps ``vllm.LLM(...)``
  * ``_build_sampling_params`` — wraps ``vllm.SamplingParams(...)``
  * ``_build_lora_request`` — wraps ``vllm.lora.request.LoRARequest(...)``

These are imported lazily (inside functions) so the module is importable
without vLLM installed, enabling dry-run and unit-test usage.

Output schema (per row)
-----------------------
::

    {
      "id":                     str,
      "prompt":                 str,   # raw problem text (not chat-templated)
      "completion":             str,   # selected trajectory: <think>...</think>\\boxed{ans}
      "teacher_num_samples":    int,   # N actually requested from teacher (adaptive)
      "teacher_majority_size":  int,   # size of the majority answer cluster
      "teacher_selected_length": int,  # char count of selected trajectory
      "teacher_answer":         str,   # extracted answer from selected trajectory
      "sample_weight":          float, # 1.0 + 2.0 * support / N  (Round 3)
      "retention_rank":         int,   # 1 = primary shortest, 2 = secondary (multi-trace)
      "token_weights":          dict,  # {"final_answer": 3.0, "rationale": 0.5, ...}
    }

Report schema
-------------
Aggregate statistics over the full run written as JSON next to the output
JSONL.

Usage
-----
    # Dry run (no GPU required)
    python scripts/round1_gcd_teacher_distill.py --dry-run \\
        --output data/processed/round1_gcd_distill.jsonl

    # Full run
    python scripts/round1_gcd_teacher_distill.py \\
        --input  data/processed/round1_nmath_v2.jsonl \\
        --output data/processed/round1_gcd_distill.jsonl \\
        --config configs/train_lora.yaml \\
        --num-samples 16 \\
        --teacher-temperature 0.7 --top-p 0.95 \\
        --teacher-majority-gate --majority-gate-subset-size 300 \\
        --workdir data/workdir/round1_gcd
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Module-level imports must be lightweight (no vllm / torch / safetensors).
from src.competition.official_metric_contract import (  # noqa: E402
    extract_final_answer,
    verify,
)
from src.common.io import load_jsonl, write_json, write_jsonl  # noqa: E402


# ---------------------------------------------------------------------------
# Dry-run fixtures
# ---------------------------------------------------------------------------

_DRY_RUN_PROMPTS: list[dict[str, Any]] = [
    {
        "id": "dry_000",
        "prompt": "What is 17 * 23?",
        "target_answer": "391",
        "solution_trace": "",
        "family": "arithmetic",
        "source_dataset": "nemotron_math_v2",
    },
    {
        "id": "dry_001",
        "prompt": "What is the square root of 144?",
        "target_answer": "12",
        "solution_trace": "",
        "family": "arithmetic",
        "source_dataset": "nemotron_math_v2",
    },
    {
        "id": "dry_002",
        "prompt": "If 2x + 5 = 19, what is x?",
        "target_answer": "7",
        "solution_trace": "",
        "family": "algebra",
        "source_dataset": "nemotron_math_v2",
    },
]


def _make_mock_completions(prompt_row: dict[str, Any], n: int) -> list[str]:
    """Generate synthetic teacher completions for dry-run validation."""
    answer = str(prompt_row.get("target_answer", "42"))
    completions: list[str] = []
    for i in range(n):
        # Vary length slightly across samples to exercise shortest-selection.
        padding = "Let me verify: " * (i % 3)
        completions.append(
            f"<think>\n{padding}The answer is {answer}.\n</think>\n\\boxed{{{answer}}}"
        )
    return completions


# ---------------------------------------------------------------------------
# Core GCD functions
# ---------------------------------------------------------------------------

def _cluster_answers(answers: list[str]) -> dict[str, list[int]]:
    """Group sample indices by answer equivalence using verify().

    Two answers are considered equivalent when verify(a, b) returns True.
    Equivalence classes are built greedily: the first unassigned answer
    becomes the canonical representative of a new class; subsequent answers
    are merged into the first class whose canonical answer they match.

    Parameters
    ----------
    answers:
        Extracted answers (output of extract_final_answer) for each sample,
        in order.

    Returns
    -------
    dict mapping canonical_answer -> list of sample indices in that cluster.
    """
    clusters: dict[str, list[int]] = {}  # canonical -> indices
    for idx, ans in enumerate(answers):
        placed = False
        for canonical in clusters:
            if verify(canonical, ans):
                clusters[canonical].append(idx)
                placed = True
                break
        if not placed:
            clusters[ans] = [idx]
    return clusters


def _pick_shortest_verified(
    trajectories: list[str],
    indices: list[int],
    target_answer: str | None,
) -> int | None:
    """Return the index (into *trajectories*) of the shortest verified sample.

    Kept as a thin wrapper over _pick_topk_verified for backwards
    compatibility with existing call sites.
    """
    picks = _pick_topk_verified(trajectories, indices, target_answer, k=1)
    return picks[0] if picks else None


def _pick_topk_verified(
    trajectories: list[str],
    indices: list[int],
    target_answer: str | None,
    k: int = 2,
    secondary_length_ratio: float = 1.35,
) -> list[int]:
    """Return up to *k* shortest verified sample indices from the majority
    cluster.  Round 3 multi-trace retention.

    The primary pick (rank 1) is the shortest verified trajectory.  Any
    additional picks (rank 2..k) must satisfy
    ``len(traj) <= secondary_length_ratio * len(primary)`` — otherwise we
    stop early.  This prevents the student from being trained on a
    noticeably longer second chain when only one clean chain exists.

    Parameters
    ----------
    trajectories:
        Full list of trajectory strings (all N samples for a single prompt).
    indices:
        Indices into *trajectories* to consider (the majority cluster).
    target_answer:
        Ground-truth answer string.  If None or empty, all cluster members
        are treated as verified (caller passes the majority canonical
        answer in that case).
    k:
        Max number of trajectories to retain (1 or 2 in Round 3 practice).
    secondary_length_ratio:
        Upper bound on (secondary_length / primary_length) for any pick
        beyond the first.  1.35 matches the Round 3 recommendation.

    Returns
    -------
    List of indices (into *trajectories*) in retention order (primary
    first).  Empty list when no verified trajectory exists in the cluster.
    """
    if not indices or k <= 0:
        return []
    verified: list[tuple[int, int]] = []  # (index, char_len)
    for idx in indices:
        traj = trajectories[idx]
        extracted = extract_final_answer(traj)
        if target_answer and verify(str(target_answer), extracted):
            verified.append((idx, len(traj)))
        elif not target_answer:
            verified.append((idx, len(traj)))
    if not verified:
        return []
    verified.sort(key=lambda pair: pair[1])
    picks = [verified[0][0]]
    primary_len = verified[0][1]
    if primary_len <= 0:
        primary_len = 1
    for idx, clen in verified[1:]:
        if len(picks) >= k:
            break
        if clen > secondary_length_ratio * primary_len:
            break
        # Dedup (different sample indices may have identical text)
        if trajectories[idx] == trajectories[picks[0]]:
            continue
        picks.append(idx)
    return picks


# ---------------------------------------------------------------------------
# Token-level weight recipe (Round 3)
# ---------------------------------------------------------------------------

# Exposed at module scope so the SFT collator and tests can import a single
# source of truth.  Consumer: src.student.lora_train (Round 1 milestone).
TOKEN_WEIGHT_RECIPE: dict[str, float] = {
    "final_answer":   3.0,   # numeric / LaTeX literal inside \\boxed{}
    "boxed_wrapper":  1.0,   # the tokens for "\\boxed{" and "}"
    "rationale":      0.5,   # tokens inside <think>...</think>
    "other":          1.0,
}


def _compute_sample_weight(support: int, n: int) -> float:
    """Support-weighted sample weight: 1 + 2 * support / N.

    Round 3 decision.  Unanimous agreement (support==N) yields weight 3.0;
    a bare majority on N=4 (support==3) yields 1 + 1.5 = 2.5; a weak
    agreement (support==1 on N=16, which would normally be dropped) yields
    1.125.
    """
    if n <= 0:
        return 1.0
    return 1.0 + 2.0 * (float(support) / float(n))


def _build_gcd_sft_row(
    prompt_row: dict[str, Any],
    completions: list[str],
    num_samples: int,
) -> dict[str, Any] | None:
    """Backward-compat single-row builder (Round 0 behaviour).

    Returns the primary (rank-1) row only.  Prefer _build_gcd_sft_rows
    for Round 3 multi-trace output.
    """
    rows = _build_gcd_sft_rows(prompt_row, completions, num_samples, max_traces=1)
    return rows[0] if rows else None


def _build_gcd_sft_rows(
    prompt_row: dict[str, Any],
    completions: list[str],
    num_samples: int,
    max_traces: int = 2,
) -> list[dict[str, Any]]:
    """Apply GCD to a single prompt's teacher completions; return up to
    *max_traces* SFT rows (Round 3 multi-trace retention).

    Each returned row carries support-weighted ``sample_weight`` and the
    token-level ``TOKEN_WEIGHT_RECIPE`` (the SFT collator applies the
    token weights at train time).
    """
    if not completions:
        return []

    # Step 1: extract answers
    extracted: list[str] = [extract_final_answer(c) for c in completions]

    # Step 2: cluster by answer equivalence
    clusters = _cluster_answers(extracted)

    # Step 3: find majority cluster (tie-break: alphabetically first canonical)
    if not clusters:
        return []
    max_size = max(len(v) for v in clusters.values())
    tied = sorted(k for k, v in clusters.items() if len(v) == max_size)
    majority_canonical = tied[0]
    majority_indices = clusters[majority_canonical]

    # Step 4: pick up to max_traces shortest verified trajectories
    ground_truth = str(prompt_row.get("target_answer", "") or "").strip() or None
    if not ground_truth:
        ground_truth = majority_canonical

    picks = _pick_topk_verified(
        completions, majority_indices, ground_truth, k=max_traces
    )
    if not picks:
        return []

    support = len(majority_indices)
    sample_weight = _compute_sample_weight(support, num_samples)

    rows: list[dict[str, Any]] = []
    base = {
        "id": str(prompt_row.get("id", "")),
        "prompt": str(prompt_row.get("prompt", "")),
        "teacher_num_samples": num_samples,
        "teacher_majority_size": support,
        "sample_weight": sample_weight,
        "token_weights": dict(TOKEN_WEIGHT_RECIPE),
    }
    for rank, sel_idx in enumerate(picks, start=1):
        traj = completions[sel_idx]
        row = dict(base)
        row.update({
            "id": f"{base['id']}__r{rank}" if rank > 1 else base["id"],
            "completion": traj,
            "teacher_selected_length": len(traj),
            "teacher_answer": extract_final_answer(traj),
            "retention_rank": rank,
        })
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Round 4: teacher majority-uplift gate
# ---------------------------------------------------------------------------

def _majority_answer(completions: list[str]) -> str | None:
    """Return the canonical answer of the majority cluster for *completions*,
    tie-broken alphabetically.  Returns None when *completions* is empty.
    """
    if not completions:
        return None
    extracted = [extract_final_answer(c) for c in completions]
    clusters = _cluster_answers(extracted)
    if not clusters:
        return None
    max_size = max(len(v) for v in clusters.values())
    tied = sorted(k for k, v in clusters.items() if len(v) == max_size)
    return tied[0]


def run_teacher_majority_gate(
    *,
    input_rows: list[dict[str, Any]],
    get_completions_greedy,
    get_completions_sampled,
    sampled_n: int,
    subset_size: int,
    min_uplift: float,
    seed: int = 42,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate majority-of-N vs greedy correctness on a prompt subset.

    Used as a Round 4 precondition before launching full GCD.  If the
    sampled teacher with majority voting does not beat greedy by at least
    *min_uplift*, GCD is wasted compute — the teacher's output distribution
    does not carry enough diverse-but-correct signal for consensus voting
    to help.

    Parameters
    ----------
    input_rows:
        Candidate prompts.  A random subset of ``subset_size`` rows is
        drawn with a fixed seed.  Each row must carry ``target_answer``.
    get_completions_greedy:
        Callable ``(row) -> str`` returning a single greedy completion.
    get_completions_sampled:
        Callable ``(row, n) -> list[str]`` returning *n* stochastic
        completions at the configured teacher temperature / top_p.
    sampled_n:
        N per prompt for the majority branch.  Default 8 per Round 4.
    subset_size:
        Number of prompts to score.  Default 300.
    min_uplift:
        Minimum (majority_rate - greedy_rate) to return ``passed=True``.
    seed:
        RNG seed for subset selection (and mock-only tie-breaking).
    report_path:
        Optional JSON destination for the gate report.

    Returns
    -------
    Report dict with keys: passed, greedy_correct, majority_correct,
    uplift, subset_size, sampled_n, min_uplift, per_prompt.
    """
    import json as _json  # noqa: PLC0415
    import random as _random  # noqa: PLC0415

    rng = _random.Random(seed)
    eligible = [r for r in input_rows if str(r.get("target_answer", "") or "").strip()]
    if not eligible:
        raise ValueError(
            "majority-gate requires rows with non-empty target_answer"
        )
    n_eff = min(subset_size, len(eligible))
    subset = rng.sample(eligible, n_eff) if n_eff < len(eligible) else list(eligible)

    per_prompt: list[dict[str, Any]] = []
    greedy_correct = 0
    majority_correct = 0
    for row in subset:
        gt = str(row.get("target_answer", "") or "").strip()
        # Greedy branch
        greedy_text = get_completions_greedy(row)
        greedy_ans = extract_final_answer(greedy_text)
        greedy_ok = bool(verify(gt, greedy_ans))
        # Majority branch
        sampled_texts = get_completions_sampled(row, sampled_n)
        majority_ans = _majority_answer(sampled_texts) or ""
        majority_ok = bool(verify(gt, majority_ans))
        greedy_correct += int(greedy_ok)
        majority_correct += int(majority_ok)
        per_prompt.append({
            "id": str(row.get("id", "")),
            "target_answer": gt,
            "greedy_answer": greedy_ans,
            "greedy_correct": greedy_ok,
            "majority_answer": majority_ans,
            "majority_correct": majority_ok,
        })

    total = len(subset)
    greedy_rate = greedy_correct / total if total else 0.0
    majority_rate = majority_correct / total if total else 0.0
    uplift = majority_rate - greedy_rate
    passed = uplift >= min_uplift
    report: dict[str, Any] = {
        "passed": passed,
        "greedy_correct": greedy_correct,
        "majority_correct": majority_correct,
        "greedy_rate": greedy_rate,
        "majority_rate": majority_rate,
        "uplift": uplift,
        "subset_size": total,
        "sampled_n": sampled_n,
        "min_uplift": min_uplift,
        "seed": seed,
        "per_prompt": per_prompt,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_json.dumps(report, ensure_ascii=False, indent=2))
    return report


# ---------------------------------------------------------------------------
# Adaptive-N early-stop criterion (Round 3)
# ---------------------------------------------------------------------------

def _adaptive_n_should_stop(
    completions: list[str],
    majority_size_threshold: float = 0.5,
    primary_length_cap: int = 2048,
) -> bool:
    """Return True when the current *completions* already support a
    confident SFT target, letting us skip further teacher sampling.

    Stop criteria (both must hold):
      * majority cluster covers > majority_size_threshold of samples
      * the shortest verified trajectory in the majority cluster has
        length ≤ primary_length_cap characters

    2048 chars ≈ 512 tokens on average — well inside the concise bucket
    for stage3 long-trace repair.  Increasing N beyond this point rarely
    improves sample quality.
    """
    if not completions:
        return False
    extracted = [extract_final_answer(c) for c in completions]
    clusters = _cluster_answers(extracted)
    if not clusters:
        return False
    max_size = max(len(v) for v in clusters.values())
    if max_size / len(completions) <= majority_size_threshold:
        return False
    tied = sorted(k for k, v in clusters.items() if len(v) == max_size)
    majority_indices = clusters[tied[0]]
    picks = _pick_topk_verified(completions, majority_indices, tied[0], k=1)
    if not picks:
        return False
    return len(completions[picks[0]]) <= primary_length_cap


# ---------------------------------------------------------------------------
# Teacher vLLM launch helpers (thin wrappers — lazy import to avoid top-level
# vLLM/torch dependency)
# ---------------------------------------------------------------------------

def _load_vllm_helpers():
    """Import and return (_instantiate_vllm, _build_sampling_params,
    _build_lora_request) from the existing eval_official_vllm_proxy module.

    # TODO: integration point — ensure eval_official_vllm_proxy is importable
    #       in the target environment before calling this function.
    """
    # pylint: disable=import-outside-toplevel
    import importlib
    mod = importlib.import_module("scripts.eval_official_vllm_proxy")
    return (
        mod._instantiate_vllm,
        mod._build_sampling_params,
        mod._build_lora_request,
    )


def _build_teacher_llm_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Build vLLM LLM kwargs for teacher inference.

    # TODO: integration point — review tensor_parallel_size and
    #       gpu_memory_utilization for the target hardware before production
    #       runs.  Current defaults are conservative for an 80 GB GPU.
    """
    from src.competition.official_metric_contract import RUNTIME_LLM_KWARGS  # noqa: PLC0415

    kwargs = dict(RUNTIME_LLM_KWARGS)
    # Override: teacher samples N>1 completions; vLLM needs best_of >= N or
    # we call generate() N times with n=1.  We use the latter (simpler).
    return kwargs


def _build_teacher_sampling_kwargs(
    temperature: float,
    top_p: float,
    max_tokens: int,
    n: int,
) -> dict[str, Any]:
    """Build SamplingParams kwargs for teacher sampling.

    # TODO: integration point — if vLLM supports n > 1 per generate() call
    #       in the installed version, set n=<num_samples> here and remove the
    #       loop in run_teacher_distillation().
    """
    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_teacher_distillation(
    *,
    input_rows: list[dict[str, Any]],
    output_path: Path,
    teacher_adapter_dir: str | Path | None,
    config: dict[str, Any],
    num_samples: int,
    teacher_temperature: float,
    top_p: float,
    max_tokens: int,
    workdir: Path,
    dry_run: bool,
    adaptive_n_tiers: list[int] | None = None,
    max_traces_per_prompt: int = 2,
) -> dict[str, Any]:
    """Orchestrate GCD teacher distillation over *input_rows*.

    For each prompt:
      1. Call teacher vLLM with N samples at temperature=teacher_temperature.
      2. Cluster answers with _cluster_answers().
      3. Pick the shortest verified trajectory from the majority cluster.
      4. Emit an SFT row.

    Parameters
    ----------
    input_rows:
        Normalised rows from round1_prepare_nemotron_math_v2 (or compatible
        schema).
    output_path:
        Destination JSONL for SFT rows.
    teacher_adapter_dir:
        Path to LoRA adapter directory for teacher.  If None, the base model
        is used without any adapter (base-model teacher).
    config:
        Parsed YAML config dict (e.g. from configs/train_lora.yaml) used to
        resolve the base model path.
    num_samples:
        N completions to request per prompt.
    teacher_temperature:
        Sampling temperature for teacher (1.0 recommended per ScPO).
    top_p:
        Nucleus sampling top-p value.
    max_tokens:
        Max new tokens per completion (7680 per runtime contract).
    workdir:
        Directory for intermediate artifacts (e.g. per-batch raw completions).
    dry_run:
        When True, skip vLLM; use mock completions instead.

    Returns
    -------
    Report dict with aggregate statistics.
    """
    workdir.mkdir(parents=True, exist_ok=True)

    # Adaptive-N tiers (Round 3).  If None, fall back to single-tier at
    # *num_samples* (Round 0 behaviour).
    if adaptive_n_tiers is None:
        adaptive_n_tiers = [num_samples]
    # Enforce strictly increasing and final tier == num_samples
    adaptive_n_tiers = sorted(set(int(t) for t in adaptive_n_tiers))
    if adaptive_n_tiers[-1] != num_samples:
        adaptive_n_tiers.append(num_samples)
        adaptive_n_tiers = sorted(set(adaptive_n_tiers))

    output_rows: list[dict[str, Any]] = []
    n_no_majority = 0
    n_no_verified = 0
    total_selected_len = 0
    total_majority_size = 0
    tier_stop_counts: dict[int, int] = {t: 0 for t in adaptive_n_tiers}
    total_samples_drawn = 0

    if dry_run:
        print(
            "[round1_gcd_teacher_distill] dry-run: using mock completions, "
            "vLLM NOT instantiated.",
            flush=True,
        )
        llm = None
        get_completions = lambda row: _make_mock_completions(  # noqa: E731
            row, int(row.get("_gcd_n", num_samples))
        )
    else:
        # --- vLLM integration point ---
        # TODO: Confirm the base model path is downloadable / cached before
        #       calling _load_vllm_helpers().
        print(
            "[round1_gcd_teacher_distill] instantiating teacher vLLM ...",
            flush=True,
        )
        _instantiate_vllm, _build_sampling_params, _build_lora_request = (
            _load_vllm_helpers()
        )
        llm_kwargs = _build_teacher_llm_kwargs(config)
        from src.student.lora_train import resolve_model_path  # noqa: PLC0415

        base_model_path = resolve_model_path(config)
        llm = _instantiate_vllm(str(base_model_path), llm_kwargs)
        tokenizer = llm.get_tokenizer()

        sampling_kwargs = _build_teacher_sampling_kwargs(
            temperature=teacher_temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,  # single sample per call; loop for N samples
        )
        sampling_params = _build_sampling_params(sampling_kwargs)
        lora_request = (
            _build_lora_request(teacher_adapter_dir)
            if teacher_adapter_dir is not None
            else None
        )

        from src.competition.official_metric_contract import build_official_prompt  # noqa: PLC0415

        def get_completions(row: dict[str, Any]) -> list[str]:
            """Run teacher N_requested times for a single row and collect
            completions.  N_requested is taken from row['_gcd_n'] (set by
            the adaptive-N loop) and falls back to num_samples.
            """
            prompt_text = str(row.get("prompt", ""))
            n_req = int(row.get("_gcd_n", num_samples))
            formatted = build_official_prompt(prompt_text, tokenizer)
            completions_local: list[str] = []
            for _ in range(n_req):
                outputs = llm.generate(
                    [formatted],
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                )
                completions_local.append(outputs[0].outputs[0].text)
            return completions_local

    t0 = time.monotonic()
    for i, row in enumerate(input_rows):
        # ---- Adaptive-N sampling loop ----
        # Progressively draw samples, checking the early-stop criterion
        # after each tier.  get_completions() always returns the *full*
        # N for simplicity, so for tier expansion we draw only the
        # incremental count.
        completions: list[str] = []
        stop_at_tier: int = adaptive_n_tiers[-1]
        for tier in adaptive_n_tiers:
            need = tier - len(completions)
            if need > 0:
                extra_row = dict(row)
                extra_row["_gcd_n"] = need
                new_comp = get_completions(extra_row)
                # Respect whatever get_completions returned; pad/truncate
                completions.extend(new_comp[:need])
            total_samples_drawn += len(completions) - (len(completions) - need if need > 0 else 0)
            if tier < adaptive_n_tiers[-1] and _adaptive_n_should_stop(completions):
                stop_at_tier = tier
                break
            stop_at_tier = tier
        tier_stop_counts[stop_at_tier] = tier_stop_counts.get(stop_at_tier, 0) + 1
        actual_n = len(completions)

        sft_rows_for_prompt = _build_gcd_sft_rows(
            row, completions, actual_n, max_traces=max_traces_per_prompt
        )
        if not sft_rows_for_prompt:
            n_no_verified += 1
            continue
        for sft_row in sft_rows_for_prompt:
            if sft_row["teacher_majority_size"] == 0:
                n_no_majority += 1
                continue
            output_rows.append(sft_row)
            total_selected_len += sft_row["teacher_selected_length"]
            total_majority_size += sft_row["teacher_majority_size"]

        if (i + 1) % 100 == 0:
            elapsed = time.monotonic() - t0
            print(
                f"[round1_gcd_teacher_distill] processed {i + 1}/{len(input_rows)} "
                f"prompts ({elapsed:.1f}s, {len(output_rows)} SFT rows, "
                f"tiers stopped: {tier_stop_counts})",
                flush=True,
            )

    n_out = len(output_rows)
    report: dict[str, Any] = {
        "num_input_rows": len(input_rows),
        "num_output_rows": n_out,
        "num_no_verified": n_no_verified,
        "num_no_majority": n_no_majority,
        "teacher_num_samples": num_samples,
        "adaptive_n_tiers": adaptive_n_tiers,
        "adaptive_n_tier_stop_counts": tier_stop_counts,
        "max_traces_per_prompt": max_traces_per_prompt,
        "total_teacher_samples_drawn": total_samples_drawn,
        "teacher_temperature": teacher_temperature,
        "teacher_top_p": top_p,
        "teacher_max_tokens": max_tokens,
        "teacher_adapter_dir": str(teacher_adapter_dir) if teacher_adapter_dir else None,
        "mean_majority_size": (total_majority_size / n_out) if n_out else 0.0,
        "mean_selected_length_chars": (total_selected_len / n_out) if n_out else 0.0,
        "token_weight_recipe": dict(TOKEN_WEIGHT_RECIPE),
        "dry_run": dry_run,
    }
    write_jsonl(output_path, output_rows)
    report_path = output_path.with_suffix("").with_suffix(".gcd_report.json")
    write_json(report_path, report)
    print(
        f"[round1_gcd_teacher_distill] wrote {n_out} SFT rows to {output_path}",
        flush=True,
    )
    print(
        f"[round1_gcd_teacher_distill] report -> {report_path}",
        flush=True,
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Greedy Consensus Distillation (GCD): run teacher inference with "
            "N samples per prompt, cluster by answer equivalence, and select "
            "the shortest verified trajectory from the majority cluster as the "
            "student SFT target.  Based on ScPO (arXiv 2411.04109) and PCPO "
            "(ACL Findings 2025).  Pass --dry-run to skip vLLM."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Path to input JSONL (output of round1_prepare_nemotron_math_v2.py "
            "or any JSONL with {id, prompt, target_answer} fields). "
            "Required unless --dry-run is set."
        ),
    )
    parser.add_argument(
        "--output",
        default="data/processed/round1_gcd_distill.jsonl",
        help="Destination JSONL for GCD SFT rows (default: %(default)s).",
    )
    parser.add_argument(
        "--teacher-adapter-dir",
        default=None,
        dest="teacher_adapter_dir",
        help=(
            "Path to LoRA adapter directory loaded onto the teacher model. "
            "If absent, the base model is used as the teacher."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/train_lora.yaml",
        help="Training config YAML used to resolve the base model path.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        dest="num_samples",
        help="Number of teacher completions to sample per prompt (default: 16).",
    )
    parser.add_argument(
        "--teacher-temperature",
        type=float,
        default=0.7,
        dest="teacher_temperature",
        help=(
            "Sampling temperature for teacher.  Round 4 default: 0.7."
            "  T=1.0 produced too many degenerate tails in the Nemotron-12B"
            " teacher; 0.7 retains diversity required by GCD clustering"
            " while cutting off low-probability noise (GPT-5.4 Pro R4)."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        dest="top_p",
        help=(
            "Nucleus sampling top-p value.  Round 4 default: 0.95 (was 1.0)."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=7680,
        dest="max_tokens",
        help=(
            "Max new tokens per teacher completion. "
            "7680 matches the Kaggle runtime contract (default: 7680)."
        ),
    )
    parser.add_argument(
        "--workdir",
        default="data/workdir/round1_gcd",
        help="Directory for intermediate workdir artifacts (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Skip vLLM instantiation and use mock completions for 3 synthetic "
            "prompts.  Validates the full GCD pipeline on CPU with no GPU deps."
        ),
    )
    parser.add_argument(
        "--adaptive-n-tiers",
        nargs="+",
        type=int,
        default=[4, 8, 16],
        dest="adaptive_n_tiers",
        help=(
            "Tiered adaptive-N sampling sequence (default: 4 8 16 per Round 3). "
            "Pass a single value to disable adaptive sampling."
        ),
    )
    parser.add_argument(
        "--max-traces-per-prompt",
        type=int,
        default=2,
        dest="max_traces_per_prompt",
        help=(
            "Max number of verified concise trajectories to retain per "
            "prompt (default 2 per Round 3).  Set to 1 to replicate "
            "Round 0 single-trace behaviour."
        ),
    )
    # ---- Round 4: teacher majority-uplift gate --------------------------
    parser.add_argument(
        "--teacher-majority-gate",
        action="store_true",
        dest="teacher_majority_gate",
        help=(
            "Round 4 precondition.  Before running full GCD, sample a"
            " subset of prompts and compare greedy (T=0) correctness to"
            " majority-of-N correctness at the configured teacher T/top_p."
            "  If the uplift is below --majority-gate-min-uplift, abort"
            " the run with a non-zero exit code — investing compute in"
            " GCD is only worthwhile when majority voting actually"
            " outperforms greedy on this teacher."
        ),
    )
    parser.add_argument(
        "--majority-gate-n",
        type=int,
        default=8,
        dest="majority_gate_n",
        help="Samples per prompt for the majority gate (default 8).",
    )
    parser.add_argument(
        "--majority-gate-subset-size",
        type=int,
        default=300,
        dest="majority_gate_subset_size",
        help="Prompts used to evaluate the majority gate (default 300).",
    )
    parser.add_argument(
        "--majority-gate-min-uplift",
        type=float,
        default=0.02,
        dest="majority_gate_min_uplift",
        help=(
            "Minimum (majority_correct - greedy_correct) rate required to"
            " proceed past the gate (default 0.02 = +2pp)."
        ),
    )
    parser.add_argument(
        "--majority-gate-seed",
        type=int,
        default=42,
        dest="majority_gate_seed",
        help="RNG seed for majority-gate subset sampling (default 42).",
    )
    parser.add_argument(
        "--majority-gate-report",
        default=None,
        dest="majority_gate_report",
        help=(
            "Optional path for the majority-gate JSON report. "
            "Defaults to <output>.majority_gate.json next to --output."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    dry_run: bool = args.dry_run or (args.input is None)
    output_path = Path(args.output)
    workdir = Path(args.workdir)

    if dry_run:
        input_rows: list[dict[str, Any]] = list(_DRY_RUN_PROMPTS)
        config: dict[str, Any] = {}
    else:
        from src.common.io import read_yaml  # noqa: PLC0415

        config = read_yaml(args.config)
        print(
            f"[round1_gcd_teacher_distill] loading {args.input} ...",
            flush=True,
        )
        input_rows = load_jsonl(args.input)
        print(
            f"[round1_gcd_teacher_distill] loaded {len(input_rows)} rows.",
            flush=True,
        )

    # ---- Round 4: optional teacher majority-uplift gate ----
    if getattr(args, "teacher_majority_gate", False):
        gate_report_path = (
            Path(args.majority_gate_report)
            if args.majority_gate_report
            else output_path.with_suffix("").with_suffix(".majority_gate.json")
        )
        if dry_run:
            # In dry-run we reuse the mock completions: greedy picks the
            # first mock, sampled draws N mocks.  Target answers are
            # already present on _DRY_RUN_PROMPTS, so the gate trivially
            # passes (uplift = 0.0 since both branches are correct by
            # construction; this still exercises the plumbing).
            def _greedy_dry(row: dict[str, Any]) -> str:
                return _make_mock_completions(row, 1)[0]

            def _sampled_dry(row: dict[str, Any], n: int) -> list[str]:
                return _make_mock_completions(row, n)

            gate_report = run_teacher_majority_gate(
                input_rows=input_rows,
                get_completions_greedy=_greedy_dry,
                get_completions_sampled=_sampled_dry,
                sampled_n=args.majority_gate_n,
                subset_size=min(args.majority_gate_subset_size, len(input_rows)),
                min_uplift=0.0,  # dry-run always passes
                seed=args.majority_gate_seed,
                report_path=gate_report_path,
            )
        else:
            # Real gate: instantiate vLLM once, run greedy (T=0) and
            # sampled branches, then dispose.
            print(
                "[round1_gcd_teacher_distill] running teacher majority-uplift"
                f" gate on {args.majority_gate_subset_size} prompts ...",
                flush=True,
            )
            _instantiate_vllm, _build_sampling_params, _build_lora_request = (
                _load_vllm_helpers()
            )
            from src.student.lora_train import resolve_model_path  # noqa: PLC0415
            from src.competition.official_metric_contract import (  # noqa: PLC0415
                build_official_prompt,
            )

            llm_kwargs = _build_teacher_llm_kwargs(config)
            base_model_path = resolve_model_path(config)
            gate_llm = _instantiate_vllm(str(base_model_path), llm_kwargs)
            gate_tokenizer = gate_llm.get_tokenizer()
            lora_request = (
                _build_lora_request(args.teacher_adapter_dir)
                if args.teacher_adapter_dir is not None
                else None
            )
            sp_greedy = _build_sampling_params(
                _build_teacher_sampling_kwargs(
                    temperature=0.0, top_p=1.0,
                    max_tokens=args.max_tokens, n=1,
                )
            )
            sp_sampled = _build_sampling_params(
                _build_teacher_sampling_kwargs(
                    temperature=args.teacher_temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens, n=1,
                )
            )

            def _greedy_real(row: dict[str, Any]) -> str:
                formatted = build_official_prompt(
                    str(row.get("prompt", "")), gate_tokenizer
                )
                out = gate_llm.generate(
                    [formatted], sampling_params=sp_greedy,
                    lora_request=lora_request,
                )
                return out[0].outputs[0].text

            def _sampled_real(row: dict[str, Any], n: int) -> list[str]:
                formatted = build_official_prompt(
                    str(row.get("prompt", "")), gate_tokenizer
                )
                texts: list[str] = []
                for _ in range(n):
                    out = gate_llm.generate(
                        [formatted], sampling_params=sp_sampled,
                        lora_request=lora_request,
                    )
                    texts.append(out[0].outputs[0].text)
                return texts

            gate_report = run_teacher_majority_gate(
                input_rows=input_rows,
                get_completions_greedy=_greedy_real,
                get_completions_sampled=_sampled_real,
                sampled_n=args.majority_gate_n,
                subset_size=args.majority_gate_subset_size,
                min_uplift=args.majority_gate_min_uplift,
                seed=args.majority_gate_seed,
                report_path=gate_report_path,
            )
            # Release vLLM handle before the main distillation instantiates
            # its own LLM, avoiding duplicate GPU footprint.
            del gate_llm

        print(
            "[round1_gcd_teacher_distill] majority-gate report -> "
            f"{gate_report_path}",
            flush=True,
        )
        print(
            "[round1_gcd_teacher_distill] greedy_rate="
            f"{gate_report['greedy_rate']:.4f} majority_rate="
            f"{gate_report['majority_rate']:.4f} uplift="
            f"{gate_report['uplift']:+.4f} (min_uplift="
            f"{gate_report['min_uplift']:+.4f})",
            flush=True,
        )
        if not gate_report["passed"]:
            print(
                "[round1_gcd_teacher_distill] ABORT: majority-uplift gate"
                " failed. Lower teacher T or pick a stronger teacher adapter"
                " before investing compute in GCD.",
                flush=True,
            )
            sys.exit(42)

    run_teacher_distillation(
        input_rows=input_rows,
        output_path=output_path,
        teacher_adapter_dir=args.teacher_adapter_dir,
        config=config,
        num_samples=args.num_samples,
        teacher_temperature=args.teacher_temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        workdir=workdir,
        dry_run=dry_run,
        adaptive_n_tiers=args.adaptive_n_tiers,
        max_traces_per_prompt=args.max_traces_per_prompt,
    )


if __name__ == "__main__":
    main()
