"""round1_gcd_teacher_distill.py
================================
Greedy Consensus Distillation (GCD): generate N teacher trajectories per
prompt, cluster by answer equivalence, and select the shortest verified
trajectory from the majority cluster as the student's SFT target.

Theory
------
Standard best-of-N distillation selects any correct trajectory, which biases
the student toward longer, noisier reasoning chains.  GCD tightens this:

  1. Sample N completions from the teacher at temperature=1.0 (diverse but
     not degenerate).
  2. Cluster the completions by answer equivalence using the competition's
     ``verify()`` function.  Each cluster's "canonical answer" is the
     extracted answer of its first member.
  3. Identify the majority cluster (most samples, tie broken by
     alphabetically-first canonical answer for reproducibility).
  4. Within the majority cluster, keep only trajectories whose extracted
     answer ``verify()``-matches the ground-truth answer (or, if no ground
     truth is available, the majority-cluster canonical answer).
  5. From those verified trajectories, select the one with the fewest tokens
     as the SFT target.  Shorter chains are preferred because they reduce the
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
      "teacher_num_samples":    int,   # N requested from teacher
      "teacher_majority_size":  int,   # size of the majority answer cluster
      "teacher_selected_length": int,  # token count of selected trajectory
      "teacher_answer":         str,   # extracted answer from selected trajectory
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
        --teacher-temperature 1.0 \\
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

    From the subset of trajectories given by *indices*, filters to those
    whose extracted answer verify()-matches *target_answer* (when provided).
    Among the verified subset, returns the index of the trajectory with the
    fewest characters (used as a token-count proxy to avoid loading a
    tokenizer in the distillation loop).

    Parameters
    ----------
    trajectories:
        Full list of trajectory strings (all N samples for a single prompt).
    indices:
        Indices into *trajectories* to consider (the majority cluster).
    target_answer:
        Ground-truth answer string.  If None or empty, the majority-cluster
        canonical answer is used (passed by the caller).

    Returns
    -------
    Index into *trajectories* of the selected trajectory, or None if no
    verified trajectory exists in the cluster.
    """
    if not indices:
        return None
    verified: list[tuple[int, int]] = []  # (index, char_len)
    for idx in indices:
        traj = trajectories[idx]
        extracted = extract_final_answer(traj)
        if target_answer and verify(str(target_answer), extracted):
            verified.append((idx, len(traj)))
        elif not target_answer:
            # No ground truth: accept all cluster members
            verified.append((idx, len(traj)))
    if not verified:
        return None
    # Sort ascending by char length; pick shortest
    verified.sort(key=lambda pair: pair[1])
    return verified[0][0]


def _build_gcd_sft_row(
    prompt_row: dict[str, Any],
    completions: list[str],
    num_samples: int,
) -> dict[str, Any] | None:
    """Apply GCD to a single prompt's teacher completions.

    Returns a GCD SFT row dict, or None when no valid trajectory was found.
    """
    if not completions:
        return None

    # Step 1: extract answers
    extracted: list[str] = [extract_final_answer(c) for c in completions]

    # Step 2: cluster by answer equivalence
    clusters = _cluster_answers(extracted)

    # Step 3: find majority cluster (tie-break: alphabetically first canonical)
    if not clusters:
        return None
    majority_canonical = max(
        clusters.keys(),
        key=lambda k: (len(clusters[k]), -ord(k[0]) if k else 0),
    )
    # Use sorted to make tie-breaking deterministic
    majority_canonical = max(
        clusters.keys(),
        key=lambda k: len(clusters[k]),
    )
    # Stable tie-break by canonical string (alphabetically first)
    max_size = max(len(v) for v in clusters.values())
    tied = sorted(k for k, v in clusters.items() if len(v) == max_size)
    majority_canonical = tied[0]
    majority_indices = clusters[majority_canonical]

    # Step 4: pick shortest verified trajectory from majority cluster
    ground_truth = str(prompt_row.get("target_answer", "") or "").strip() or None
    if not ground_truth:
        # Fall back to majority-cluster canonical answer when no GT
        ground_truth = majority_canonical

    selected_idx = _pick_shortest_verified(completions, majority_indices, ground_truth)
    if selected_idx is None:
        return None

    selected_traj = completions[selected_idx]
    teacher_answer = extract_final_answer(selected_traj)

    return {
        "id": str(prompt_row.get("id", "")),
        "prompt": str(prompt_row.get("prompt", "")),
        "completion": selected_traj,
        "teacher_num_samples": num_samples,
        "teacher_majority_size": len(majority_indices),
        "teacher_selected_length": len(selected_traj),
        "teacher_answer": teacher_answer,
    }


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

    output_rows: list[dict[str, Any]] = []
    n_no_majority = 0
    n_no_verified = 0
    total_selected_len = 0
    total_majority_size = 0

    if dry_run:
        print(
            "[round1_gcd_teacher_distill] dry-run: using mock completions, "
            "vLLM NOT instantiated.",
            flush=True,
        )
        llm = None
        get_completions = lambda row: _make_mock_completions(row, num_samples)  # noqa: E731
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
            """Run teacher N times for a single row and collect completions."""
            prompt_text = str(row.get("prompt", ""))
            formatted = build_official_prompt(prompt_text, tokenizer)
            completions_local: list[str] = []
            for _ in range(num_samples):
                outputs = llm.generate(
                    [formatted],
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                )
                completions_local.append(outputs[0].outputs[0].text)
            return completions_local

    t0 = time.monotonic()
    for i, row in enumerate(input_rows):
        completions = get_completions(row)
        sft_row = _build_gcd_sft_row(row, completions, num_samples)
        if sft_row is None:
            # GCD found no majority cluster or no verified trajectory
            n_no_verified += 1
            continue
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
                f"rows ({elapsed:.1f}s elapsed, {len(output_rows)} SFT rows so far)",
                flush=True,
            )

    n_out = len(output_rows)
    report: dict[str, Any] = {
        "num_input_rows": len(input_rows),
        "num_output_rows": n_out,
        "num_no_verified": n_no_verified,
        "num_no_majority": n_no_majority,
        "teacher_num_samples": num_samples,
        "teacher_temperature": teacher_temperature,
        "teacher_top_p": top_p,
        "teacher_max_tokens": max_tokens,
        "teacher_adapter_dir": str(teacher_adapter_dir) if teacher_adapter_dir else None,
        "mean_majority_size": (total_majority_size / n_out) if n_out else 0.0,
        "mean_selected_length_chars": (total_selected_len / n_out) if n_out else 0.0,
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
        default=1.0,
        dest="teacher_temperature",
        help=(
            "Sampling temperature for teacher. 1.0 (default) gives diverse "
            "completions as required by the GCD clustering step."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        dest="top_p",
        help="Nucleus sampling top-p value (default: 1.0).",
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
    )


if __name__ == "__main__":
    main()
