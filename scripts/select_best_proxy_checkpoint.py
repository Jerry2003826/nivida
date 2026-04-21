"""Pick the best checkpoint inside a single stage's output directory.

Runs the canonical hard-triad and all-family proxy evals over every
``checkpoint-*`` subdirectory plus the stage's final adapter, compares them
pairwise via the shared proxy_selection helper with ``tiebreak_default="final"``,
and copies the winning adapter (plus its eval artifacts) to a stable path.

Why external:
- The trainer does not know the hard-triad / all-family proxies (those are
  measured after training with a separate inference pass), so
  ``load_best_model_at_end`` would only track ``eval_loss``, which we do not
  trust as the competition proxy.
- Keeping selection logic in :mod:`src.student.proxy_selection` means the
  rule stays identical to ``scripts.select_final_adapter`` (stage2 vs stage3).

Checkpoint directories saved by the HF Trainer sometimes lack
``adapter_config.json`` depending on the peft version; in that case we copy
the config from the stage's final directory before inference, since all
checkpoints of the same stage share the same rank / target_modules.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import read_yaml, write_json  # noqa: E402
from src.experiments.eval_competition_replica import evaluate_replica  # noqa: E402
from src.student.inference import run_inference  # noqa: E402
from src.student.package_submission import validate_adapter_dir  # noqa: E402
from src.student.proxy_selection import (  # noqa: E402
    compare_proxy_pairs,
    copy_adapter_dir,
    load_proxy_eval,
)


CHECKPOINT_STEP_RE = re.compile(r"^checkpoint-(\d+)$")
SCRATCH_ROOT = Path("artifacts/_proxy_checkpoint_scratch")


def derive_default_workdir(stage_output_dir: Path) -> Path:
    raw_name = stage_output_dir.as_posix().strip("/")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "__", raw_name).strip("_")
    if not safe_name:
        safe_name = "stage_output_dir"
    return SCRATCH_ROOT / safe_name


def discover_candidate_dirs(stage_output_dir: Path) -> list[tuple[str, Path]]:
    """Return ``[(name, path), ...]`` sorted by checkpoint step, final dir last."""
    if not stage_output_dir.is_dir():
        raise SystemExit(f"stage_output_dir not found: {stage_output_dir}")

    checkpoint_entries: list[tuple[int, str, Path]] = []
    for entry in stage_output_dir.iterdir():
        if not entry.is_dir():
            continue
        match = CHECKPOINT_STEP_RE.match(entry.name)
        if match:
            checkpoint_entries.append((int(match.group(1)), entry.name, entry))
    checkpoint_entries.sort(key=lambda t: t[0])

    results: list[tuple[str, Path]] = [
        (name, path) for _, name, path in checkpoint_entries
    ]
    results.append(("final", stage_output_dir))
    return results


def maybe_backfill_adapter_config(candidate_dir: Path, reference_dir: Path) -> bool:
    """Copy ``adapter_config.json`` from the reference dir when the candidate has weights but no config.

    Returns True when a backfill copy happened.
    """
    has_weights = any(
        (candidate_dir / name).exists()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )
    if not has_weights:
        return False
    target = candidate_dir / "adapter_config.json"
    if target.exists():
        return False
    source = reference_dir / "adapter_config.json"
    if not source.exists():
        return False
    if candidate_dir.resolve() == reference_dir.resolve():
        return False
    shutil.copy(source, target)
    return True


def is_valid_candidate(candidate_dir: Path) -> bool:
    try:
        validate_adapter_dir(str(candidate_dir))
    except Exception:
        return False
    return True


def _average_proxy_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Average numeric fields across repeated proxy eval payloads.

    Non-numeric fields are taken from the last run; repeat-level breakdowns are
    preserved under the ``repeats`` key for audit.
    """
    if not payloads:
        raise ValueError("no payloads to average")
    if len(payloads) == 1:
        return payloads[0]
    numeric_keys: set[str] = set()
    for payload in payloads:
        for key, value in payload.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numeric_keys.add(key)
    averaged = dict(payloads[-1])
    for key in numeric_keys:
        averaged[key] = sum(float(p.get(key, 0.0)) for p in payloads) / len(payloads)
    averaged["num_repeats"] = len(payloads)
    averaged["repeats"] = payloads
    return averaged


def run_proxy_pair(
    *,
    config: dict[str, Any],
    adapter_dir: Path,
    input_path: Path,
    workdir: Path,
    candidate_name: str,
    proxy_label: str,
    max_new_tokens: int,
    official_eval: bool = True,
    num_repeats: int = 3,
) -> tuple[Path, dict[str, Any]]:
    """Run run_inference + evaluate_replica for one (candidate, proxy) pair.

    Always uses the official Kaggle harness sampling parameters (do_sample=True,
    T=1.0, top_p=1.0, max_new_tokens=3584, chat_thinking prompt) by default, and
    averages ``num_repeats`` sampled runs because the leaderboard scorer is
    itself a sampling-based judge.

    Returns ``(eval_json_path, structured_eval_dict)``.
    """
    eval_path = workdir / f"{candidate_name}_{proxy_label}_eval.json"
    # When official_eval is on, prefer the official max_new_tokens=3584 unless
    # the caller explicitly opts to clip with a smaller value.
    effective_max_tokens = max_new_tokens
    if official_eval and max_new_tokens and max_new_tokens < 3584:
        effective_max_tokens = 3584
    payloads: list[dict[str, Any]] = []
    pred_paths: list[Path] = []
    repeats = max(1, int(num_repeats) if official_eval else 1)
    for repeat_idx in range(repeats):
        suffix = f"_r{repeat_idx}" if repeats > 1 else ""
        pred_path = workdir / f"{candidate_name}_{proxy_label}_pred{suffix}.jsonl"
        run_inference(
            config,
            input_path=input_path,
            adapter_dir=adapter_dir,
            output_path=pred_path,
            max_new_tokens=effective_max_tokens,
            official_eval=official_eval,
        )
        payload = evaluate_replica(
            prediction_path=pred_path,
            label_path=input_path,
            require_complete_coverage=True,
        )
        payload["_repeat_index"] = repeat_idx
        payload["_prediction_path"] = str(pred_path)
        payloads.append(payload)
        pred_paths.append(pred_path)
    averaged = _average_proxy_payloads(payloads)
    averaged["official_eval"] = official_eval
    averaged["max_new_tokens_used"] = effective_max_tokens
    averaged["prediction_paths"] = [str(p) for p in pred_paths]
    write_json(eval_path, averaged)
    return eval_path, load_proxy_eval(eval_path)


def select_best_checkpoint(
    *,
    config: dict[str, Any],
    stage_output_dir: Path,
    hard_input: Path,
    all_input: Path,
    output_best_dir: Path,
    output_hard_eval: Path,
    output_all_eval: Path,
    output_json: Path,
    max_new_tokens: int,
    workdir: Path | None,
    official_eval: bool = True,
    num_repeats: int = 3,
) -> dict[str, Any]:
    if workdir is None:
        workdir = derive_default_workdir(stage_output_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    candidates = discover_candidate_dirs(stage_output_dir)
    if not candidates:
        raise SystemExit(f"no candidate directories found under {stage_output_dir}")

    # Final dir is always last in the list and is used as the reference for
    # adapter_config.json backfill.
    final_name, final_dir = candidates[-1]
    assert final_name == "final"

    records: list[dict[str, Any]] = []
    for name, path in candidates:
        backfilled = maybe_backfill_adapter_config(path, final_dir)
        record: dict[str, Any] = {
            "name": name,
            "path": str(path),
            "adapter_config_fallback_applied": backfilled,
            "skipped": False,
            "skip_reason": None,
        }
        if not is_valid_candidate(path):
            record["skipped"] = True
            record["skip_reason"] = "validate_adapter_dir failed"
            records.append(record)
            continue

        hard_path, hard_eval = run_proxy_pair(
            config=config,
            adapter_dir=path,
            input_path=hard_input,
            workdir=workdir,
            candidate_name=name,
            proxy_label="hard",
            max_new_tokens=max_new_tokens,
            official_eval=official_eval,
            num_repeats=num_repeats,
        )
        all_path, all_eval = run_proxy_pair(
            config=config,
            adapter_dir=path,
            input_path=all_input,
            workdir=workdir,
            candidate_name=name,
            proxy_label="all",
            max_new_tokens=max_new_tokens,
            official_eval=official_eval,
            num_repeats=num_repeats,
        )
        record["hard_eval"] = hard_eval
        record["all_eval"] = all_eval
        record["hard_eval_path"] = str(hard_path)
        record["all_eval_path"] = str(all_path)
        records.append(record)

    scored = [r for r in records if not r["skipped"]]
    if not scored:
        raise SystemExit(
            "no valid candidate produced proxy evals; every checkpoint failed "
            "validate_adapter_dir. Inspect candidate skip_reason in the output json."
        )

    # Pairwise fold: seed with the first valid candidate, compare against the
    # rest. When 'final' is one of the two, it wins ties; otherwise the
    # current best holds the tie.
    best = scored[0]
    pairwise_decisions: list[dict[str, Any]] = []
    for challenger in scored[1:]:
        tiebreak = (
            "final"
            if "final" in (best["name"], challenger["name"])
            else best["name"]
        )
        decision = compare_proxy_pairs(
            left_name=best["name"],
            left_all=best["all_eval"],
            left_hard=best["hard_eval"],
            right_name=challenger["name"],
            right_all=challenger["all_eval"],
            right_hard=challenger["hard_eval"],
            tiebreak_default=tiebreak,
        )
        pairwise_decisions.append(
            {"left": best["name"], "right": challenger["name"], **decision}
        )
        if decision["winner"] == challenger["name"]:
            best = challenger

    copy_adapter_dir(best["path"], str(output_best_dir))
    output_hard_eval.parent.mkdir(parents=True, exist_ok=True)
    output_all_eval.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best["hard_eval_path"], output_hard_eval)
    shutil.copy(best["all_eval_path"], output_all_eval)

    summary = {
        "stage_output_dir": str(stage_output_dir),
        "workdir": str(workdir),
        "selected_candidate": best["name"],
        "selected_adapter_dir": str(output_best_dir),
        "selected_hard_eval": str(output_hard_eval),
        "selected_all_eval": str(output_all_eval),
        "candidates": records,
        "pairwise_decisions": pairwise_decisions,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the best checkpoint inside a stage output dir by running "
            "hard-triad and all-family proxy evals over every saved checkpoint "
            "plus the final adapter."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage-output-dir", required=True, type=Path)
    parser.add_argument("--hard-proxy-input", required=True, type=Path)
    parser.add_argument("--all-proxy-input", required=True, type=Path)
    parser.add_argument("--output-best-dir", required=True, type=Path)
    parser.add_argument("--output-hard-eval", required=True, type=Path)
    parser.add_argument("--output-all-eval", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=3584)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Directory for per-candidate prediction / eval intermediates.",
    )
    parser.add_argument(
        "--no-official-eval",
        action="store_true",
        help=(
            "Disable the official Kaggle harness sampling override. Only useful "
            "for debugging; production selection must mirror the leaderboard."
        ),
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=3,
        help=(
            "Number of sampled generations per candidate per proxy; payload "
            "numerics are averaged across repeats. Only used when official "
            "eval is on (do_sample=True)."
        ),
    )
    args = parser.parse_args()

    config = read_yaml(args.config)
    select_best_checkpoint(
        config=config,
        stage_output_dir=args.stage_output_dir,
        hard_input=args.hard_proxy_input,
        all_input=args.all_proxy_input,
        output_best_dir=args.output_best_dir,
        output_hard_eval=args.output_hard_eval,
        output_all_eval=args.output_all_eval,
        output_json=args.output_json,
        max_new_tokens=args.max_new_tokens,
        workdir=args.workdir,
        official_eval=(not args.no_official_eval),
        num_repeats=args.num_repeats,
    )


if __name__ == "__main__":
    main()
