"""Diff a local eval payload against the authoritative Kaggle runtime contract.

Purpose
-------
Every checkpoint we select must be scored under the *same* sampling / vLLM
configuration that the Kaggle leaderboard uses. The repo exposes two
contracts:

* ``RUNTIME_*`` in :mod:`src.competition.official_metric_contract` — the
  authoritative Kaggle Overview/Evaluation tab values (T=0.0, top_p=1.0,
  max_tokens=7680, max_model_len=8192, max_num_seqs=64). Confirmed by
  Kaggle Staff (Ryan Holbrook) on
  discussion #687798:
  https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687798

* ``OFFICIAL_*`` — byte-equivalent to the defaults written into the metric
  notebook's ``score()`` signature. Kaggle Staff has explicitly confirmed
  these defaults are NOT what the runner uses. Retained for fingerprint
  parity only.

The eval proxy (:mod:`scripts.eval_official_vllm_proxy`) emits
``eval_llm_kwargs`` / ``eval_sampling_kwargs`` / ``eval_contract`` fields into
its output JSON. This tool loads one of those payloads and compares the
actual kwargs used during scoring against the authoritative contract.

Exit codes
----------
* ``0`` — eval matches ``--contract`` (default: ``runtime``).
* ``2`` — eval diverges from the selected contract. Stderr carries a
  machine-readable diff; stdout carries the same diff as JSON when
  ``--json`` is set. Use this exit code in CI / pipeline gates to block
  selection based on a wrong-distribution eval.
* ``3`` — malformed or missing eval payload.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition.official_metric_contract import (  # noqa: E402
    OFFICIAL_LLM_KWARGS,
    OFFICIAL_SAMPLING_KWARGS,
    RUNTIME_CONTRACT_SOURCE,
    RUNTIME_LLM_KWARGS,
    RUNTIME_SAMPLING_KWARGS,
)


def _contract_kwargs(contract: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if contract == "runtime":
        return dict(RUNTIME_LLM_KWARGS), dict(RUNTIME_SAMPLING_KWARGS)
    if contract == "notebook":
        return dict(OFFICIAL_LLM_KWARGS), dict(OFFICIAL_SAMPLING_KWARGS)
    raise ValueError(
        f"Unknown contract {contract!r}; expected 'runtime' or 'notebook'"
    )


def _diff_mapping(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Return {key: {'actual': ..., 'expected': ...}} for mismatched keys.

    Keys missing on either side are also reported.
    """
    diffs: dict[str, Any] = {}
    all_keys = set(actual.keys()) | set(expected.keys())
    for key in sorted(all_keys):
        a = actual.get(key, "__missing__")
        e = expected.get(key, "__missing__")
        if a != e:
            diffs[key] = {"actual": a, "expected": e}
    return diffs


def verify_payload(
    payload: Mapping[str, Any], *, contract: str
) -> dict[str, Any]:
    """Compare a single eval payload against the target contract.

    Returns a report dict with keys:
      * ``contract`` — requested target contract.
      * ``payload_contract`` — the contract the payload says it was run under
        (or "__unknown__").
      * ``llm_kwargs_diff`` / ``sampling_kwargs_diff`` — mismatched keys.
      * ``ok`` — True iff both diffs are empty.
      * ``contract_mismatch`` — True when the payload declares a different
        contract than requested (still runs the kwargs diff for audit).
    """
    expected_llm, expected_sampling = _contract_kwargs(contract)
    actual_llm = payload.get("eval_llm_kwargs")
    actual_sampling = payload.get("eval_sampling_kwargs")
    if not isinstance(actual_llm, dict) or not isinstance(actual_sampling, dict):
        return {
            "contract": contract,
            "payload_contract": payload.get("eval_contract", "__unknown__"),
            "ok": False,
            "error": (
                "Payload is missing 'eval_llm_kwargs' or 'eval_sampling_kwargs'. "
                "Regenerate the eval with scripts/eval_official_vllm_proxy.py so "
                "the contract fields are recorded."
            ),
        }

    llm_diff = _diff_mapping(actual_llm, expected_llm)
    sampling_diff = _diff_mapping(actual_sampling, expected_sampling)
    payload_contract = payload.get("eval_contract", "__unknown__")
    contract_mismatch = (
        isinstance(payload_contract, str)
        and payload_contract not in ("__unknown__",)
        and payload_contract != contract
    )
    ok = (not llm_diff) and (not sampling_diff) and (not contract_mismatch)

    report: dict[str, Any] = {
        "contract": contract,
        "contract_source": (
            RUNTIME_CONTRACT_SOURCE if contract == "runtime" else "metric notebook defaults"
        ),
        "payload_contract": payload_contract,
        "contract_mismatch": contract_mismatch,
        "llm_kwargs_diff": llm_diff,
        "sampling_kwargs_diff": sampling_diff,
        "ok": ok,
    }
    # Surface a short human-readable summary for grep-friendly logs.
    if ok:
        report["summary"] = (
            f"[OK] eval payload matches contract={contract!r} "
            f"(adapter_dir={payload.get('adapter_dir', '?')})."
        )
    else:
        bullet_items: list[str] = []
        if contract_mismatch:
            bullet_items.append(
                f"payload declared contract {payload_contract!r} but caller "
                f"requested {contract!r}"
            )
        for key, info in sampling_diff.items():
            bullet_items.append(
                f"sampling.{key}: actual={info['actual']!r} "
                f"expected={info['expected']!r}"
            )
        for key, info in llm_diff.items():
            bullet_items.append(
                f"llm.{key}: actual={info['actual']!r} "
                f"expected={info['expected']!r}"
            )
        report["summary"] = "[DIFF] " + "; ".join(bullet_items)
    return report


def _load_payload(path: Path) -> Mapping[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"failed to load eval payload at {path}: {exc}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diff a local eval payload's sampling / vLLM kwargs against the "
            "authoritative Kaggle runtime contract. Exits non-zero on any "
            "divergence so pipelines can gate on it."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        required=True,
        action="append",
        help=(
            "Path to an eval JSON payload (typically from "
            "scripts.eval_official_vllm_proxy). May be passed multiple times "
            "to verify several payloads in one run."
        ),
    )
    parser.add_argument(
        "--contract",
        choices=["runtime", "notebook"],
        default="runtime",
        help=(
            "Target contract. 'runtime' (default) is the authoritative "
            "Kaggle Overview tab. 'notebook' is the legacy metric notebook "
            "defaults; only use for parity fingerprinting."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON to stdout instead of a summary.",
    )
    parser.add_argument(
        "--strict-missing-metadata",
        action="store_true",
        help=(
            "Treat payloads missing eval_llm_kwargs/eval_sampling_kwargs as a "
            "hard failure (exit 3). Default is to report and exit 3."
        ),
    )
    args = parser.parse_args(argv)

    reports: list[dict[str, Any]] = []
    any_missing = False
    any_diff = False
    for payload_path in args.eval_json:
        payload = _load_payload(payload_path)
        report = verify_payload(payload, contract=args.contract)
        report["eval_json"] = str(payload_path)
        if "error" in report:
            any_missing = True
        elif not report["ok"]:
            any_diff = True
        reports.append(report)

    if args.json:
        print(json.dumps(reports, indent=2, sort_keys=True))
    else:
        for report in reports:
            label = report.get("eval_json", "?")
            print(f"{label}: {report['summary']}")
            if "error" in report:
                print(f"  error: {report['error']}")

    if any_missing:
        return 3
    if any_diff:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
