"""Tests for ``scripts.select_best_proxy_checkpoint``.

``run_inference`` and ``evaluate_replica`` are patched out so the tests don't
need a real model. Fake proxy scores are injected per-candidate to exercise
the selection rule and the checkpoint_config backfill branch.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.select_best_proxy_checkpoint as sbpc
from src.common.io import read_json


# --- helpers ------------------------------------------------------------------


def _make_valid_adapter_dir(path: Path, *, marker: str = "weights") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_model.safetensors").write_text(marker, encoding="utf-8")
    (path / "adapter_config.json").write_text('{"r": 32}', encoding="utf-8")
    return path


def _make_weights_only_dir(path: Path, *, marker: str = "weights") -> Path:
    """Simulate a checkpoint that has model weights but no adapter_config."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_model.safetensors").write_text(marker, encoding="utf-8")
    return path


def _make_empty_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _install_fake_inference_and_replica(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate_scores: dict[str, dict[str, tuple[float, int]]],
) -> None:
    """Replace run_inference / evaluate_replica with deterministic fakes.

    ``candidate_scores`` maps candidate name -> {'hard': (rate, n), 'all': (rate, n)}.
    We recover the candidate name from ``output_path``'s filename stem, which
    the real script always constructs as ``<candidate>_<proxy>_pred.jsonl``.
    """

    def _candidate_from_pred_name(path: Path) -> tuple[str, str]:
        stem = path.name
        for suffix, proxy in (
            ("_hard_pred.jsonl", "hard"),
            ("_all_pred.jsonl", "all"),
        ):
            if stem.endswith(suffix):
                return stem[: -len(suffix)], proxy
        raise ValueError(f"unexpected prediction filename: {path}")

    def _fake_run_inference(config, *, input_path, adapter_dir, output_path, max_new_tokens, **_kwargs):
        # Accept arbitrary extra kwargs (official_eval, runtime_eval, ...) so
        # the fake stays forward-compatible with the real run_inference
        # signature.
        candidate_name, _ = _candidate_from_pred_name(Path(output_path))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps({"candidate": candidate_name}), encoding="utf-8"
        )
        return Path(output_path)

    def _fake_evaluate_replica(*, prediction_path, label_path, require_complete_coverage=False):
        payload = json.loads(Path(prediction_path).read_text(encoding="utf-8"))
        candidate_name = payload["candidate"]
        _, proxy = _candidate_from_pred_name(Path(prediction_path))
        rate, n = candidate_scores[candidate_name][proxy]
        return {
            "competition_correct_rate": rate,
            "num_examples": n,
            "coverage": {"num_missing": 0, "num_unexpected": 0, "num_duplicate": 0},
        }

    monkeypatch.setattr(sbpc, "run_inference", _fake_run_inference)
    monkeypatch.setattr(sbpc, "evaluate_replica", _fake_evaluate_replica)


def _run_selector(
    *,
    tmp_path: Path,
    stage_dir: Path,
    candidate_scores: dict[str, dict[str, tuple[float, int]]],
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    _install_fake_inference_and_replica(monkeypatch, candidate_scores=candidate_scores)
    hard_input = tmp_path / "hard_labels.jsonl"
    all_input = tmp_path / "all_labels.jsonl"
    hard_input.write_text('{"id":"x"}\n', encoding="utf-8")
    all_input.write_text('{"id":"y"}\n', encoding="utf-8")

    output_best_dir = tmp_path / "best"
    output_hard_eval = tmp_path / "best_hard_eval.json"
    output_all_eval = tmp_path / "best_all_eval.json"
    output_json = tmp_path / "selection.json"
    workdir = tmp_path / "work"

    sbpc.select_best_checkpoint(
        config={"base_model": "x", "model_source": "huggingface"},
        stage_output_dir=stage_dir,
        hard_input=hard_input,
        all_input=all_input,
        output_best_dir=output_best_dir,
        output_hard_eval=output_hard_eval,
        output_all_eval=output_all_eval,
        output_json=output_json,
        max_new_tokens=2048,
        workdir=workdir,
    )
    return read_json(output_json)


# --- tests --------------------------------------------------------------------


def test_selector_picks_final_when_all_family_monotonic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_dir = tmp_path / "stage"
    _make_valid_adapter_dir(stage_dir / "checkpoint-500")
    _make_valid_adapter_dir(stage_dir / "checkpoint-1000")
    _make_valid_adapter_dir(stage_dir)  # final

    scores = {
        "checkpoint-500": {"hard": (0.30, 300), "all": (0.40, 400)},
        "checkpoint-1000": {"hard": (0.35, 300), "all": (0.48, 400)},
        "final": {"hard": (0.36, 300), "all": (0.55, 400)},
    }
    summary = _run_selector(
        tmp_path=tmp_path, stage_dir=stage_dir, candidate_scores=scores, monkeypatch=monkeypatch
    )
    assert summary["selected_candidate"] == "final"


def test_selector_picks_intermediate_checkpoint_when_final_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_dir = tmp_path / "stage"
    _make_valid_adapter_dir(stage_dir / "checkpoint-500")
    _make_valid_adapter_dir(stage_dir / "checkpoint-1500")
    _make_valid_adapter_dir(stage_dir)

    scores = {
        "checkpoint-500": {"hard": (0.30, 300), "all": (0.40, 400)},
        "checkpoint-1500": {"hard": (0.38, 300), "all": (0.58, 400)},  # peak
        "final": {"hard": (0.36, 300), "all": (0.52, 400)},
    }
    summary = _run_selector(
        tmp_path=tmp_path, stage_dir=stage_dir, candidate_scores=scores, monkeypatch=monkeypatch
    )
    assert summary["selected_candidate"] == "checkpoint-1500"


def test_selector_backfills_missing_adapter_config_and_keeps_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_dir = tmp_path / "stage"
    _make_valid_adapter_dir(stage_dir)  # final (reference)
    _make_weights_only_dir(stage_dir / "checkpoint-700")  # no adapter_config.json

    scores = {
        "checkpoint-700": {"hard": (0.40, 300), "all": (0.60, 400)},  # peak
        "final": {"hard": (0.36, 300), "all": (0.52, 400)},
    }
    summary = _run_selector(
        tmp_path=tmp_path, stage_dir=stage_dir, candidate_scores=scores, monkeypatch=monkeypatch
    )
    # Backfill should let the candidate be scored and it is the best of the two.
    ckpt_record = next(c for c in summary["candidates"] if c["name"] == "checkpoint-700")
    assert ckpt_record["adapter_config_fallback_applied"] is True
    assert ckpt_record["skipped"] is False
    assert summary["selected_candidate"] == "checkpoint-700"


def test_selector_skips_candidate_without_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_dir = tmp_path / "stage"
    _make_valid_adapter_dir(stage_dir)  # final
    _make_empty_dir(stage_dir / "checkpoint-200")  # no weights at all

    scores = {
        "final": {"hard": (0.36, 300), "all": (0.52, 400)},
    }
    summary = _run_selector(
        tmp_path=tmp_path, stage_dir=stage_dir, candidate_scores=scores, monkeypatch=monkeypatch
    )
    broken = next(c for c in summary["candidates"] if c["name"] == "checkpoint-200")
    assert broken["skipped"] is True
    assert broken["skip_reason"]
    assert summary["selected_candidate"] == "final"


def test_selector_prefers_final_on_complete_tie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_dir = tmp_path / "stage"
    _make_valid_adapter_dir(stage_dir / "checkpoint-500")
    _make_valid_adapter_dir(stage_dir)

    scores = {
        "checkpoint-500": {"hard": (0.40, 300), "all": (0.52, 400)},
        "final": {"hard": (0.40, 300), "all": (0.52, 400)},
    }
    summary = _run_selector(
        tmp_path=tmp_path, stage_dir=stage_dir, candidate_scores=scores, monkeypatch=monkeypatch
    )
    assert summary["selected_candidate"] == "final"


def test_selector_materialises_best_adapter_and_evals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_dir = tmp_path / "stage"
    _make_valid_adapter_dir(stage_dir / "checkpoint-1000", marker="from-1000")
    _make_valid_adapter_dir(stage_dir, marker="from-final")

    scores = {
        "checkpoint-1000": {"hard": (0.40, 300), "all": (0.60, 400)},  # winner
        "final": {"hard": (0.30, 300), "all": (0.50, 400)},
    }
    _run_selector(
        tmp_path=tmp_path, stage_dir=stage_dir, candidate_scores=scores, monkeypatch=monkeypatch
    )
    assert (tmp_path / "best" / "adapter_model.safetensors").read_text(encoding="utf-8") == "from-1000"
    assert (tmp_path / "best_hard_eval.json").exists()
    assert (tmp_path / "best_all_eval.json").exists()


def test_discover_candidate_dirs_orders_checkpoints_then_final(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage"
    _make_valid_adapter_dir(stage_dir)
    _make_valid_adapter_dir(stage_dir / "checkpoint-1500")
    _make_valid_adapter_dir(stage_dir / "checkpoint-200")
    _make_valid_adapter_dir(stage_dir / "checkpoint-800")
    # Non-checkpoint siblings should be ignored by the discovery step.
    (stage_dir / "not_a_checkpoint").mkdir()

    candidates = sbpc.discover_candidate_dirs(stage_dir)
    names = [name for name, _ in candidates]
    assert names == ["checkpoint-200", "checkpoint-800", "checkpoint-1500", "final"]
