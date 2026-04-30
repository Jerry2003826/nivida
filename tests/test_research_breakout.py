from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.common.io import read_json, read_yaml, write_json, write_jsonl
from src.research.artifact_manifest import build_cloud_artifact_manifest
from src.research.candidate_registry import (
    DEFAULT_BASELINE_NAME,
    build_default_registry,
    registry_with_updated_candidate,
    validate_registry,
)
from src.research.lb_correlation import append_correlation_entry
from src.research.prompt_profiles import materialize_prompt_profile_row
from src.research.solver_finalizer import apply_solver_assisted_finalizer
from src.research.weak_family_data import build_research_rescue_data
from scripts.rank_research_candidates import _enrich_rows


REPO_ROOT = Path(__file__).resolve().parents[1]


def _report(path: Path, *, acc: float, boxed: float = 1.0, family: str = "bit", n: int = 4) -> Path:
    write_json(
        path,
        {
            "overall": {
                "official_verify_accuracy": acc,
                "local_competition_accuracy": acc,
                "boxed_valid_rate": boxed,
                "avg_prediction_words": 3.0,
            },
            "family": {
                family: {
                    "n": n,
                    "official_verify_accuracy": acc,
                    "local_competition_accuracy": acc,
                    "local_exact_accuracy": acc,
                    "boxed_valid_rate": boxed,
                }
            },
            "subtype_official_verify_accuracy": {f"{family}:toy": {"n": n, "accuracy": acc}},
        },
    )
    return path


def test_default_candidate_registry_is_official_balanced_gated() -> None:
    registry = build_default_registry()

    assert validate_registry(registry) == []
    assert registry["baseline"] == DEFAULT_BASELINE_NAME
    assert registry["baseline_public_score"] == 0.57
    names = {candidate["name"] for candidate in registry["candidates"]}
    assert {
        "answer_only_continuation",
        "short_trace_continuation",
        "mixed_answer_short",
        "equation_rescue",
        "bit_rescue",
        "eq_bit_rescue",
        "equation_rescue_v2",
        "bit_rescue_v2",
        "eq_bit_rescue_v2",
        "rank64_answer_only",
        "final_answer_weighted_loss",
        "soup_answer_short",
        "soup_eq_bit",
        "soup_all_rescue",
        "soup_official_answer_rescue",
    } <= names
    rank64 = next(candidate for candidate in registry["candidates"] if candidate["name"] == "rank64_answer_only")
    assert rank64["submission_safe"] is False
    rescue_v2 = next(candidate for candidate in registry["candidates"] if candidate["name"] == "eq_bit_rescue_v2")
    assert rescue_v2["submission_safe"] is False
    assert "solver_breakout_v2" in rescue_v2["artifacts"]
    solver = next(candidate for candidate in registry["candidates"] if candidate["name"] == "official_balanced_solver_assisted")
    prompt = next(candidate for candidate in registry["candidates"] if candidate["name"] == "official_balanced_prompt_ensemble")
    assert solver["submission_safe"] is False
    assert prompt["submission_safe"] is False
    assert "adapter-only" in solver["research_only_reason"]


def test_v2_rescue_registry_configs_point_to_v2_data() -> None:
    registry = build_default_registry()
    by_name = {candidate["name"]: candidate for candidate in registry["candidates"]}

    for name in ("equation_rescue_v2", "bit_rescue_v2", "eq_bit_rescue_v2"):
        candidate = by_name[name]
        config_path = REPO_ROOT / candidate["config_path"]
        config = read_yaml(config_path)

        assert config["training"]["output_dir"] == candidate["adapter_path"]
        assert config["training"]["dataset_path"].endswith(f"{candidate['data_recipe']}_train.jsonl")
        assert config["training"]["eval_path"].endswith(f"{candidate['data_recipe']}_valid.jsonl")
        assert config["training"]["final_answer_loss"]["enabled"] is True


def test_registry_rejects_non_adapter_only_submit_safe_candidates() -> None:
    registry = registry_with_updated_candidate(
        build_default_registry(),
        "official_balanced_solver_assisted",
        {"submission_safe": True},
    )

    errors = validate_registry(registry)

    assert any("solver_assisted must be submission_safe=false" in error for error in errors)


def test_research_ranking_auto_uses_official_balanced_and_blocks_unsafe(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    write_json(registry_path, build_default_registry())
    baseline = _report(tmp_path / "official.json", acc=0.57)
    unsafe = _report(tmp_path / "rank64.json", acc=0.75)
    winner = _report(tmp_path / "winner.json", acc=0.60)
    output_json = tmp_path / "ranking.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/rank_research_candidates.py",
            "--registry",
            str(registry_path),
            "--report",
            f"official_balanced={baseline}",
            "--report",
            f"rank64_answer_only={unsafe}",
            "--report",
            f"answer_only_continuation={winner}",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(tmp_path / "ranking.csv"),
            "--output-md",
            str(tmp_path / "ranking.md"),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    payload = read_json(output_json)
    assert payload["baseline"] == "official_balanced"
    assert payload["rows"][0]["model"] == "rank64_answer_only"
    assert payload["rows"][0]["pass_gate"] is False
    assert payload["rows"][0]["gate_reason"] == "candidate is marked submission_unsafe"
    assert next(row for row in payload["rows"] if row["submit_candidate"])["model"] == "answer_only_continuation"


def test_research_ranking_blocks_solver_assisted_even_if_registry_is_wrong(tmp_path: Path) -> None:
    registry = registry_with_updated_candidate(
        build_default_registry(),
        "official_balanced_solver_assisted",
        {"submission_safe": True, "research_only_reason": "wrong on purpose"},
    )
    rows = _enrich_rows(
        [
            {
                "rank": 1,
                "model": "official_balanced_solver_assisted",
                "pass_gate": True,
                "gate_reason": "pass",
                "official_verify_accuracy": 0.8,
                "delta_vs_baseline": 0.2,
                "boxed_valid_rate": 1.0,
            },
            {
                "rank": 2,
                "model": "soup_answer_short",
                "pass_gate": True,
                "gate_reason": "pass",
                "official_verify_accuracy": 0.6,
                "delta_vs_baseline": 0.03,
                "boxed_valid_rate": 1.0,
            },
        ],
        registry,
    )

    solver_row = next(row for row in rows if row["model"] == "official_balanced_solver_assisted")
    assert solver_row["submission_class"] == "research_only"
    assert solver_row["pass_gate"] is False
    assert solver_row["gate_reason"] == "candidate type is research-only: solver_assisted"
    assert next(row for row in rows if row["submit_candidate"])["model"] == "soup_answer_short"


def test_prompt_profile_materializes_suffix_and_metadata() -> None:
    row = {"id": "x", "prompt": "Solve this", "target_answer": "1"}

    profiled = materialize_prompt_profile_row(row, "format_strict")

    assert "Return exactly one final answer" in profiled["prompt"]
    assert profiled["prompt_profile"] == "format_strict"
    assert profiled["metadata"]["prompt_profile"] == "format_strict"


def test_solver_assisted_finalizer_overrides_only_high_confidence_family(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "finalized.jsonl"
    report = tmp_path / "report.json"
    label_rows = [
        {
            "id": "bit",
            "raw_prompt": "toy",
            "official_instruction": "toy",
            "parsed_examples": [
                {"input": "0000", "output": "1111"},
                {"input": "1010", "output": "0101"},
            ],
            "query": "0011",
            "target_answer": "1100",
            "metadata": {"official_family": "bit", "subtype": "bit_xor_mask"},
        },
        {
            "id": "other",
            "raw_prompt": "toy",
            "official_instruction": "toy",
            "parsed_examples": [{"input": "a", "output": "b"}],
            "query": "c",
            "target_answer": "d",
            "metadata": {"official_family": "cipher", "subtype": "toy"},
        },
    ]
    write_jsonl(labels, label_rows)
    write_jsonl(
        predictions,
        [
            {"id": "bit", "generation": r"\boxed{0000}"},
            {"id": "other", "generation": r"\boxed{x}"},
        ],
    )

    summary = apply_solver_assisted_finalizer(
        predictions_path=predictions,
        labels_path=labels,
        output_path=output,
        report_path=report,
        min_confidence=0.0,
        min_support_coverage=1.0,
        beam_width=8,
        max_depth=2,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary["num_overrides"] == 1
    assert rows[0]["generation"] == r"\boxed{1100}"
    assert rows[0]["raw_generation"] == r"\boxed{0000}"
    assert rows[1]["generation"] == r"\boxed{x}"


def test_research_rescue_data_filters_weak_families_and_safe_short_trace(tmp_path: Path) -> None:
    answer_train = tmp_path / "answer_train.jsonl"
    answer_valid = tmp_path / "answer_valid.jsonl"
    short_train = tmp_path / "short_train.jsonl"
    short_valid = tmp_path / "short_valid.jsonl"
    answer_rows = [
        {"id": "eq", "trace_style": "answer_only", "official_family": "equation", "metadata": {"official_family": "equation", "extras": {"template_risk_class": "unseen_key_template_miss"}}},
        {"id": "bit", "trace_style": "answer_only", "official_family": "bit", "metadata": {"official_family": "bit", "extras": {}}},
        {"id": "cipher", "trace_style": "answer_only", "official_family": "cipher", "metadata": {"official_family": "cipher", "extras": {}}},
    ]
    short_rows = [
        {"id": "eq", "trace_style": "short_trace", "official_family": "equation", "subtype": "equation_template", "metadata": {"official_family": "equation", "extras": {"template_risk_class": "unseen_key_template_miss"}}},
        {"id": "bit", "trace_style": "short_trace", "official_family": "bit", "metadata": {"official_family": "bit", "extras": {"solver_verifiable": True, "support_coverage": 1.0}}},
    ]
    for path in (answer_train, answer_valid):
        write_jsonl(path, answer_rows)
    for path in (short_train, short_valid):
        write_jsonl(path, short_rows)

    summary = build_research_rescue_data(
        answer_train=answer_train,
        answer_valid=answer_valid,
        short_train=short_train,
        short_valid=short_valid,
        out_dir=tmp_path / "out",
        recipes=["equation_rescue", "bit_rescue", "eq_bit_rescue", "eq_bit_rescue_v2"],
    )

    assert summary["recipes"]["equation_rescue"]["train_rows"] == 1
    assert summary["recipes"]["bit_rescue"]["train_rows"] == 2
    assert "train_provenance" in summary["recipes"]["bit_rescue"]
    assert Path(summary["recipes"]["bit_rescue"]["train_provenance"]).is_file()
    eq_bit_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "eq_bit_rescue_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["id"] for row in eq_bit_rows} == {"eq", "bit"}
    assert all(row["research_recipe"] == "eq_bit_rescue" for row in eq_bit_rows)
    assert all(row["research_provenance"]["answer_hash"] for row in eq_bit_rows)
    bit_row = next(row for row in eq_bit_rows if row["id"] == "bit")
    assert bit_row["metadata"]["extras"]["bit_operator_family"] == "unknown"
    eq_bit_v2_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "eq_bit_rescue_v2_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["id"] for row in eq_bit_v2_rows} == {"eq", "bit"}
    assert all(row["research_recipe"] == "eq_bit_rescue_v2" for row in eq_bit_v2_rows)


def test_cloud_artifact_manifest_counts_prediction_lines_and_hashes_candidate(tmp_path: Path) -> None:
    adapter = tmp_path / "artifacts" / "adapter"
    adapter.mkdir(parents=True)
    adapter.joinpath("adapter_model.safetensors").write_bytes(b"weights")
    raw_dir = tmp_path / "out" / "smoke" / "candidate" / "raw"
    raw_dir.mkdir(parents=True)
    raw_dir.joinpath("repeat_0.jsonl").write_text('{"id": "a"}\n{"id": "b"}\n', encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    write_json(preflight, {"status": "pass"})

    manifest = build_cloud_artifact_manifest(
        repo_root=tmp_path,
        out_dir=tmp_path / "out",
        eval_inputs=["smoke"],
        candidates=["candidate=artifacts/adapter"],
        preflight_path=preflight,
    )

    assert manifest["preflight_status"] == "pass"
    assert manifest["candidates"][0]["weights"][0]["sha256"]
    assert manifest["prediction_line_counts"] == {"smoke": {"candidate": 2}}


def test_lb_correlation_log_records_public_and_local_metrics(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    adapter.joinpath("adapter_config.json").write_text("{}", encoding="utf-8")
    adapter.joinpath("adapter_model.safetensors").write_bytes(b"weights")
    report = _report(tmp_path / "report.json", acc=0.61, boxed=0.9)

    payload = append_correlation_entry(
        log_path=tmp_path / "lb_log.json",
        candidate="soup_answer_short",
        public_score=0.58,
        exact_report=report,
        adapter_path=adapter,
        training_recipe="adapter_soup",
        merge_weights={"answer_only_continuation": 0.5, "short_trace_continuation": 0.5},
    )

    entry = payload["entries"][0]
    assert entry["candidate"] == "soup_answer_short"
    assert entry["public_score"] == 0.58
    assert entry["adapter_hashes"]["adapter_model.safetensors"]
    assert entry["exact_report"]["official_verify_accuracy"] == 0.61
