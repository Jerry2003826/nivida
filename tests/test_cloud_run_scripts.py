from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _script(name: str) -> str:
    return (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_cloud_inference_scripts_accept_venv_directory_or_activate_path() -> None:
    for name in [
        "run_cloud_compute_only_v3.sh",
        "run_cloud_inference_only_v3.sh",
        "run_cloud_vllm_exact_eval_v3.sh",
        "check_cloud_vllm_env.sh",
    ]:
        text = _script(name)
        assert "resolve_activate_path" in text
        assert '[[ -d "$candidate" && -f "$candidate/bin/activate" ]]' in text
        assert 'VENV_ROOT="$(cd "$(dirname "$ACTIVATE_PATH")/.." && pwd)"' in text
        assert "Set VENV to either a venv directory or its bin/activate file." in text


def test_vllm_exact_eval_script_runs_preflight_and_official_proxy() -> None:
    text = _script("run_cloud_vllm_exact_eval_v3.sh")
    assert "scripts/check_cloud_vllm_env.sh" in text
    assert "scripts/check_cloud_eval_inputs.py" in text
    assert "run_eval_artifact_preflight" in text
    assert "materialize_builtin_eval_inputs" in text
    assert "head -n 6 \"$source\" > \"$target\"" in text
    assert "scripts/eval_official_vllm_proxy.py" in text
    assert "--write-raw-predictions" in text
    assert "--contract \"$CONTRACT\"" in text
    assert "add_candidate answer_final artifacts/adapter_stage2_official_balanced_answer_only" in text
    assert "add_candidate equation_rescue artifacts/adapter_stage2_equation_rescue" in text
    assert "INCLUDE_SUBMISSION_UNSAFE" in text
    assert "require_candidate" in text
    assert "ensure_adapter_config" in text
    assert "write_cloud_artifact_manifest.py" in text
    assert "artifact_manifest.json" in text


def test_vllm_raw_output_scorer_bridges_to_exact_ranking() -> None:
    text = _script("score_vllm_exact_eval_outputs.py")
    assert "scripts/evaluate_predictions_exact.py" in text
    assert "scripts/rank_exact_eval_reports.py" in text
    assert "scripts/rank_research_candidates.py" in text
    assert "--research-registry" in text
    assert "--prediction-key" in text
    assert "repeat_0.jsonl" in text


def test_score_vllm_exact_eval_outputs_smoke(tmp_path: Path) -> None:
    predictions_root = tmp_path / "vllm"
    label_path = tmp_path / "labels.jsonl"
    output_root = tmp_path / "scored"

    label_rows = [
        {"id": "a", "target_answer": "1", "official_family": "bit", "subtype": "toy"},
        {"id": "b", "target_answer": "2", "official_family": "bit", "subtype": "toy"},
    ]
    label_path.write_text(
        "\n".join(json.dumps(row) for row in label_rows) + "\n",
        encoding="utf-8",
    )

    for model, generations in {
        "b_thin": [r"\boxed{1}", r"\boxed{0}"],
        "answer_final": [r"\boxed{1}", r"\boxed{2}"],
    }.items():
        raw_dir = predictions_root / "smoke" / model / "raw"
        raw_dir.mkdir(parents=True)
        raw_dir.joinpath("repeat_0.jsonl").write_text(
            "\n".join(
                json.dumps({"id": row["id"], "generation": generation})
                for row, generation in zip(label_rows, generations)
            )
            + "\n",
            encoding="utf-8",
        )

    subprocess.run(
        [
            sys.executable,
            "scripts/score_vllm_exact_eval_outputs.py",
            "--predictions-root",
            str(predictions_root),
            "--output-root",
            str(output_root),
            "--label",
            f"smoke={label_path}",
            "--baseline",
            "b_thin",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    manifest = json.loads((output_root / "score_manifest.json").read_text(encoding="utf-8"))
    assert manifest["evals"]["smoke"]["submit_candidate"] == "answer_final"
    assert manifest["evals"]["smoke"]["rank_script"] == "scripts/rank_research_candidates.py"
    ranking = json.loads((output_root / "smoke" / "ranking.json").read_text(encoding="utf-8"))
    assert ranking["rows"][0]["model"] == "answer_final"


def test_score_vllm_exact_eval_outputs_auto_prefers_official_balanced(tmp_path: Path) -> None:
    predictions_root = tmp_path / "vllm"
    label_path = tmp_path / "labels.jsonl"
    output_root = tmp_path / "scored"

    label_rows = [
        {"id": "a", "target_answer": "1", "official_family": "logic", "subtype": "toy"},
        {"id": "b", "target_answer": "2", "official_family": "logic", "subtype": "toy"},
        {"id": "c", "target_answer": "3", "official_family": "logic", "subtype": "toy"},
    ]
    label_path.write_text(
        "\n".join(json.dumps(row) for row in label_rows) + "\n",
        encoding="utf-8",
    )

    for model, generations in {
        "b_thin": [r"\boxed{1}", r"\boxed{0}", r"\boxed{0}"],
        "official_balanced": [r"\boxed{1}", r"\boxed{2}", r"\boxed{0}"],
        "answer_final": [r"\boxed{1}", r"\boxed{2}", r"\boxed{3}"],
    }.items():
        raw_dir = predictions_root / "smoke" / model / "raw"
        raw_dir.mkdir(parents=True)
        raw_dir.joinpath("repeat_0.jsonl").write_text(
            "\n".join(
                json.dumps({"id": row["id"], "generation": generation})
                for row, generation in zip(label_rows, generations)
            )
            + "\n",
            encoding="utf-8",
        )

    subprocess.run(
        [
            sys.executable,
            "scripts/score_vllm_exact_eval_outputs.py",
            "--predictions-root",
            str(predictions_root),
            "--output-root",
            str(output_root),
            "--label",
            f"smoke={label_path}",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    manifest = json.loads((output_root / "score_manifest.json").read_text(encoding="utf-8"))
    assert manifest["evals"]["smoke"]["baseline"] == "official_balanced"
    assert manifest["evals"]["smoke"]["submit_candidate"] == "answer_final"
    ranking = json.loads((output_root / "smoke" / "ranking.json").read_text(encoding="utf-8"))
    assert ranking["baseline"] == "official_balanced"


def test_vllm_preflight_checks_torch_abi_requirement() -> None:
    text = _script("check_cloud_vllm_env.sh")
    assert 'md.metadata(name).get_all("Requires-Dist")' in text
    assert "vLLM torch ABI mismatch" in text
    assert "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu" in text


def test_cloud_inference_script_runs_artifact_preflight() -> None:
    text = _script("run_cloud_inference_only_v3.sh")
    assert "scripts/check_cloud_eval_inputs.py" in text
    assert "run_eval_artifact_preflight" in text
    assert "require_candidate" in text
    assert "ensure_adapter_config" in text
