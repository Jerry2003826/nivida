from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _make_annotated_row(
    *,
    id: str,
    family: str,
    answer_type: str,
    teacher_conf: float,
    support_pairs: list[tuple[str, str, str]],
    query_pred: str,
    target: str,
    program_sig: str | None = "sig-default",
    synth_source: str = "chain_search",
    include_cached_support_pairs: bool = True,
    repo_support_coverage: float = 1.0,
    repo_solver_verifiable: bool = True,
) -> dict:
    extras = {
        "declared_answer_type": answer_type,
        "support_coverage": repo_support_coverage,
        "solver_verifiable": repo_solver_verifiable,
    }
    if include_cached_support_pairs:
        extras["support_pairs"] = [
            {"input": support[0], "target": support[1], "prediction": support[2]}
            for support in support_pairs
        ]
        extras["query_prediction"] = query_pred

    return {
        "id": id,
        "raw_prompt": f"prompt for {id}",
        "query": f"query for {id}",
        "parsed_examples": [
            {"input": support[0], "output": support[1]}
            for support in support_pairs
        ],
        "family": family,
        "official_family": family,
        "target_answer": target,
        "metadata": {
            "official_family": family,
            "teacher_confidence": teacher_conf,
            "program_signature": program_sig,
            "source": synth_source,
            "extras": extras,
        },
    }


def test_binary_float_bug_is_caught(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    _write_jsonl(
        train,
        [
            _make_annotated_row(
                id="r1",
                family="bit",
                answer_type="answer_integer",
                teacher_conf=0.9,
                support_pairs=[("a", "101", "101.0"), ("b", "110", "110.0")],
                query_pred="101.0",
                target="101",
            ),
        ],
    )
    synth = tmp_path / "synth.jsonl"
    synth.write_text("", encoding="utf-8")
    out = tmp_path / "audit.json"
    subprocess.check_call(
        [
            "python",
            str(REPO / "scripts/audit_teacher_gate_extractor_parity.py"),
            "--train-jsonl",
            str(train),
            "--synth-jsonl",
            str(synth),
            "--output",
            str(out),
            "--samples-to-include",
            "5",
        ]
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["strict_repo_accepted"] == 1
    assert data["summary"]["strict_official_accepted"] == 0
    assert data["summary"]["strict_pollution_rate"] == 1.0
    assert data["decision_hint"]["action"] == "rebuild_stage2"
    assert any(example["id"] == "r1" for example in data["examples"]["strict_repo_only"])


def test_numeric_tolerance_ok_both_sides(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    _write_jsonl(
        train,
        [
            _make_annotated_row(
                id="r2",
                family="gravity",
                answer_type="answer_decimal",
                teacher_conf=0.9,
                support_pairs=[("a", "24.64", "24.6401")],
                query_pred="24.6401",
                target="24.64",
            ),
        ],
    )
    synth = tmp_path / "synth.jsonl"
    synth.write_text("", encoding="utf-8")
    out = tmp_path / "audit.json"
    subprocess.check_call(
        [
            "python",
            str(REPO / "scripts/audit_teacher_gate_extractor_parity.py"),
            "--train-jsonl",
            str(train),
            "--synth-jsonl",
            str(synth),
            "--output",
            str(out),
        ]
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["strict_repo_accepted"] == 1
    assert data["summary"]["strict_official_accepted"] == 1
    assert data["summary"]["strict_pollution_rate"] == 0.0


def test_parameter_order_used_correctly(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    _write_jsonl(
        train,
        [
            _make_annotated_row(
                id="r3",
                family="bit",
                answer_type="answer_integer",
                teacher_conf=0.9,
                support_pairs=[("a", "11011", "00011011")],
                query_pred="00011011",
                target="11011",
            ),
        ],
    )
    synth = tmp_path / "synth.jsonl"
    synth.write_text("", encoding="utf-8")
    out = tmp_path / "audit.json"
    subprocess.check_call(
        [
            "python",
            str(REPO / "scripts/audit_teacher_gate_extractor_parity.py"),
            "--train-jsonl",
            str(train),
            "--synth-jsonl",
            str(synth),
            "--output",
            str(out),
        ]
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["strict_pollution_rate"] == 1.0


def test_missing_program_signature_rejected_by_repo_strict_gate(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    _write_jsonl(
        train,
        [
            _make_annotated_row(
                id="r4",
                family="bit",
                answer_type="answer_integer",
                teacher_conf=0.95,
                support_pairs=[("a", "101", "101")],
                query_pred="101",
                target="101",
                program_sig=None,
                repo_support_coverage=1.0,
                repo_solver_verifiable=True,
            ),
        ],
    )
    synth = tmp_path / "synth.jsonl"
    synth.write_text("", encoding="utf-8")
    out = tmp_path / "audit.json"
    subprocess.check_call(
        [
            "python",
            str(REPO / "scripts/audit_teacher_gate_extractor_parity.py"),
            "--train-jsonl",
            str(train),
            "--synth-jsonl",
            str(synth),
            "--output",
            str(out),
        ]
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["strict_repo_accepted"] == 0
    assert data["summary"]["strict_pollution_rate"] is None
    assert data["decision_hint"]["action"] == "insufficient_evidence"


def test_missing_provenance_returns_insufficient_evidence(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    _write_jsonl(
        train,
        [
            _make_annotated_row(
                id="r5",
                family="equation",
                answer_type="answer_integer",
                teacher_conf=0.9,
                support_pairs=[("1+1", "2", "2")],
                query_pred="2",
                target="2",
                program_sig="sig-r5",
                include_cached_support_pairs=False,
                repo_support_coverage=1.0,
                repo_solver_verifiable=True,
            ),
        ],
    )
    synth = tmp_path / "synth.jsonl"
    synth.write_text("", encoding="utf-8")
    out = tmp_path / "audit.json"
    subprocess.check_call(
        [
            "python",
            str(REPO / "scripts/audit_teacher_gate_extractor_parity.py"),
            "--train-jsonl",
            str(train),
            "--synth-jsonl",
            str(synth),
            "--output",
            str(out),
            "--allow-rerun-chain-search",
        ]
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "insufficient_evidence"
    assert data["reason"] == "stage2 provenance missing or mismatched"
