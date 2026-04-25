from __future__ import annotations

from scripts.audit_solver_coverage import _audit_example, _group_summary


def _candidate(query_prediction: str):
    step = type("Step", (), {"op_name": "identity", "params": {}})()
    return type(
        "Candidate",
        (),
        {
            "steps": [step],
            "predictions": ["b"],
            "query_prediction": query_prediction,
            "score": 1.0,
            "confidence": 0.9,
            "exact_ratio": 1.0,
        },
    )()


class _Engine:
    def solve_example(self, example, *, top_k: int):
        assert top_k == 3
        return [_candidate("wrong"), _candidate("gold")]


def test_audit_example_records_support_full_oracle_rank() -> None:
    row = {
        "id": "oracle",
        "raw_prompt": "",
        "official_instruction": "",
        "parsed_examples": [{"input": "a", "output": "b"}],
        "query": "q",
        "target_answer": "gold",
        "metadata": {"official_family": "equation", "subtype": "equation_template"},
    }

    record = _audit_example(_Engine(), row, top_k=3, retag=False)

    assert record["query_correct"] is False
    assert record["oracle_at_k"] is True
    assert record["oracle_rank"] == 2
    assert record["support_full_candidate_count"] == 2


def test_group_summary_reports_oracle_rate() -> None:
    rows = [
        {"query_correct": False, "oracle_at_k": True, "support_accuracy": 1.0, "failure_class": "miss"},
        {"query_correct": False, "oracle_at_k": False, "support_accuracy": 1.0, "failure_class": "miss"},
    ]

    summary = _group_summary(rows, "failure_class")

    assert summary["miss"]["oracle_at_k"] == 0.5
