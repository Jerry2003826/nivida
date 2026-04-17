from __future__ import annotations

from pathlib import Path

from src.common.io import write_jsonl
from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair
from src.experiments.run_teacher_benchmark import benchmark_examples
from src.teacher.global_rule_graph import GlobalRuleGraph
from src.teacher.hardcase_miner import mine_hard_cases
from src.teacher.synth_generator import generate_synthetic_examples


def test_global_rule_graph_roundtrip(tmp_path: Path) -> None:
    graph = GlobalRuleGraph.from_records(
        [
            {"family": "equation", "steps": ["binary_equation_rule"]},
            {"family": "equation", "steps": ["binary_equation_rule", "evaluate_expression"]},
            {"family": "bit", "steps": ["binary_rotate_left", "binary_xor_mask"]},
        ]
    )
    path = tmp_path / "graph.json"
    graph.save(path)
    loaded = GlobalRuleGraph.load(path)
    assert loaded.start_prior("binary_equation_rule", family="equation") > 0.0
    assert loaded.transition_prior("binary_rotate_left", "binary_xor_mask", family="bit") > 0.0
    assert loaded.step_position_weights["0"]["binary_equation_rule"] > 0.0


def test_synth_generation_respects_dedupe(tmp_path: Path) -> None:
    examples, _ = generate_synthetic_examples(
        num_samples=2,
        family_weights={"numeral": 1.0},
        max_chain_length=1,
        hard_negative_ratio=0.0,
        dedupe_against_real=None,
        seed=7,
    )
    assert examples
    real_path = tmp_path / "real.jsonl"
    write_jsonl(real_path, [examples[0].to_dict()])
    deduped_examples, summary = generate_synthetic_examples(
        num_samples=2,
        family_weights={"numeral": 1.0},
        max_chain_length=1,
        hard_negative_ratio=0.0,
        dedupe_against_real=str(real_path),
        seed=7,
    )
    assert deduped_examples
    assert summary["skipped_duplicates"] >= 1
    assert examples[0].query != deduped_examples[0].query or examples[0].target_answer != deduped_examples[0].target_answer


def test_hardcase_miner_prefers_equation_and_bit_failures() -> None:
    rows = [
        {"id": "easy", "official_family": "cipher", "competition_correct": True, "numeric": True, "boxed_valid": True, "teacher_confidence": 0.9, "steps": []},
        {"id": "eq_fail", "official_family": "equation", "competition_correct": False, "numeric": False, "boxed_valid": False, "teacher_confidence": 0.8, "steps": ["operator_template"]},
        {"id": "bit_fail", "official_family": "bit", "competition_correct": False, "numeric": False, "boxed_valid": False, "teacher_confidence": 0.7, "steps": ["binary_affine_transform"]},
    ]
    hard_cases = mine_hard_cases({"records": rows}, max_items=2)
    assert [row["id"] for row in hard_cases] == ["eq_fail", "bit_fail"]


def test_teacher_benchmark_schema_and_filters() -> None:
    examples = [
        PuzzleExample(
            id="eq_ok",
            raw_prompt="",
            official_instruction="",
            parsed_examples=[PuzzlePair(input="96$54", output="5184"), PuzzlePair(input="50$41", output="2050"), PuzzlePair(input="51$95", output="4845")],
            query="59$49",
            target_answer="2891",
            metadata=PuzzleMetadata(official_family="equation", subtype="numeric", source="official"),
        ),
        PuzzleExample(
            id="bit_ok",
            raw_prompt="",
            official_instruction="",
            parsed_examples=[PuzzlePair(input="10100101", output="01010101"), PuzzlePair(input="00001111", output="11111111")],
            query="11110000",
            target_answer="00000000",
            metadata=PuzzleMetadata(official_family="bit", subtype="mask_logic", source="official"),
        ),
    ]
    payload = benchmark_examples(
        examples,
        beam_width=8,
        max_depth=1,
        top_k=3,
        max_per_family=10,
        family_filter={"equation"},
        failures_only=False,
    )
    assert payload["num_examples"] == 1
    assert payload["records"][0]["official_family"] == "equation"
    assert {"official_family", "subtype", "prediction", "target", "exact", "numeric", "boxed_valid", "teacher_confidence", "steps", "failure_type"} <= set(payload["records"][0])

    failures = benchmark_examples(
        examples,
        beam_width=8,
        max_depth=1,
        top_k=3,
        max_per_family=10,
        family_filter=None,
        failures_only=True,
    )
    assert failures["records"] == []
