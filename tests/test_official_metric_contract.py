from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from src.competition.official_metric_contract import (
    EXTRACT_FINAL_ANSWER_SHA256,
    EXPECTED_EXTRACT_FINAL_ANSWER_SHA256,
    EXPECTED_VERIFY_SHA256,
    METRIC_NOTEBOOK_RELATIVE_PATH,
    OFFICIAL_GUARD_TEXT,
    OFFICIAL_LLM_KWARGS,
    OFFICIAL_SAMPLING_KWARGS,
    RUNTIME_CONTRACT_SOURCE,
    RUNTIME_LLM_KWARGS,
    RUNTIME_SAMPLING_KWARGS,
    VERIFY_SHA256,
    current_contract_fingerprint,
    extract_final_answer,
    verify,
)


def test_sampling_kwargs_exact() -> None:
    assert OFFICIAL_SAMPLING_KWARGS == {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 3584,
    }


def test_llm_kwargs_exact() -> None:
    assert OFFICIAL_LLM_KWARGS == {
        "tensor_parallel_size": 1,
        "max_num_seqs": 128,
        "gpu_memory_utilization": 0.85,
        "dtype": "auto",
        "max_model_len": 4096,
        "trust_remote_code": True,
        "enable_lora": True,
        "max_lora_rank": 32,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
    }


def test_runtime_sampling_kwargs_exact() -> None:
    # Authoritative Kaggle Overview/Evaluation tab, confirmed by
    # Ryan Holbrook on discussion #687798.
    assert RUNTIME_SAMPLING_KWARGS == {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 7680,
    }


def test_runtime_llm_kwargs_exact() -> None:
    assert RUNTIME_LLM_KWARGS == {
        "tensor_parallel_size": 1,
        "max_num_seqs": 64,
        "gpu_memory_utilization": 0.85,
        "dtype": "auto",
        "max_model_len": 8192,
        "trust_remote_code": True,
        "enable_lora": True,
        "max_lora_rank": 32,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
    }


def test_runtime_contract_source_non_empty() -> None:
    assert "Kaggle" in RUNTIME_CONTRACT_SOURCE
    assert "687798" in RUNTIME_CONTRACT_SOURCE


def test_runtime_and_notebook_contracts_diverge() -> None:
    # Guard rail: if these ever become equal again, the runtime contract
    # must be re-verified against Kaggle Overview before selection is
    # trusted.
    assert RUNTIME_SAMPLING_KWARGS != OFFICIAL_SAMPLING_KWARGS
    assert RUNTIME_LLM_KWARGS != OFFICIAL_LLM_KWARGS


def test_guard_text_exact() -> None:
    assert OFFICIAL_GUARD_TEXT == (
        "\nPlease put your final answer inside `\\boxed{}`."
        " For example: `\\boxed{your answer}`"
    )


def test_extract_boxed_basic() -> None:
    assert extract_final_answer(r"The answer is \boxed{42}") == "42"


def test_extract_none() -> None:
    assert extract_final_answer(None) == "NOT_FOUND"


def test_extract_fallback_final_answer_phrase() -> None:
    assert extract_final_answer("The final answer is: 3.14") == "3.14"


def test_extract_fullwidth_colon_pattern() -> None:
    assert extract_final_answer("Final answer：42") == "42"
    assert extract_final_answer("final answer：3.14") == "3.14"


def test_fullwidth_colon_non_numeric_guard() -> None:
    assert extract_final_answer("Final answer：XLVII") == "XLVII"


def test_extract_fallback_last_number() -> None:
    assert extract_final_answer("Just a number 100 in text") == "100"


def test_extract_last_nonempty_boxed_wins() -> None:
    text = r"\boxed{1} and later \boxed{2}"
    assert extract_final_answer(text) == "2"


def test_extract_unclosed_boxed_accepted() -> None:
    assert extract_final_answer(r"the ans is \boxed{99") == "99"


def test_boxed_regex_non_numeric_guard() -> None:
    assert extract_final_answer(r"The answer is \boxed{XLVII}") == "XLVII"


def test_verify_binary_leading_zero_rejected() -> None:
    assert verify("00011011", "11011") is False


def test_verify_binary_exact_matches() -> None:
    assert verify("10011000", "10011000") is True


def test_verify_binary_case_insensitive_still_works() -> None:
    assert verify("101", "101") is True


def test_verify_binary_float_like_ignored() -> None:
    assert verify("101", "101.0") is False


def test_verify_numeric_tolerance_ok() -> None:
    assert verify("24.64", "24.6401") is True


def test_binary_guard_only_fires_on_pure_01() -> None:
    assert verify("24", "24.1") is True


def test_verify_numeric_tolerance_rejects_far() -> None:
    assert verify("24.64", "25.00") is False


def test_verify_string_case_insensitive() -> None:
    assert verify("XLVII", "xlvii") is True


def test_verify_parameter_order_sanity() -> None:
    assert verify("11011", "00011011") is False


def test_fingerprints_present() -> None:
    assert len(EXTRACT_FINAL_ANSWER_SHA256) == 64
    assert len(VERIFY_SHA256) == 64
    assert len(EXPECTED_EXTRACT_FINAL_ANSWER_SHA256) == 64
    assert len(EXPECTED_VERIFY_SHA256) == 64


def test_notebook_source_parity() -> None:
    nb_path = Path(METRIC_NOTEBOOK_RELATIVE_PATH)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    official_src = "".join(
        "".join(cell["source"]) + "\n\n"
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )
    lines = official_src.splitlines(keepends=True)
    tree = ast.parse(official_src)

    def get_func(name: str) -> str:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return "".join(lines[node.lineno - 1:node.end_lineno])
        raise AssertionError(f"{name} not found in official notebook")

    from src.competition import official_metric_contract as contract

    assert inspect.getsource(contract.extract_final_answer) == get_func("extract_final_answer")
    assert inspect.getsource(contract.verify) == get_func("verify")


def test_contract_fingerprint_reports_expected_notebook_digests() -> None:
    fingerprint = current_contract_fingerprint().to_dict()
    assert fingerprint["expected_extract_final_answer_sha256"] == EXTRACT_FINAL_ANSWER_SHA256
    assert fingerprint["expected_verify_sha256"] == VERIFY_SHA256
