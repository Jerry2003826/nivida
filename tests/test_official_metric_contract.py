from __future__ import annotations

from src.competition.official_metric_contract import (
    EXTRACT_FINAL_ANSWER_SHA256,
    OFFICIAL_GUARD_TEXT,
    OFFICIAL_LLM_KWARGS,
    OFFICIAL_SAMPLING_KWARGS,
    VERIFY_SHA256,
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


def test_extract_fallback_last_number() -> None:
    assert extract_final_answer("Just a number 100 in text") == "100"


def test_extract_last_nonempty_boxed_wins() -> None:
    text = r"\boxed{1} and later \boxed{2}"
    assert extract_final_answer(text) == "2"


def test_extract_unclosed_boxed_accepted() -> None:
    assert extract_final_answer(r"the ans is \boxed{99") == "99"


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


def test_verify_numeric_tolerance_rejects_far() -> None:
    assert verify("24.64", "25.00") is False


def test_verify_string_case_insensitive() -> None:
    assert verify("XLVII", "xlvii") is True


def test_verify_parameter_order_sanity() -> None:
    assert verify("11011", "00011011") is False


def test_fingerprints_present() -> None:
    assert len(EXTRACT_FINAL_ANSWER_SHA256) == 64
    assert len(VERIFY_SHA256) == 64
