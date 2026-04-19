"""Byte-for-byte copy of the official Kaggle metric for
`nvidia-nemotron-model-reasoning-challenge`.

Source: 官方资料/kaggle_metric/nvidia-nemotron-metric.ipynb (cell 0)
Snapshot date is tracked by METRIC_NOTEBOOK_SNAPSHOT_DATE. Do NOT rewrite
these functions into "equivalent" forms - any divergence breaks parity with
the scoring runner.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


METRIC_NOTEBOOK_RELATIVE_PATH = (
    "官方资料/kaggle_metric/nvidia-nemotron-metric.ipynb"
)
METRIC_NOTEBOOK_SNAPSHOT_DATE = "2026-04-19"


OFFICIAL_LLM_KWARGS: dict[str, Any] = {
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

OFFICIAL_SAMPLING_KWARGS: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 3584,
}

OFFICIAL_GUARD_TEXT = (
    "\nPlease put your final answer inside `\\boxed{}`."
    " For example: `\\boxed{your answer}`"
)

OFFICIAL_CHAT_TEMPLATE_KWARGS: dict[str, Any] = {
    "tokenize": False,
    "add_generation_prompt": True,
    "enable_thinking": True,
}


def extract_final_answer(text: str | None) -> str:
    r"""Extracts the final answer from the model response.

    Prioritizes extracting answers inside `\boxed{}`.
    If no `\boxed{}` format is found, attempts to extract numbers from other formats.

    Examples:
        >>> extract_final_answer(r"The answer is \boxed{42}")
        '42'
        >>> extract_final_answer("The final answer is: 3.14")
        '3.14'
        >>> extract_final_answer("Just a number 100 in text")
        '100'
        >>> extract_final_answer(None)
        'NOT_FOUND'
    """
    if text is None:
        return 'NOT_FOUND'

    # Search for boxed answer
    # Match all instances of \boxed{...} or unclosed \boxed{ at the end
    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    # Other common formats if \boxed{} is not found
    patterns = [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[:：]\s*([^\n]+)',
        r'final answer\s*[:：]\s*([^\n]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    # If no structured format is found, extract the last valid number in the text
    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        return matches[-1]

    # If no numeric answer is found, return the last line of text as a fallback
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else 'NOT_FOUND'


def verify(stored_answer: str, predicted: str) -> bool:
    """Verify if the answer matches.

    For numerical answers, allow them to be judged as equal within a certain relative tolerance (1e-2);
    otherwise, compare strictly as strings (case-insensitive).

    Examples:
        >>> verify("10011000", "10011000")
        True
        >>> verify("10011000", "10011001")
        False
        >>> verify("24.64", "24.6401")
        True
        >>> verify("XLVII", "xlvii")
        True
        >>> verify("11011", "00011011")
        False
    """
    # Clean up strings
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    # If the answer is a binary string, compare strictly as strings
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        # Try to convert the answers to floating point numbers
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        # Use a small absolute tolerance for numbers near zero
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        # Fallback to case-insensitive string comparison
        return predicted.lower() == stored_answer.lower()


def _source_of(func: Any) -> str:
    return inspect.getsource(func)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


EXTRACT_FINAL_ANSWER_SHA256 = _sha256(_source_of(extract_final_answer))
VERIFY_SHA256 = _sha256(_source_of(verify))


def build_official_prompt(user_prompt: str, tokenizer: Any) -> str:
    """Construct the exact prompt string the official runner sends to vLLM."""
    user_content = user_prompt + OFFICIAL_GUARD_TEXT
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            **OFFICIAL_CHAT_TEMPLATE_KWARGS,
        )
    except Exception:
        return user_content


@dataclass(frozen=True, slots=True)
class ContractFingerprint:
    snapshot_date: str
    extract_sha256: str
    verify_sha256: str
    llm_kwargs: Mapping[str, Any]
    sampling_kwargs: Mapping[str, Any]
    guard_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_notebook_snapshot_date": self.snapshot_date,
            "metric_notebook_relative_path": METRIC_NOTEBOOK_RELATIVE_PATH,
            "extract_final_answer_sha256": self.extract_sha256,
            "verify_sha256": self.verify_sha256,
            "official_llm_kwargs": dict(self.llm_kwargs),
            "official_sampling_kwargs": dict(self.sampling_kwargs),
            "official_guard_text": self.guard_text,
        }


def current_contract_fingerprint() -> ContractFingerprint:
    return ContractFingerprint(
        snapshot_date=METRIC_NOTEBOOK_SNAPSHOT_DATE,
        extract_sha256=EXTRACT_FINAL_ANSWER_SHA256,
        verify_sha256=VERIFY_SHA256,
        llm_kwargs=OFFICIAL_LLM_KWARGS,
        sampling_kwargs=OFFICIAL_SAMPLING_KWARGS,
        guard_text=OFFICIAL_GUARD_TEXT,
    )
