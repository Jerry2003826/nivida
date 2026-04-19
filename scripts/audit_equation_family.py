from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import write_json
from src.competition.official_prompts import detect_official_family


PROMPT_TYPES: tuple[str, ...] = (
    "symbolic_char",
    "operator_numeric_mixed",
    "position_transduction",
    "unknown",
)
ANSWER_TYPES: tuple[str, ...] = (
    "answer_single_char",
    "answer_integer",
    "answer_decimal",
    "answer_short_symbol_string",
    "answer_other_text",
)

_PAIR_LINE_RE = re.compile(r"^(\S+)\s*=\s*(\S+)$", re.MULTILINE)
_PAIR_INLINE_RE = re.compile(r"(\S+)\s*=\s*(\S+)(?=[\n,;])")
_OPERATOR_RE = re.compile(r"\d+\s*[+\-*/|\\%^]\s*\d+\s*=\s*\S+")
_ARITHMETIC_TOKEN_RE = re.compile(r"\d+[+\-*/|\\]\d+")
_QUERY_RE = re.compile(r"Now,\s*determine the result for:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

_RECOMMENDED_BRANCH = {
    ("symbolic_char", "answer_single_char"): "equation-symbolic-char-substitution",
    ("symbolic_char", "answer_short_symbol_string"): "equation-symbolic-sequence-transducer",
    ("symbolic_char", "answer_integer"): "equation-symbolic-to-int",
    ("operator_numeric_mixed", "answer_integer"): "equation-operator-induction",
    ("operator_numeric_mixed", "answer_decimal"): "equation-numeric-precision",
    ("position_transduction", "answer_short_symbol_string"): "equation-position-transducer",
}


def extract_shown_examples(prompt: str) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    examples: list[tuple[str, str]] = []
    for pattern in (_PAIR_LINE_RE, _PAIR_INLINE_RE):
        for left, right in pattern.findall(prompt):
            pair = (left, right)
            if pair in seen:
                continue
            seen.add(pair)
            examples.append(pair)
    return examples


def extract_query(prompt: str) -> str | None:
    match = _QUERY_RE.search(prompt)
    if not match:
        return None
    return match.group(1).strip()


def _analysis_text(prompt: str) -> str:
    examples = extract_shown_examples(prompt)
    query = extract_query(prompt)
    if not examples and not query:
        return prompt
    chunks: list[str] = []
    for left, right in examples:
        chunks.extend([left, right])
    if query:
        chunks.append(query)
    return " ".join(chunks)


def classify_prompt_type(prompt: str) -> str:
    prompt = str(prompt or "")
    analysis_text = _analysis_text(prompt)
    condensed = re.sub(r"\s+", "", analysis_text)
    symbolic_ratio = 0.0
    if condensed:
        symbolic_ratio = sum(not char.isalnum() for char in condensed) / len(condensed)

    non_digit_letter_ratio = 0.0
    if analysis_text:
        non_digit_letter_ratio = (
            sum(not (char.isdigit() or char.isalpha()) for char in analysis_text)
            / len(analysis_text)
        )

    if symbolic_ratio >= 0.60 or (
        non_digit_letter_ratio >= 0.50 and not _ARITHMETIC_TOKEN_RE.search(analysis_text)
    ):
        return "symbolic_char"

    has_digits = any(char.isdigit() for char in analysis_text)
    has_operator_like = any(char in "+-*/|\\%^" for char in analysis_text)
    if has_digits and has_operator_like and _OPERATOR_RE.search(prompt):
        return "operator_numeric_mixed"

    examples = extract_shown_examples(prompt)
    if examples:
        equal_length = sum(
            len(left.replace(" ", "")) == len(right.replace(" ", ""))
            for left, right in examples
        )
        if equal_length / len(examples) >= 0.70:
            return "position_transduction"

    return "unknown"


def classify_answer_type(answer: Any) -> str:
    text = str("" if answer is None else answer).strip()
    if re.fullmatch(r"-?\d+", text):
        return "answer_integer"
    if re.fullmatch(r"-?\d+\.\d+", text) or re.fullmatch(r"-?\d+/\d+", text):
        return "answer_decimal"
    if len(text) == 1:
        return "answer_single_char"
    if 2 <= len(text) <= 5:
        return "answer_short_symbol_string"
    return "answer_other_text"


def recommend_branch(prompt_type: str, answer_type: str) -> str:
    if answer_type == "answer_other_text":
        return "investigate-manual"
    return _RECOMMENDED_BRANCH.get((prompt_type, answer_type), "investigate-manual")


def _bucket_payload() -> dict[str, int]:
    return {answer_type: 0 for answer_type in ANSWER_TYPES}


def audit_equation_family(
    *,
    input_path: str | Path,
    family_column: str,
    prompt_column: str,
    answer_column: str,
    family_value: str,
) -> tuple[dict[str, Any], dict[tuple[str, str], list[dict[str, Any]]]]:
    frame = pd.read_csv(input_path)
    if family_column in frame.columns:
        family_series = frame[family_column].astype(str)
    elif prompt_column in frame.columns:
        family_series = frame[prompt_column].fillna("").map(
            lambda prompt: detect_official_family(str(prompt)) or "unknown"
        )
    else:
        raise KeyError(
            f"Neither family column '{family_column}' nor prompt column '{prompt_column}' is available."
        )
    filtered = frame[family_series == family_value].copy()

    prompt_counts = Counter({prompt_type: 0 for prompt_type in PROMPT_TYPES})
    answer_counts = Counter({answer_type: 0 for answer_type in ANSWER_TYPES})
    joint_counts = {prompt_type: _bucket_payload() for prompt_type in PROMPT_TYPES}
    samples_by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = {
        (prompt_type, answer_type): []
        for prompt_type in PROMPT_TYPES
        for answer_type in ANSWER_TYPES
    }

    for index, row in filtered.iterrows():
        prompt = str(row.get(prompt_column, "") or "")
        answer = str(row.get(answer_column, "") or "")
        prompt_type = classify_prompt_type(prompt)
        answer_type = classify_answer_type(answer)
        prompt_counts[prompt_type] += 1
        answer_counts[answer_type] += 1
        joint_counts[prompt_type][answer_type] += 1
        samples_by_bucket[(prompt_type, answer_type)].append(
            {
                "id": str(row.get("id", index)),
                "prompt": prompt,
                "answer": answer,
                "query": extract_query(prompt),
                "shown_examples": [
                    {"input": left, "output": right}
                    for left, right in extract_shown_examples(prompt)
                ],
            }
        )

    total_examples = int(len(filtered))
    dominant_prompt = "unknown"
    dominant_answer = "answer_other_text"
    dominant_count = 0
    for prompt_type in PROMPT_TYPES:
        for answer_type in ANSWER_TYPES:
            count = joint_counts[prompt_type][answer_type]
            if count > dominant_count:
                dominant_prompt = prompt_type
                dominant_answer = answer_type
                dominant_count = count

    payload = {
        "input_path": str(input_path),
        "family_value": family_value,
        "total_examples": total_examples,
        "prompt_counts": {key: int(prompt_counts[key]) for key in PROMPT_TYPES},
        "answer_counts": {key: int(answer_counts[key]) for key in ANSWER_TYPES},
        "joint_counts": joint_counts,
        "dominant_joint_bucket": {
            "prompt_type": dominant_prompt,
            "answer_type": dominant_answer,
            "count": int(dominant_count),
            "fraction": 0.0 if total_examples == 0 else dominant_count / total_examples,
        },
        "recommended_branch": recommend_branch(dominant_prompt, dominant_answer),
    }
    return payload, samples_by_bucket


def render_samples_markdown(
    *,
    payload: dict[str, Any],
    samples_by_bucket: dict[tuple[str, str], list[dict[str, Any]]],
    samples_per_cell: int,
) -> str:
    rng = random.Random(42)
    lines: list[str] = []
    total_examples = max(1, int(payload["total_examples"]))
    for prompt_type in PROMPT_TYPES:
        for answer_type in ANSWER_TYPES:
            bucket = samples_by_bucket[(prompt_type, answer_type)]
            if not bucket:
                continue
            count = len(bucket)
            fraction = count / total_examples
            chosen = rng.sample(bucket, k=min(samples_per_cell, count))
            lines.append(
                f"## {prompt_type} × {answer_type} (n={count}, {fraction:.1%})"
            )
            lines.append("")
            for index, sample in enumerate(chosen, start=1):
                prompt_preview = (
                    sample["prompt"][:300].replace("\r\n", "\n").replace("\r", "\n")
                )
                lines.append(f"### Sample {index} (id={sample['id']})")
                lines.append("")
                lines.append("**Prompt** (first 300 chars):")
                lines.append(f"> {prompt_preview!r}")
                lines.append("")
                lines.append(f"**Answer**: `{sample['answer']}`")
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit the equation family prompt/answer distribution.")
    parser.add_argument("--input", default="官方资料/competition_data/unzipped/train.csv")
    parser.add_argument("--output", default="data/processed/equation_family_audit.json")
    parser.add_argument("--sample-output", default="data/processed/equation_family_samples.md")
    parser.add_argument("--family-column", default="family")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--family-value", default="equation")
    parser.add_argument("--samples-per-cell", type=int, default=5)
    args = parser.parse_args(argv)

    payload, samples_by_bucket = audit_equation_family(
        input_path=args.input,
        family_column=args.family_column,
        prompt_column=args.prompt_column,
        answer_column=args.answer_column,
        family_value=args.family_value,
    )
    write_json(args.output, payload)
    sample_path = Path(args.sample_output)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(
        render_samples_markdown(
            payload=payload,
            samples_by_bucket=samples_by_bucket,
            samples_per_cell=max(0, int(args.samples_per_cell)),
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
