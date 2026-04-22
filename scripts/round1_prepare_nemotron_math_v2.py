"""round1_prepare_nemotron_math_v2.py
====================================
Filter and normalise a raw Nemotron-Math-v2 JSONL dump into the per-row schema
consumed by downstream Round 1 pipeline stages (GCD teacher distillation,
student SFT, etc.).

Problem this solves
-------------------
Nemotron-Math-v2 (NVIDIA, arXiv 2512.15489) is a large mixed-source math
dataset that ships with a ``used_in`` metadata field indicating which NVIDIA
model / training run the row was selected for.  The Nano-v3 training subset is
the most aligned with our competition base model (nvidia/Nemotron-3-Nano-30B-
A3B-BF16).  This script:

  1. Loads the raw JSONL.
  2. Retains only rows whose ``used_in`` field contains the --used-in-filter
     value (default: "nano_v3").
  3. Drops rows that involve code / tool use (Python interpreter calls,
     ``<tool_call>`` blocks, code blocks that define functions).
  4. Keeps only rows with a numeric or short binary answer that the competition
     ``verify()`` function can grade.
  5. Emits a normalised JSONL with schema::

       {id, prompt, target_answer, solution_trace, family,
        source_dataset="nemotron_math_v2"}

  6. Writes a summary JSON report alongside the output.

--dry-run mode
--------------
When --dry-run is passed (or when --input is absent), the script emits three
synthetic rows to stdout and, if --output is provided, writes them to disk.
No file reading occurs, so the script is smoke-testable without any GPU or
large-file dependency.

Upstream research
-----------------
* Nemotron-Math-v2: arXiv 2512.15489 (NVIDIA, 2024).
  The ``used_in`` field semantics are documented in the dataset README and
  referenced in Table 3 of the paper.

Usage
-----
    python scripts/round1_prepare_nemotron_math_v2.py \\
        --input  data/raw/nemotron_math_v2.jsonl \\
        --output data/processed/round1_nmath_v2.jsonl \\
        --used-in-filter nano_v3 \\
        --max-examples 50000

    python scripts/round1_prepare_nemotron_math_v2.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- heavy imports are deferred below main() so --dry-run has zero deps ---


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_DATASET = "nemotron_math_v2"

# Patterns that flag a row as involving tool/code use.
_CODE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"<tool_call>", re.IGNORECASE),
    re.compile(r"```python", re.IGNORECASE),
    re.compile(r"def\s+\w+\s*\("),          # function definition inside trace
    re.compile(r"import\s+\w+"),             # Python import statement
    re.compile(r"subprocess\."),
]

# Synthetic dry-run rows that prove the pipeline without real data.
_DRY_RUN_ROWS: list[dict[str, Any]] = [
    {
        "id": "dry_run_000",
        "prompt": "What is the sum of the first 10 natural numbers?",
        "target_answer": "55",
        "solution_trace": (
            "The sum of the first n natural numbers is n*(n+1)/2. "
            "For n=10: 10*11/2 = 55. \\boxed{55}"
        ),
        "family": "arithmetic",
        "source_dataset": SOURCE_DATASET,
    },
    {
        "id": "dry_run_001",
        "prompt": (
            "A rectangle has length 8 cm and width 5 cm. "
            "What is its area in square centimetres?"
        ),
        "target_answer": "40",
        "solution_trace": (
            "Area = length * width = 8 * 5 = 40 cm^2. \\boxed{40}"
        ),
        "family": "geometry",
        "source_dataset": SOURCE_DATASET,
    },
    {
        "id": "dry_run_002",
        "prompt": "Solve for x: 3x - 7 = 11.",
        "target_answer": "6",
        "solution_trace": (
            "3x = 11 + 7 = 18, so x = 18/3 = 6. \\boxed{6}"
        ),
        "family": "algebra",
        "source_dataset": SOURCE_DATASET,
    },
]


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def _is_tool_or_code_row(row: dict[str, Any]) -> bool:
    """Return True if any field in the row contains a code/tool-use marker."""
    check_fields = ("solution_trace", "solution", "completion", "prompt", "text")
    for field_name in check_fields:
        text = str(row.get(field_name, ""))
        for pattern in _CODE_PATTERNS:
            if pattern.search(text):
                return True
    return False


def _has_valid_answer(row: dict[str, Any]) -> bool:
    """Return True when the target_answer is a numeric or short binary string.

    Accepts:
    - Integer or float representations (possibly negative, possibly decimal).
    - Binary strings of digits 0/1 (up to 64 characters, consistent with
      competition bit-manipulation problems).
    - Short Roman numerals or single-word string answers (kept for generality;
      verify() handles these via case-insensitive string compare).

    Rejects:
    - Empty or missing answers.
    - Multi-sentence natural-language answers (not gradable by verify()).
    """
    answer = str(row.get("target_answer", "") or "").strip()
    if not answer or answer in ("NOT_FOUND", "None", "null"):
        return False
    # Numeric
    try:
        float(answer)
        return True
    except ValueError:
        pass
    # Binary string
    if re.fullmatch(r"[01]{1,64}", answer):
        return True
    # Short string (single token, e.g. Roman numeral, yes/no, letter)
    if len(answer) <= 32 and "\n" not in answer:
        return True
    return False


def _extract_used_in(row: dict[str, Any]) -> list[str]:
    """Return the used_in tags from a row as a list of strings.

    Nemotron-Math-v2 may store used_in as a string, a list, or a
    pipe-separated string depending on HuggingFace dataset serialisation.
    """
    raw = row.get("used_in", "")
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        if "|" in raw:
            return [s.strip() for s in raw.split("|") if s.strip()]
        if raw.strip():
            return [raw.strip()]
    return []


def _normalise_row(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    """Map a raw Nemotron-Math-v2 row to the normalised output schema."""
    # id: prefer existing id fields, fall back to index
    row_id = str(
        row.get("id")
        or row.get("uid")
        or row.get("problem_id")
        or f"nmath_v2_{row_index:07d}"
    )
    # prompt: the mathematical problem statement
    prompt = str(
        row.get("prompt")
        or row.get("problem")
        or row.get("question")
        or row.get("input")
        or ""
    ).strip()
    # target answer
    target_answer = str(
        row.get("target_answer")
        or row.get("answer")
        or row.get("gold_answer")
        or row.get("label")
        or ""
    ).strip()
    # solution trace
    solution_trace = str(
        row.get("solution_trace")
        or row.get("solution")
        or row.get("completion")
        or row.get("output")
        or ""
    ).strip()
    # problem family / category
    family = str(
        row.get("family")
        or row.get("subject")
        or row.get("category")
        or row.get("type")
        or "unknown"
    ).strip()

    return {
        "id": row_id,
        "prompt": prompt,
        "target_answer": target_answer,
        "solution_trace": solution_trace,
        "family": family,
        "source_dataset": SOURCE_DATASET,
    }


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def process_rows(
    rows: list[dict[str, Any]],
    used_in_filter: str,
    max_examples: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter and normalise raw rows.

    Returns (output_rows, report_dict).
    """
    n_total = len(rows)
    n_used_in_dropped = 0
    n_code_dropped = 0
    n_answer_dropped = 0
    output: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        # 1. used_in filter
        used_in_tags = _extract_used_in(row)
        if used_in_filter and not any(used_in_filter in tag for tag in used_in_tags):
            n_used_in_dropped += 1
            continue
        # 2. Drop tool/code rows
        if _is_tool_or_code_row(row):
            n_code_dropped += 1
            continue
        # 3. Normalise first so we can check the answer
        normed = _normalise_row(row, idx)
        if not _has_valid_answer(normed):
            n_answer_dropped += 1
            continue
        output.append(normed)
        if max_examples is not None and len(output) >= max_examples:
            break

    report: dict[str, Any] = {
        "source_dataset": SOURCE_DATASET,
        "used_in_filter": used_in_filter,
        "n_input_rows": n_total,
        "n_used_in_dropped": n_used_in_dropped,
        "n_code_dropped": n_code_dropped,
        "n_answer_dropped": n_answer_dropped,
        "n_output_rows": len(output),
        "max_examples_cap": max_examples,
    }
    return output, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter and normalise a raw Nemotron-Math-v2 JSONL dump. "
            "Outputs a processed JSONL and a JSON summary report. "
            "Pass --dry-run (or omit --input) to emit 3 synthetic rows "
            "without reading any file."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to the raw Nemotron-Math-v2 JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/round1_nmath_v2.jsonl",
        help="Path for the output processed JSONL (default: %(default)s).",
    )
    parser.add_argument(
        "--used-in-filter",
        default="nano_v3",
        dest="used_in_filter",
        help=(
            "Only keep rows whose used_in field contains this substring "
            "(default: %(default)s). Pass empty string to disable the filter."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        dest="max_examples",
        help="Optional cap on the number of output rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Emit 3 synthetic rows to prove the pipeline. Skips file reading. "
            "If --output is given, the synthetic rows are still written."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    dry_run: bool = args.dry_run or (args.input is None)
    used_in_filter: str = args.used_in_filter or ""
    max_examples: int | None = args.max_examples
    output_path = Path(args.output)
    report_path = output_path.with_suffix("").with_suffix(".report.json")

    if dry_run:
        print(
            "[round1_prepare_nemotron_math_v2] dry-run mode: "
            "emitting 3 synthetic rows (no file I/O required).",
            flush=True,
        )
        output_rows = list(_DRY_RUN_ROWS)
        report: dict[str, Any] = {
            "source_dataset": SOURCE_DATASET,
            "used_in_filter": used_in_filter,
            "n_input_rows": len(output_rows),
            "n_used_in_dropped": 0,
            "n_code_dropped": 0,
            "n_answer_dropped": 0,
            "n_output_rows": len(output_rows),
            "max_examples_cap": max_examples,
            "dry_run": True,
        }
        for row in output_rows:
            import json as _json
            print(_json.dumps(row, ensure_ascii=False))
    else:
        # Deferred heavy import -- not needed for --dry-run
        from src.common.io import load_jsonl, write_json, write_jsonl  # noqa: PLC0415

        print(
            f"[round1_prepare_nemotron_math_v2] loading {args.input} ...",
            flush=True,
        )
        raw_rows = load_jsonl(args.input)
        print(
            f"[round1_prepare_nemotron_math_v2] loaded {len(raw_rows)} rows, "
            f"filtering (used_in_filter={used_in_filter!r}) ...",
            flush=True,
        )
        output_rows, report = process_rows(raw_rows, used_in_filter, max_examples)

    if not dry_run or args.output:
        from src.common.io import write_json, write_jsonl  # noqa: PLC0415

        write_jsonl(output_path, output_rows)
        write_json(report_path, report)
        print(
            f"[round1_prepare_nemotron_math_v2] wrote {len(output_rows)} rows "
            f"to {output_path}",
            flush=True,
        )
        print(
            f"[round1_prepare_nemotron_math_v2] report -> {report_path}",
            flush=True,
        )

    print(
        f"[round1_prepare_nemotron_math_v2] done. "
        f"retained={report['n_output_rows']} / input={report['n_input_rows']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
