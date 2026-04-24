from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_REPO = REPO_ROOT / "nemotron_local_repo"
if LOCAL_REPO.is_dir() and str(LOCAL_REPO) not in sys.path:
    sys.path.insert(0, str(LOCAL_REPO))

from src.competition.official_prompts import detect_official_family  # noqa: E402


def _load(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [dict(row) for row in obj]
        if isinstance(obj, dict) and isinstance(obj.get("rows"), list):
            return [dict(row) for row in obj["rows"]]
        raise ValueError(f"Unsupported JSON shape: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata")
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str) and meta.strip():
        try:
            return json.loads(meta)
        except json.JSONDecodeError:
            try:
                import ast

                obj = ast.literal_eval(meta)
                return obj if isinstance(obj, dict) else {}
            except (SyntaxError, ValueError):
                return {}
    return {}


def _pick(row: dict[str, Any], names: list[str], default: Any = None) -> Any:
    lowered = {key.lower(): key for key in row}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            return row.get(key)
    return default


def _family(row: dict[str, Any]) -> str:
    meta = _metadata(row)
    value = _pick(row, ["official_family", "family"], None)
    if value is None:
        value = meta.get("official_family") or meta.get("family")
    if value is None:
        prompt = str(_pick(row, ["raw_prompt", "prompt", "question", "text"], ""))
        value = detect_official_family(prompt) if prompt else None
    return "unknown" if value is None else str(value)


def _subtype(row: dict[str, Any]) -> str:
    meta = _metadata(row)
    value = _pick(row, ["subtype"], None)
    if value is None:
        value = meta.get("subtype")
    return "unknown" if value is None else str(value)


def _prompt(row: dict[str, Any]) -> str:
    return str(_pick(row, ["raw_prompt", "prompt", "question", "text"], ""))


def _answer(row: dict[str, Any]) -> str:
    value = _pick(row, ["target_answer", "answer", "target", "label", "output"], "")
    return "" if value is None else str(value)


def _answer_kind(answer: str) -> str:
    text = answer.strip()
    if not text:
        return "missing"
    if re.fullmatch(r"[01]+", text):
        return "binary"
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return "numeric"
    if re.fullmatch(r"[IVXLCDM]+", text, flags=re.IGNORECASE):
        return "roman"
    if " " in text:
        return "text_phrase"
    return "text_token"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def analyze_file(path: Path) -> dict[str, Any]:
    rows = _load(path)
    family_counter: Counter[str] = Counter()
    subtype_counter: Counter[str] = Counter()
    kind_counter: Counter[str] = Counter()
    prompt_lengths_by_family: dict[str, list[int]] = defaultdict(list)
    answer_lengths_by_family: dict[str, list[int]] = defaultdict(list)
    empty_answer = 0
    for row in rows:
        family = _family(row)
        subtype = _subtype(row)
        prompt = _prompt(row)
        answer = _answer(row)
        family_counter[family] += 1
        subtype_counter[f"{family}:{subtype}"] += 1
        kind_counter[_answer_kind(answer)] += 1
        prompt_lengths_by_family[family].append(len(prompt))
        answer_lengths_by_family[family].append(len(answer))
        if not answer.strip():
            empty_answer += 1
    family_rows = []
    for family, count in family_counter.most_common():
        lengths = prompt_lengths_by_family[family]
        ans_lengths = answer_lengths_by_family[family]
        family_rows.append(
            {
                "family": family,
                "n": count,
                "share": count / len(rows) if rows else 0.0,
                "prompt_chars_mean": _mean(lengths),
                "prompt_chars_p50": _percentile(lengths, 0.50),
                "prompt_chars_p90": _percentile(lengths, 0.90),
                "answer_chars_mean": _mean(ans_lengths),
            }
        )
    return {
        "path": str(path),
        "num_rows": len(rows),
        "empty_answer_rows": empty_answer,
        "family_counts": dict(family_counter.most_common()),
        "subtype_counts": dict(subtype_counter.most_common()),
        "answer_kind_counts": dict(kind_counter.most_common()),
        "family_rows": family_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze task families, subtype balance, and answer formats.")
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="CSV/JSONL/JSON file. Repeatable. Defaults to key local datasets.",
    )
    parser.add_argument("--output-json", type=Path, default=Path("data/processed/data_family_analysis.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/data_family_analysis_family_rows.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("DATA_FAMILY_ANALYSIS.md"))
    args = parser.parse_args()
    inputs = args.input or [
        Path("data/official_kaggle/train.csv"),
        Path("data/official_kaggle/test.csv"),
        Path("data/processed/stage2_official_valid_hard_triad.jsonl"),
        Path("data/processed/proxy_all_family_valid.jsonl"),
        Path("data/processed/stage2_distill_train.jsonl"),
        Path("data/processed/stage2_distill_valid.jsonl"),
    ]

    reports = [analyze_file(path) for path in inputs if path.is_file()]
    payload = {"reports": reports}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_rows = []
    for report in reports:
        for row in report["family_rows"]:
            csv_rows.append({"path": report["path"], **row})
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "path",
            "family",
            "n",
            "share",
            "prompt_chars_mean",
            "prompt_chars_p50",
            "prompt_chars_p90",
            "answer_chars_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# Data Family Analysis",
        "",
        "This is local-only. It summarizes distribution, not model quality.",
        "",
    ]
    for report in reports:
        lines.extend(
            [
                f"## `{report['path']}`",
                "",
                f"- rows: `{report['num_rows']}`",
                f"- empty answers: `{report['empty_answer_rows']}`",
                f"- answer kinds: `{json.dumps(report['answer_kind_counts'], ensure_ascii=False)}`",
                "",
                "| family | n | share | prompt_mean | prompt_p90 | answer_mean |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in report["family_rows"]:
            lines.append(
                f"| {row['family']} | {row['n']} | {row['share']:.3f} | "
                f"{row['prompt_chars_mean']:.1f} | {row['prompt_chars_p90']:.0f} | "
                f"{row['answer_chars_mean']:.1f} |"
            )
        top_subtypes = list(report["subtype_counts"].items())[:12]
        lines.extend(
            [
                "",
                "Top subtypes:",
                "",
                "`" + json.dumps(dict(top_subtypes), ensure_ascii=False) + "`",
                "",
            ]
        )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "files": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
