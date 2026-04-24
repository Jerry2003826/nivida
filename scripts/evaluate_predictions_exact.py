from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_REPO = REPO_ROOT / "nemotron_local_repo"
if LOCAL_REPO.is_dir() and str(LOCAL_REPO) not in sys.path:
    sys.path.insert(0, str(LOCAL_REPO))

from src.competition.answer_extract import extract_single_boxed_answer  # noqa: E402
from src.competition.metrics import competition_correct, exact_match  # noqa: E402
from src.competition.official_prompts import detect_official_family  # noqa: E402


def _load_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [dict(row) for row in obj]
        if isinstance(obj, dict) and isinstance(obj.get("rows"), list):
            return [dict(row) for row in obj["rows"]]
        raise ValueError(f"Unsupported JSON shape in {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def official_extract_final_answer(text: str | None) -> str:
    if text is None:
        return "NOT_FOUND"
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [match.strip() for match in matches if match.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def official_verify(stored_answer: str, predicted: str) -> bool:
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()
    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()
    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


def _pick(row: dict[str, Any], names: list[str], default: Any = None) -> Any:
    lowered = {key.lower(): key for key in row}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            return row.get(key)
    return default


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


def _row_id(row: dict[str, Any], index: int) -> str:
    return str(_pick(row, ["id", "sample_id", "uid"], index))


def _target(row: dict[str, Any]) -> str | None:
    value = _pick(row, ["target_answer", "answer", "target", "label", "output"])
    return None if value is None else str(value)


def _prediction(row: dict[str, Any], prediction_key: str | None) -> str:
    if prediction_key:
        value = row.get(prediction_key, "")
    else:
        value = _pick(row, ["prediction", "raw_generation", "completion", "answer", "output"], "")
    return "" if value is None else str(value)


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


def _answer_kind(target: str | None) -> str:
    if target is None:
        return "missing"
    text = target.strip()
    if re.fullmatch(r"[01]+", text):
        return "binary"
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return "numeric"
    if re.fullmatch(r"[IVXLCDM]+", text, flags=re.IGNORECASE):
        return "roman"
    if " " in text:
        return "text_phrase"
    return "text_token"


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model prediction JSONL/CSV against labeled rows with official-style parsing."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--prediction-key")
    parser.add_argument("--join", choices=["id", "order"], default="id")
    parser.add_argument("--output-json", type=Path, default=Path("data/processed/eval/exact_eval_report.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/eval/exact_eval_records.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("LOCAL_EXACT_EVAL_REPORT.md"))
    args = parser.parse_args()

    predictions = _load_table(args.predictions)
    labels = _load_table(args.labels)
    if args.join == "id":
        label_by_id = {_row_id(row, index): row for index, row in enumerate(labels)}
        joined = []
        missing_ids = []
        for index, pred_row in enumerate(predictions):
            rid = _row_id(pred_row, index)
            label_row = label_by_id.get(rid)
            if label_row is None:
                missing_ids.append(rid)
                continue
            joined.append((rid, pred_row, label_row))
    else:
        joined = [
            (_row_id(pred_row, index), pred_row, labels[index])
            for index, pred_row in enumerate(predictions[: len(labels)])
        ]
        missing_ids = []

    records: list[dict[str, Any]] = []
    for rid, pred_row, label_row in joined:
        prediction = _prediction(pred_row, args.prediction_key)
        target = _target(label_row)
        boxed = extract_single_boxed_answer(prediction)
        official_extracted = official_extract_final_answer(prediction)
        official_correct = False if target is None else official_verify(str(target), official_extracted)
        local_competition_correct = False if target is None else competition_correct(prediction, str(target))
        local_exact = False if target is None else exact_match(prediction, str(target))
        row = {
            "id": rid,
            "family": _family(label_row),
            "subtype": _subtype(label_row),
            "answer_kind": _answer_kind(target),
            "target": "" if target is None else target,
            "prediction": prediction,
            "official_extracted": official_extracted,
            "boxed_valid": boxed.is_valid,
            "boxed_error": boxed.error or "",
            "local_exact": local_exact,
            "local_competition_correct": local_competition_correct,
            "official_verify_correct": official_correct,
            "prediction_chars": len(prediction),
            "prediction_words": len(prediction.split()),
        }
        records.append(row)

    buckets: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    subtype_buckets: dict[str, list[bool]] = defaultdict(list)
    kind_buckets: dict[str, list[bool]] = defaultdict(list)
    boxed_by_family: dict[str, list[bool]] = defaultdict(list)
    for row in records:
        family = row["family"]
        buckets[family]["official"].append(bool(row["official_verify_correct"]))
        buckets[family]["local_competition"].append(bool(row["local_competition_correct"]))
        buckets[family]["local_exact"].append(bool(row["local_exact"]))
        subtype_buckets[f"{family}:{row['subtype']}"].append(bool(row["official_verify_correct"]))
        kind_buckets[row["answer_kind"]].append(bool(row["official_verify_correct"]))
        boxed_by_family[family].append(bool(row["boxed_valid"]))

    total = len(records)
    summary = {
        "predictions": str(args.predictions),
        "labels": str(args.labels),
        "join": args.join,
        "num_predictions": len(predictions),
        "num_labels": len(labels),
        "num_joined": total,
        "missing_prediction_ids": missing_ids,
        "overall": {
            "official_verify_accuracy": _rate([bool(row["official_verify_correct"]) for row in records]),
            "local_competition_accuracy": _rate([bool(row["local_competition_correct"]) for row in records]),
            "local_exact_accuracy": _rate([bool(row["local_exact"]) for row in records]),
            "boxed_valid_rate": _rate([bool(row["boxed_valid"]) for row in records]),
            "avg_prediction_words": sum(row["prediction_words"] for row in records) / total if total else 0.0,
        },
        "family": {
            family: {
                "n": len(values["official"]),
                "official_verify_accuracy": _rate(values["official"]),
                "local_competition_accuracy": _rate(values["local_competition"]),
                "local_exact_accuracy": _rate(values["local_exact"]),
                "boxed_valid_rate": _rate(boxed_by_family[family]),
            }
            for family, values in sorted(buckets.items())
        },
        "subtype_official_verify_accuracy": {
            name: {"n": len(values), "accuracy": _rate(values)}
            for name, values in sorted(subtype_buckets.items())
        },
        "answer_kind_official_verify_accuracy": {
            name: {"n": len(values), "accuracy": _rate(values)}
            for name, values in sorted(kind_buckets.items())
        },
        "boxed_errors": dict(Counter(row["boxed_error"] or "valid" for row in records)),
        "records": records,
    }
    _write_json(args.output_json, summary)
    _write_csv(args.output_csv, records)

    lines = [
        "# Local Exact Eval Report",
        "",
        f"- predictions: `{args.predictions}`",
        f"- labels: `{args.labels}`",
        f"- joined rows: `{total}`",
        f"- official-verify accuracy: `{summary['overall']['official_verify_accuracy']:.4f}`",
        f"- local competition accuracy: `{summary['overall']['local_competition_accuracy']:.4f}`",
        f"- boxed-valid rate: `{summary['overall']['boxed_valid_rate']:.4f}`",
        "",
        "## Family",
        "",
        "| family | n | official_verify | local_competition | local_exact | boxed_valid |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, row in summary["family"].items():
        lines.append(
            f"| {family} | {row['n']} | {row['official_verify_accuracy']:.4f} | "
            f"{row['local_competition_accuracy']:.4f} | {row['local_exact_accuracy']:.4f} | "
            f"{row['boxed_valid_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Answer Kind",
            "",
            "| answer_kind | n | official_verify |",
            "| --- | ---: | ---: |",
        ]
    )
    for kind, row in summary["answer_kind_official_verify_accuracy"].items():
        lines.append(f"| {kind} | {row['n']} | {row['accuracy']:.4f} |")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "num_joined": total}, indent=2))


if __name__ == "__main__":
    main()
