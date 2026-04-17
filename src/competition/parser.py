from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from src.common.io import load_table, read_yaml, write_jsonl
from src.common.logging_utils import configure_logging
from src.common.text_normalise import canonical_text
from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair


LOGGER = configure_logging(name="competition.parser")

_INPUT_OUTPUT_PATTERN = re.compile(
    r"Input(?:\s*\d+)?\s*:\s*(?P<input>.*?)\s*Output(?:\s*\d+)?\s*:\s*(?P<output>.*?)(?=(?:Input(?:\s*\d+)?\s*:|Query(?:\s*input)?\s*:|Test(?:\s*input)?\s*:|$))",
    flags=re.IGNORECASE | re.DOTALL,
)
_ARROW_PAIR_PATTERN = re.compile(
    r"^(?P<input>[^\n\r]+?)\s*->\s*(?P<output>[^\n\r]+?)$",
    flags=re.MULTILINE,
)
_BECOMES_PAIR_PATTERN = re.compile(
    r"^(?P<input>[^\n\r]+?)\s+becomes\s+(?P<output>[^\n\r]+?)$",
    flags=re.MULTILINE | re.IGNORECASE,
)
_GRAVITY_PAIR_PATTERN = re.compile(
    r"^For\s+t\s*=\s*(?P<input>[+-]?\d+(?:\.\d+)?)s,\s+distance\s*=\s*(?P<output>[+-]?\d+(?:\.\d+)?)\s*m$",
    flags=re.MULTILINE | re.IGNORECASE,
)
_EQUATION_PAIR_PATTERN = re.compile(
    r"^(?P<input>[^=\n\r]{1,64}?)\s*=\s*(?P<output>[^=\n\r]{1,64}?)$",
    flags=re.MULTILINE,
)
_QUERY_PATTERN = re.compile(
    r"(?:Query(?:\s*input)?|Test(?:\s*input)?|Apply to|Now transform)\s*:\s*(?P<query>.*?)(?=(?:Answer|Target|$))",
    flags=re.IGNORECASE | re.DOTALL,
)
_QUERY_FOR_PATTERN = re.compile(
    r"(?:Now,\s*)?(?:determine|find|compute|predict|infer)\s+the\s+(?:output|result)\s+for\s*:\s*(?P<query>[^\n\r]+)",
    flags=re.IGNORECASE,
)
_GRAVITY_QUERY_PATTERN = re.compile(
    r"Now,\s*determine\s+the\s+falling\s+distance\s+for\s+t\s*=\s*(?P<query>[+-]?\d+(?:\.\d+)?)s",
    flags=re.IGNORECASE,
)
_CONVERT_QUERY_PATTERN = re.compile(
    r"Now,\s*convert\s+the\s+following\s+measurement\s*:\s*(?P<query>[^\n\r]+)",
    flags=re.IGNORECASE,
)
_DECRYPT_QUERY_PATTERN = re.compile(
    r"Now,\s*decrypt\s+the\s+following\s+text\s*:\s*(?P<query>[^\n\r]+)",
    flags=re.IGNORECASE,
)
_NUMERAL_QUERY_PATTERN = re.compile(
    r"Now,\s*write\s+the\s+number\s+(?P<query>[^\s\n\r]+)\s+in\s+the\s+Wonderland\s+numeral\s+system",
    flags=re.IGNORECASE,
)


def _safe_literal_eval(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    raw = raw.strip()
    if not raw:
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw


def _normalise_segment(text: Any) -> str:
    return canonical_text("" if text is None else str(text))


def _extract_pairs_from_prompt(prompt: str) -> list[PuzzlePair]:
    pairs: list[PuzzlePair] = []
    for match in _INPUT_OUTPUT_PATTERN.finditer(prompt):
        pairs.append(
            PuzzlePair(
                input=_normalise_segment(match.group("input")),
                output=_normalise_segment(match.group("output")),
            )
        )
    if pairs:
        return pairs

    for match in _ARROW_PAIR_PATTERN.finditer(prompt):
        lhs = _normalise_segment(match.group("input"))
        rhs = _normalise_segment(match.group("output"))
        lowered = f"{lhs} {rhs}".lower()
        if "input" in lowered and "output" in lowered:
            continue
        if lhs and rhs:
            pairs.append(PuzzlePair(input=lhs, output=rhs))
    for match in _BECOMES_PAIR_PATTERN.finditer(prompt):
        lhs = _normalise_segment(match.group("input"))
        rhs = _normalise_segment(match.group("output"))
        if lhs and rhs:
            pairs.append(PuzzlePair(input=lhs, output=rhs))
    for match in _GRAVITY_PAIR_PATTERN.finditer(prompt):
        lhs = _normalise_segment(match.group("input"))
        rhs = _normalise_segment(match.group("output"))
        if lhs and rhs:
            pairs.append(PuzzlePair(input=lhs, output=rhs))
    for match in _EQUATION_PAIR_PATTERN.finditer(prompt):
        lhs = _normalise_segment(match.group("input"))
        rhs = _normalise_segment(match.group("output"))
        lowered = lhs.lower()
        if lowered.startswith(("in alice", "for t", "now,", "here are")):
            continue
        if "given d" in rhs.lower():
            continue
        if lhs and rhs and PuzzlePair(input=lhs, output=rhs) not in pairs:
            pairs.append(PuzzlePair(input=lhs, output=rhs))
    return pairs


def _extract_query_from_prompt(prompt: str) -> str:
    query_match = _QUERY_PATTERN.search(prompt)
    if query_match:
        return _normalise_segment(query_match.group("query"))
    query_match = _QUERY_FOR_PATTERN.search(prompt)
    if query_match:
        return _normalise_segment(query_match.group("query"))
    for pattern in (
        _GRAVITY_QUERY_PATTERN,
        _CONVERT_QUERY_PATTERN,
        _DECRYPT_QUERY_PATTERN,
        _NUMERAL_QUERY_PATTERN,
    ):
        query_match = pattern.search(prompt)
        if query_match:
            return _normalise_segment(query_match.group("query"))
    return ""


def _parse_pairs_field(raw: Any) -> list[PuzzlePair]:
    parsed = _safe_literal_eval(raw)
    if isinstance(parsed, list):
        pairs: list[PuzzlePair] = []
        for item in parsed:
            if isinstance(item, dict):
                inp = item.get("input", item.get("source", item.get("x", "")))
                out = item.get("output", item.get("target", item.get("y", "")))
                pairs.append(PuzzlePair(input=_normalise_segment(inp), output=_normalise_segment(out)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append(PuzzlePair(input=_normalise_segment(item[0]), output=_normalise_segment(item[1])))
        return pairs
    return []


def _pick_column(row: dict[str, Any], names: list[str]) -> Any:
    lowered = {key.lower(): key for key in row}
    for name in names:
        if name.lower() in lowered:
            return row[lowered[name.lower()]]
    return None


def parse_row(row: dict[str, Any], *, source: str, split: str, row_index: int) -> PuzzleExample:
    example_id = _normalise_segment(_pick_column(row, ["id", "sample_id", "uid"]) or f"{split}_{row_index}")
    raw_prompt = _normalise_segment(_pick_column(row, ["raw_prompt", "prompt", "question", "text"]) or "")
    target_answer = _pick_column(row, ["target_answer", "answer", "output", "label"])
    query_input = _pick_column(row, ["query_input", "query", "test_input"])
    pair_field = _pick_column(row, ["train_pairs", "examples", "pairs"])

    train_pairs = _parse_pairs_field(pair_field)
    if not train_pairs and raw_prompt:
        train_pairs = _extract_pairs_from_prompt(raw_prompt)

    if query_input is None:
        query_input = _extract_query_from_prompt(raw_prompt)

    metadata = PuzzleMetadata(
        source=source,
        split=split,
        extras={
            key: value
            for key, value in row.items()
            if key.lower() not in {"id", "sample_id", "uid", "raw_prompt", "prompt", "question", "text", "target_answer", "answer", "output", "label", "query_input", "query", "test_input", "train_pairs", "examples", "pairs"}
        },
    )
    return PuzzleExample(
        id=example_id,
        raw_prompt=raw_prompt,
        train_pairs=train_pairs,
        query_input=_normalise_segment(query_input),
        target_answer=None if target_answer is None else _normalise_segment(target_answer),
        metadata=metadata,
    )


def parse_competition_file(path: str | Path, *, source: str, split: str) -> list[PuzzleExample]:
    frame = load_table(path)
    rows = frame.to_dict(orient="records")
    return [parse_row(row, source=source, split=split, row_index=index) for index, row in enumerate(rows)]


def main() -> None:
    cli = argparse.ArgumentParser(description="Parse Kaggle competition data into canonical JSONL.")
    cli.add_argument("--config", help="YAML config path.")
    cli.add_argument("--input", help="CSV/Parquet/JSONL input file.")
    cli.add_argument("--output", help="Canonical JSONL output path.")
    cli.add_argument("--source", default="kaggle")
    cli.add_argument("--split", default="train")
    args = cli.parse_args()

    if args.config:
        config = read_yaml(args.config)
        args.input = config.get("input_path", args.input)
        args.output = config.get("output_path", args.output)
        args.source = config.get("source", args.source)
        args.split = config.get("dataset_split", config.get("split", args.split))

    if not args.input or not args.output:
        raise ValueError("Both --input and --output are required.")

    examples = parse_competition_file(args.input, source=args.source, split=args.split)
    write_jsonl(args.output, [example.to_dict() for example in examples])
    LOGGER.info("Parsed %s examples -> %s", len(examples), args.output)


if __name__ == "__main__":
    main()
