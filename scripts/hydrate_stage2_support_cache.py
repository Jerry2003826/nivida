from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import load_jsonl, read_json, write_json, write_jsonl  # noqa: E402
from src.competition.schema import PuzzleExample  # noqa: E402
from src.teacher.chain_search import ChainSearchEngine  # noqa: E402
from src.teacher.program_signature import annotate_example_from_candidates  # noqa: E402
from src.teacher.stage2_annotation_provenance import (  # noqa: E402
    STAGE2_ANNOTATION_BEAM_WIDTH,
    STAGE2_ANNOTATION_MAX_DEPTH,
    STAGE2_ANNOTATION_TOP_K,
    sha256_file,
)


DEFAULT_INPUT_JSONL = Path("../data/processed/stage2_official_train_no_hard_valid.jsonl")


def _extras(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        row["metadata"] = metadata
    extras = metadata.setdefault("extras", {})
    if not isinstance(extras, dict):
        extras = {}
        metadata["extras"] = extras
    return extras


def _has_support_cache(row: dict[str, Any]) -> bool:
    extras = _extras(row)
    return isinstance(extras.get("support_pairs"), list) and "query_prediction" in extras


def _hydrate_row(row: dict[str, Any], *, engine: ChainSearchEngine) -> dict[str, Any]:
    example = PuzzleExample.from_dict(row)
    candidates = engine.solve_example(example, top_k=STAGE2_ANNOTATION_TOP_K)
    annotate_example_from_candidates(example, candidates)
    return example.to_dict()


def _update_provenance(
    *,
    provenance_json: Path,
    output_jsonl: Path,
    summary: dict[str, Any],
) -> bool:
    if not provenance_json.is_file():
        return False
    provenance = read_json(provenance_json)
    if not isinstance(provenance, dict):
        return False
    provenance["output_jsonl_path"] = str(output_jsonl)
    provenance["output_jsonl_sha256"] = sha256_file(output_jsonl)
    provenance["support_pair_cache"] = {
        "version": 1,
        "complete": bool(summary["complete"]),
        "hydrated_rows": int(summary["hydrated_rows"]),
        "already_cached_rows": int(summary["already_cached_rows"]),
        "remaining_missing_rows": int(summary["remaining_missing_rows"]),
    }
    write_json(provenance_json, provenance)
    return True


def hydrate_support_cache(
    *,
    input_jsonl: str | Path = DEFAULT_INPUT_JSONL,
    output_jsonl: str | Path | None = None,
    provenance_json: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl) if output_jsonl is not None else input_path
    provenance_path = (
        Path(provenance_json)
        if provenance_json is not None
        else Path(f"{output_path}.provenance.json")
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing stage2 JSONL: {input_path}")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if limit is not None and output_path == input_path:
        raise ValueError("Use a separate --output-jsonl when --limit is set.")

    rows = load_jsonl(input_path)
    engine = ChainSearchEngine(
        beam_width=STAGE2_ANNOTATION_BEAM_WIDTH,
        max_depth=STAGE2_ANNOTATION_MAX_DEPTH,
    )
    hydrated_rows: list[dict[str, Any]] = []
    already_cached = 0
    hydrated = 0
    remaining = 0

    for row in rows:
        if _has_support_cache(row):
            already_cached += 1
            hydrated_rows.append(row)
            continue
        if limit is not None and hydrated >= limit:
            hydrated_rows.append(row)
            continue
        hydrated_rows.append(_hydrate_row(row, engine=engine))
        hydrated += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path == input_path:
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        write_jsonl(tmp_path, hydrated_rows)
        tmp_path.replace(output_path)
    else:
        write_jsonl(output_path, hydrated_rows)

    remaining += sum(1 for row in hydrated_rows if not _has_support_cache(row))
    summary = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "num_rows": len(rows),
        "already_cached_rows": already_cached,
        "hydrated_rows": hydrated,
        "remaining_missing_rows": remaining,
        "complete": remaining == 0,
        "output_jsonl_sha256": sha256_file(output_path),
    }
    summary["provenance_updated"] = _update_provenance(
        provenance_json=provenance_path,
        output_jsonl=output_path,
        summary=summary,
    )
    summary["provenance_json"] = str(provenance_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hydrate cached support pairs for stage2 teacher JSONL rows.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--provenance-json", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    summary = hydrate_support_cache(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        provenance_json=args.provenance_json,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
