from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.io import load_jsonl, write_json, write_jsonl
from src.competition.parser import parse_competition_file
from src.competition.split_builder import build_splits
from src.teacher.chain_search import ChainSearchEngine
from src.teacher.family_tagger import apply_family_tags
from src.teacher.program_signature import annotate_example_from_candidates
from src.teacher.stage2_annotation_provenance import (
    STAGE2_ANNOTATION_BEAM_WIDTH,
    STAGE2_ANNOTATION_MAX_DEPTH,
    STAGE2_ANNOTATION_TOP_K,
    build_stage2_annotation_provenance,
    build_stage2_subset_provenance,
    sha256_file,
)
from src.student.sft_dataset_builder import export_split_subset


DEFAULT_RAW_TRAIN = Path("../data/official_kaggle/train.csv")
DEFAULT_PROCESSED_DIR = Path("../data/processed")
DEFAULT_SPLITS_DIR = Path("../data/splits/official")
DEFAULT_SUMMARY_OUTPUT = Path("data/processed/rebuild_stage2_teacher_inputs_summary.json")


@dataclass(frozen=True)
class SubsetSpec:
    name: str
    output_name: str
    split_name: str
    split_role: str
    exclude_split_name: str | None = None
    exclude_split_role: str | None = None

    def selection_payload(self) -> dict[str, str | None]:
        return {
            "split_name": self.split_name,
            "split_role": self.split_role,
            "exclude_split_name": self.exclude_split_name,
            "exclude_split_role": self.exclude_split_role,
        }


STAGE2_SUBSETS: tuple[SubsetSpec, ...] = (
    SubsetSpec(
        name="stage2_train_no_hard_valid",
        output_name="stage2_official_train_no_hard_valid.jsonl",
        split_name="rule_novelty_all",
        split_role="train",
        exclude_split_name="hard_triad_rule_novelty",
        exclude_split_role="valid",
    ),
    SubsetSpec(
        name="stage2_valid_hard_triad",
        output_name="stage2_official_valid_hard_triad.jsonl",
        split_name="hard_triad_rule_novelty",
        split_role="valid",
    ),
)


def _annotate_official(raw_train: Path) -> list[Any]:
    examples = parse_competition_file(raw_train, source="kaggle", split="train")
    examples = apply_family_tags(examples)
    engine = ChainSearchEngine(
        beam_width=STAGE2_ANNOTATION_BEAM_WIDTH,
        max_depth=STAGE2_ANNOTATION_MAX_DEPTH,
    )
    for example in examples:
        candidates = engine.solve_example(example, top_k=STAGE2_ANNOTATION_TOP_K)
        annotate_example_from_candidates(example, candidates)
    return examples


def rebuild_stage2_teacher_inputs(
    *,
    raw_train: str | Path = DEFAULT_RAW_TRAIN,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    summary_output: str | Path = DEFAULT_SUMMARY_OUTPUT,
    seed: int = 42,
    rule_novelty_valid_ratio: float = 0.15,
    hard_triad_valid_ratio: float = 0.15,
) -> dict[str, Any]:
    raw_path = Path(raw_train)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing official raw train CSV: {raw_path}")

    processed = Path(processed_dir)
    splits = Path(splits_dir)
    official_output = processed / "official_train_tagged.jsonl"
    split_output = splits / "splits.json"

    examples = _annotate_official(raw_path)
    write_jsonl(official_output, [example.to_dict() for example in examples])
    write_json(
        f"{official_output}.provenance.json",
        build_stage2_annotation_provenance(
            input_path=raw_path,
            output_path=official_output,
        ),
    )

    split_payload = build_splits(
        examples,
        rule_novelty_valid_ratio=rule_novelty_valid_ratio,
        hard_triad_valid_ratio=hard_triad_valid_ratio,
        seed=seed,
    )
    write_json(split_output, split_payload)

    subset_summaries: dict[str, Any] = {}
    subset_ids: dict[str, set[str]] = {}
    for spec in STAGE2_SUBSETS:
        output_path = processed / spec.output_name
        rows = export_split_subset(
            examples,
            split_file=split_output,
            split_name=spec.split_name,
            split_role=spec.split_role,
            exclude_split_file=split_output if spec.exclude_split_name else None,
            exclude_split_name=spec.exclude_split_name,
            exclude_split_role=spec.exclude_split_role,
        )
        write_jsonl(output_path, rows)
        write_json(
            f"{output_path}.provenance.json",
            build_stage2_subset_provenance(
                input_path=official_output,
                output_path=output_path,
                split_file=split_output,
                selection=spec.selection_payload(),
                raw_input_path=raw_path,
            ),
        )
        ids = {str(row.get("id")) for row in rows}
        subset_ids[spec.name] = ids
        subset_summaries[spec.name] = {
            "path": str(output_path),
            "num_rows": len(rows),
            "sha256": sha256_file(output_path),
            "provenance_path": f"{output_path}.provenance.json",
            "selection": spec.selection_payload(),
        }

    train_ids = subset_ids["stage2_train_no_hard_valid"]
    valid_ids = subset_ids["stage2_valid_hard_triad"]
    summary = {
        "raw_train": {
            "path": str(raw_path),
            "sha256": sha256_file(raw_path),
        },
        "official_train_tagged": {
            "path": str(official_output),
            "num_rows": len(load_jsonl(official_output)),
            "sha256": sha256_file(official_output),
            "provenance_path": f"{official_output}.provenance.json",
        },
        "splits": {
            "path": str(split_output),
            "sha256": sha256_file(split_output),
            "seed": seed,
            "rule_novelty_valid_ratio": rule_novelty_valid_ratio,
            "hard_triad_valid_ratio": hard_triad_valid_ratio,
        },
        "subsets": subset_summaries,
        "train_valid_overlap": {
            "count": len(train_ids & valid_ids),
            "ids": sorted(train_ids & valid_ids)[:50],
        },
    }
    write_json(summary_output, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild current-code stage2 teacher JSONL inputs and provenance."
    )
    parser.add_argument("--raw-train", type=Path, default=DEFAULT_RAW_TRAIN)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rule-novelty-valid-ratio", type=float, default=0.15)
    parser.add_argument("--hard-triad-valid-ratio", type=float, default=0.15)
    args = parser.parse_args(argv)

    summary = rebuild_stage2_teacher_inputs(
        raw_train=args.raw_train,
        processed_dir=args.processed_dir,
        splits_dir=args.splits_dir,
        summary_output=args.summary_output,
        seed=args.seed,
        rule_novelty_valid_ratio=args.rule_novelty_valid_ratio,
        hard_triad_valid_ratio=args.hard_triad_valid_ratio,
    )
    print(
        {
            "official_rows": summary["official_train_tagged"]["num_rows"],
            "stage2_train_rows": summary["subsets"]["stage2_train_no_hard_valid"]["num_rows"],
            "stage2_valid_rows": summary["subsets"]["stage2_valid_hard_triad"]["num_rows"],
            "train_valid_overlap": summary["train_valid_overlap"]["count"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
