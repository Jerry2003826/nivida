from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.teacher.chain_search import ChainSearchEngine


ANNOTATION_ENGINE = "ChainSearchEngine"
STAGE2_ANNOTATION_BEAM_WIDTH = 8
STAGE2_ANNOTATION_MAX_DEPTH = 2
STAGE2_ANNOTATION_TOP_K = 2

_OPERATOR_PRIORITY_CASES: tuple[tuple[str | None, str | None, str | None], ...] = (
    (None, None, None),
    ("bit", None, None),
    ("bit", "rotate", None),
    ("bit", "bit_rotate", None),
    ("bit", "mask_logic", None),
    ("bit", "bit_xor_mask", None),
    ("bit", "nibble_permute", None),
    ("bit", "bit_nibble", None),
    ("bit", "binary_affine", None),
    ("bit", "bit_affine", None),
    ("bit", "bit_permutation", None),
    ("cipher", None, None),
    ("cipher", "token_substitution", None),
    ("cipher", "cipher_token_sub", None),
    ("cipher", "char_substitution", None),
    ("cipher", "caesar_affine", None),
    ("cipher", "cipher_char_sub", None),
    ("cipher", "substitution_permutation", None),
    ("cipher", "cipher_perm", None),
    ("cipher", "partial_map_completion", None),
    ("cipher", "cipher_vocab", None),
    ("equation", None, "symbolic"),
    ("equation", "equation_delete", "symbolic"),
    ("equation", "equation_template", "symbolic"),
    ("equation", "equation_position", "symbolic"),
    ("equation", None, "numeric"),
    ("unit", None, None),
    ("unit", "convert", None),
    ("unit", "unit_convert", None),
    ("numeral", None, None),
    ("numeral", "roman", None),
    ("numeral", "numeral_roman", None),
    ("gravity", None, None),
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_git_commit(*, repo_root: str | Path | None = None) -> str | None:
    cwd = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
        ).strip()
    except Exception:
        return None


def operator_priority_manifest() -> dict[str, list[str]]:
    engine = ChainSearchEngine(
        beam_width=STAGE2_ANNOTATION_BEAM_WIDTH,
        max_depth=STAGE2_ANNOTATION_MAX_DEPTH,
    )
    manifest: dict[str, list[str]] = {}
    for family, subtype, equation_mode in _OPERATOR_PRIORITY_CASES:
        key = (
            f"family={family or 'None'}|"
            f"subtype={subtype or 'None'}|"
            f"equation_mode={equation_mode or 'None'}"
        )
        manifest[key] = engine._prioritized_op_names(family, subtype, equation_mode)
    return manifest


def operator_priority_hash() -> str:
    return sha256_text(
        json.dumps(
            operator_priority_manifest(),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def expected_stage2_annotation_settings() -> dict[str, Any]:
    return {
        "annotation_engine": ANNOTATION_ENGINE,
        "beam_width": STAGE2_ANNOTATION_BEAM_WIDTH,
        "max_depth": STAGE2_ANNOTATION_MAX_DEPTH,
        "top_k": STAGE2_ANNOTATION_TOP_K,
        "operator_priority_hash": operator_priority_hash(),
    }


def build_stage2_annotation_provenance(
    *,
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    input_file = Path(input_path)
    output_file = Path(output_path)
    return {
        **expected_stage2_annotation_settings(),
        "code_commit": current_git_commit(),
        "input_jsonl_sha256": sha256_file(input_file),
        "output_jsonl_sha256": sha256_file(output_file),
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def stage2_provenance_matches_local(
    provenance: Mapping[str, Any],
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    required = expected_stage2_annotation_settings()
    found = {key: provenance.get(key) for key in required}
    return found == required, required, found
