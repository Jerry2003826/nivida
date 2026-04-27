from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

from src.common.io import load_jsonl, write_json, write_jsonl


RECIPE_FAMILIES: dict[str, set[str] | None] = {
    "mixed_answer_short": None,
    "equation_rescue": {"equation"},
    "bit_rescue": {"bit"},
    "eq_bit_rescue": {"equation", "bit"},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _family(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("official_family") or metadata.get("official_family") or "unknown")


def _extras(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    extras = metadata.get("extras")
    return dict(extras) if isinstance(extras, dict) else {}


def _is_safe_short_trace(row: dict[str, Any]) -> bool:
    if str(row.get("trace_style", "")) != "short_trace":
        return False
    family = _family(row)
    extras = _extras(row)
    if family == "equation" and str(row.get("subtype", "")) == "equation_template":
        return str(extras.get("template_risk_class", "")) == "low_risk_support_stable"
    if family == "bit":
        return bool(extras.get("solver_verifiable", False)) and float(extras.get("support_coverage", 0.0) or 0.0) >= 1.0
    return True


def _annotate_row(row: dict[str, Any], *, recipe: str, source_path: Path) -> dict[str, Any]:
    output = dict(row)
    metadata = dict(output.get("metadata") or {})
    extras = dict(metadata.get("extras") or {})
    extras.update(
        {
            "research_recipe": recipe,
            "research_source_file": source_path.as_posix(),
            "solver_risk_class": extras.get("template_risk_class") or output.get("risk_class") or "",
            "target_expressible": bool(
                extras.get("template_target_expressible", output.get("target_expressible", False))
            ),
        }
    )
    metadata["extras"] = extras
    output["metadata"] = metadata
    output["research_recipe"] = recipe
    return output


def _select_rows(
    *,
    recipe: str,
    answer_rows: list[dict[str, Any]],
    short_rows: list[dict[str, Any]],
    answer_path: Path,
    short_path: Path,
) -> list[dict[str, Any]]:
    families = RECIPE_FAMILIES[recipe]
    selected: list[dict[str, Any]] = []
    for row in answer_rows:
        if families is None or _family(row) in families:
            selected.append(_annotate_row(row, recipe=recipe, source_path=answer_path))
    for row in short_rows:
        if (families is None or _family(row) in families) and _is_safe_short_trace(row):
            selected.append(_annotate_row(row, recipe=recipe, source_path=short_path))
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in selected:
        key = (str(row.get("id", "")), str(row.get("trace_style", "")))
        deduped.setdefault(key, row)
    return list(deduped.values())


def build_research_rescue_data(
    *,
    answer_train: str | Path,
    answer_valid: str | Path,
    short_train: str | Path,
    short_valid: str | Path,
    out_dir: str | Path,
    recipes: list[str] | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    answer_train_path = Path(answer_train)
    answer_valid_path = Path(answer_valid)
    short_train_path = Path(short_train)
    short_valid_path = Path(short_valid)
    recipe_names = recipes or list(RECIPE_FAMILIES)
    for recipe in recipe_names:
        if recipe not in RECIPE_FAMILIES:
            raise ValueError(f"Unknown research data recipe: {recipe}")

    train_answer_rows = load_jsonl(answer_train_path)
    valid_answer_rows = load_jsonl(answer_valid_path)
    train_short_rows = load_jsonl(short_train_path)
    valid_short_rows = load_jsonl(short_valid_path)
    summary: dict[str, Any] = {
        "status": "done",
        "timestamp_utc": _utc_now(),
        "inputs": {
            "answer_train": {"path": str(answer_train_path), "sha256": _sha256(answer_train_path)},
            "answer_valid": {"path": str(answer_valid_path), "sha256": _sha256(answer_valid_path)},
            "short_train": {"path": str(short_train_path), "sha256": _sha256(short_train_path)},
            "short_valid": {"path": str(short_valid_path), "sha256": _sha256(short_valid_path)},
        },
        "recipes": {},
    }

    for recipe in recipe_names:
        train_rows = _select_rows(
            recipe=recipe,
            answer_rows=train_answer_rows,
            short_rows=train_short_rows,
            answer_path=answer_train_path,
            short_path=short_train_path,
        )
        valid_rows = _select_rows(
            recipe=recipe,
            answer_rows=valid_answer_rows,
            short_rows=valid_short_rows,
            answer_path=answer_valid_path,
            short_path=short_valid_path,
        )
        train_out = out / f"{recipe}_train.jsonl"
        valid_out = out / f"{recipe}_valid.jsonl"
        write_jsonl(train_out, train_rows)
        write_jsonl(valid_out, valid_rows)
        summary["recipes"][recipe] = {
            "families": None if RECIPE_FAMILIES[recipe] is None else sorted(RECIPE_FAMILIES[recipe] or []),
            "train_output": str(train_out),
            "valid_output": str(valid_out),
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "train_sha256": _sha256(train_out),
            "valid_sha256": _sha256(valid_out),
        }
    write_json(out / "research_rescue_data_summary.json", summary)
    return summary

