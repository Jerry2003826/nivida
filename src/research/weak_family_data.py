from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

from src.common.io import load_jsonl, write_json, write_jsonl


GENERATOR_VERSION = "weak_family_rescue_v2_20260427"
DEFAULT_SEED = 20260427
RECIPE_FAMILIES: dict[str, set[str] | None] = {
    "mixed_answer_short": None,
    "equation_rescue": {"equation"},
    "bit_rescue": {"bit"},
    "eq_bit_rescue": {"equation", "bit"},
    "equation_rescue_v2": {"equation"},
    "bit_rescue_v2": {"bit"},
    "eq_bit_rescue_v2": {"equation", "bit"},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_text(value: Any) -> str:
    payload = value if isinstance(value, str) else str(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _family(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("official_family") or metadata.get("official_family") or "unknown")


def _extras(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    extras = metadata.get("extras")
    return dict(extras) if isinstance(extras, dict) else {}


def _target_answer(row: dict[str, Any]) -> str:
    for key in ("target_answer", "answer", "expected_answer"):
        value = row.get(key)
        if value is not None:
            return str(value)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    value = metadata.get("target_answer")
    return "" if value is None else str(value)


def _equation_risk_features(row: dict[str, Any], extras: dict[str, Any]) -> dict[str, Any]:
    risk = str(extras.get("template_risk_class") or row.get("risk_class") or "")
    query_key_seen = bool(extras.get("query_key_seen", "unseen_key" not in risk))
    return {
        "equation_target_expressible": bool(
            extras.get("template_target_expressible", row.get("target_expressible", False))
        ),
        "equation_support_consistency": extras.get(
            "support_consistency",
            "stable" if "support_stable" in risk or risk == "low_risk_support_stable" else "unknown",
        ),
        "equation_query_key_seen": query_key_seen,
        "equation_operator_gap": bool(extras.get("operator_gap", "operator_gap" in risk)),
    }


def _bit_operator_family(row: dict[str, Any], extras: dict[str, Any] | None = None) -> str:
    extras = extras or {}
    explicit = str(
        extras.get("bit_operator_family")
        or row.get("top_operator_family")
        or row.get("oracle_operator_family")
        or ""
    )
    if explicit:
        return explicit
    subtype = str(row.get("subtype", "")).lower()
    if "affine" in subtype or "xor" in subtype:
        return "affine_gf2"
    if "rotate" in subtype or "rotation" in subtype:
        return "rotation"
    if "reverse" in subtype or "reversal" in subtype:
        return "reversal"
    if "nibble" in subtype or "byte" in subtype:
        return "nibble_byte_transform"
    if "permutation" in subtype or "permute" in subtype:
        return "plain_permutation"
    if "boolean" in subtype or "expr" in subtype:
        return "boolean_template"
    return subtype or "unknown"


def _bit_risk_features(row: dict[str, Any], extras: dict[str, Any]) -> dict[str, Any]:
    support_coverage = float(extras.get("support_coverage", 0.0) or 0.0)
    leave_one_out = bool(
        extras.get("leave_one_out_stable", extras.get("solver_verifiable", False) and support_coverage >= 1.0)
    )
    return {
        "bit_operator_family": _bit_operator_family(row, extras),
        "bit_leave_one_out_stable": leave_one_out,
        "bit_support_coverage": support_coverage,
        "bit_expression_complexity": extras.get("expression_complexity", extras.get("solver_complexity", "")),
        "bit_oracle_rank": extras.get("oracle_rank", ""),
        "bit_top_oracle_hamming_distance": extras.get("top_oracle_hamming_distance", ""),
    }


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


def _annotate_row(
    row: dict[str, Any],
    *,
    recipe: str,
    source_path: Path,
    source_sha256: str,
    seed: int,
) -> dict[str, Any]:
    output = dict(row)
    metadata = dict(output.get("metadata") or {})
    extras = dict(metadata.get("extras") or {})
    family = _family(row)
    risk_class = str(extras.get("template_risk_class") or output.get("risk_class") or "")
    short_trace_allowed = _is_safe_short_trace(row)
    if family == "equation":
        extras.update(_equation_risk_features(row, extras))
    if family == "bit":
        extras.update(_bit_risk_features(row, extras))
    extras.update(
        {
            "research_recipe": recipe,
            "research_source_file": source_path.as_posix(),
            "research_source_sha256": source_sha256,
            "research_generator_version": GENERATOR_VERSION,
            "research_seed": int(seed),
            "solver_risk_class": risk_class,
            "target_expressible": bool(
                extras.get("template_target_expressible", output.get("target_expressible", False))
            ),
            "short_trace_allowed": bool(short_trace_allowed),
            "answer_hash": _hash_text(_target_answer(row)),
        }
    )
    metadata["extras"] = extras
    output["metadata"] = metadata
    output["research_recipe"] = recipe
    output["research_provenance"] = {
        "generator_version": GENERATOR_VERSION,
        "seed": int(seed),
        "source_file": source_path.as_posix(),
        "source_sha256": source_sha256,
        "answer_hash": extras["answer_hash"],
        "family": family,
        "subtype": str(output.get("subtype", metadata.get("subtype", ""))),
        "risk_class": risk_class,
        "trace_style": str(output.get("trace_style", "")),
        "short_trace_allowed": bool(short_trace_allowed),
    }
    return output


def _select_rows(
    *,
    recipe: str,
    answer_rows: list[dict[str, Any]],
    short_rows: list[dict[str, Any]],
    answer_path: Path,
    answer_sha256: str,
    short_path: Path,
    short_sha256: str,
    seed: int,
) -> list[dict[str, Any]]:
    families = RECIPE_FAMILIES[recipe]
    selected: list[dict[str, Any]] = []
    for row in answer_rows:
        if families is None or _family(row) in families:
            selected.append(
                _annotate_row(
                    row,
                    recipe=recipe,
                    source_path=answer_path,
                    source_sha256=answer_sha256,
                    seed=seed,
                )
            )
    for row in short_rows:
        if (families is None or _family(row) in families) and _is_safe_short_trace(row):
            selected.append(
                _annotate_row(
                    row,
                    recipe=recipe,
                    source_path=short_path,
                    source_sha256=short_sha256,
                    seed=seed,
                )
            )
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
    seed: int = DEFAULT_SEED,
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
    input_hashes = {
        "answer_train": _sha256(answer_train_path),
        "answer_valid": _sha256(answer_valid_path),
        "short_train": _sha256(short_train_path),
        "short_valid": _sha256(short_valid_path),
    }
    summary: dict[str, Any] = {
        "status": "done",
        "timestamp_utc": _utc_now(),
        "generator_version": GENERATOR_VERSION,
        "seed": int(seed),
        "inputs": {
            "answer_train": {"path": answer_train_path.as_posix(), "sha256": input_hashes["answer_train"]},
            "answer_valid": {"path": answer_valid_path.as_posix(), "sha256": input_hashes["answer_valid"]},
            "short_train": {"path": short_train_path.as_posix(), "sha256": input_hashes["short_train"]},
            "short_valid": {"path": short_valid_path.as_posix(), "sha256": input_hashes["short_valid"]},
        },
        "recipes": {},
    }

    for recipe in recipe_names:
        train_rows = _select_rows(
            recipe=recipe,
            answer_rows=train_answer_rows,
            short_rows=train_short_rows,
            answer_path=answer_train_path,
            answer_sha256=input_hashes["answer_train"],
            short_path=short_train_path,
            short_sha256=input_hashes["short_train"],
            seed=seed,
        )
        valid_rows = _select_rows(
            recipe=recipe,
            answer_rows=valid_answer_rows,
            short_rows=valid_short_rows,
            answer_path=answer_valid_path,
            answer_sha256=input_hashes["answer_valid"],
            short_path=short_valid_path,
            short_sha256=input_hashes["short_valid"],
            seed=seed,
        )
        train_out = out / f"{recipe}_train.jsonl"
        valid_out = out / f"{recipe}_valid.jsonl"
        write_jsonl(train_out, train_rows)
        write_jsonl(valid_out, valid_rows)
        train_provenance = _write_output_provenance(
            train_out,
            recipe=recipe,
            split="train",
            rows=train_rows,
            input_hashes=input_hashes,
            seed=seed,
        )
        valid_provenance = _write_output_provenance(
            valid_out,
            recipe=recipe,
            split="valid",
            rows=valid_rows,
            input_hashes=input_hashes,
            seed=seed,
        )
        summary["recipes"][recipe] = {
            "families": None if RECIPE_FAMILIES[recipe] is None else sorted(RECIPE_FAMILIES[recipe] or []),
            "train_output": train_out.as_posix(),
            "valid_output": valid_out.as_posix(),
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "train_sha256": _sha256(train_out),
            "valid_sha256": _sha256(valid_out),
            "train_provenance": train_provenance.as_posix(),
            "valid_provenance": valid_provenance.as_posix(),
            "train_risk_counts": _risk_counts(train_rows),
            "valid_risk_counts": _risk_counts(valid_rows),
        }
    write_json(out / "research_rescue_data_summary.json", summary)
    return summary


def _risk_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        provenance = row.get("research_provenance") if isinstance(row.get("research_provenance"), dict) else {}
        risk = str(provenance.get("risk_class") or "unknown")
        counts[risk] = counts.get(risk, 0) + 1
    return dict(sorted(counts.items()))


def _write_output_provenance(
    output_path: Path,
    *,
    recipe: str,
    split: str,
    rows: list[dict[str, Any]],
    input_hashes: dict[str, str],
    seed: int,
) -> Path:
    provenance_path = output_path.with_suffix(output_path.suffix + ".provenance.json")
    payload = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "created_utc": _utc_now(),
        "recipe": recipe,
        "split": split,
        "seed": int(seed),
        "output_path": output_path.as_posix(),
        "output_sha256": _sha256(output_path),
        "row_count": len(rows),
        "input_hashes": input_hashes,
        "families": None if RECIPE_FAMILIES[recipe] is None else sorted(RECIPE_FAMILIES[recipe] or []),
        "risk_counts": _risk_counts(rows),
        "answer_hashes_sha256": _hash_text(
            "\n".join(
                sorted(
                    str((row.get("research_provenance") or {}).get("answer_hash", ""))
                    for row in rows
                )
            )
        ),
    }
    write_json(provenance_path, payload)
    return provenance_path
