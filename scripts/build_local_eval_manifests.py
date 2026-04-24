from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LOCAL_REPO = REPO_ROOT / "nemotron_local_repo"
if LOCAL_REPO.is_dir() and str(LOCAL_REPO) not in sys.path:
    sys.path.insert(0, str(LOCAL_REPO))

from src.competition.official_prompts import detect_official_family  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_existing_input(path: Path) -> Path:
    if path.exists():
        return path
    workspace_sibling = REPO_ROOT.parent / path
    if workspace_sibling.exists():
        return workspace_sibling
    return path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                import ast

                obj = ast.literal_eval(raw)
                return obj if isinstance(obj, dict) else {}
            except (SyntaxError, ValueError):
                return {}
    return {}


def family(row: dict[str, Any]) -> str:
    meta = metadata(row)
    value = row.get("official_family") or row.get("family") or meta.get("official_family")
    if value is None:
        prompt = str(row.get("raw_prompt") or row.get("prompt") or "")
        value = detect_official_family(prompt) if prompt else None
    return "unknown" if value is None else str(value)


def subtype(row: dict[str, Any]) -> str:
    meta = metadata(row)
    return str(row.get("subtype") or meta.get("subtype") or "unknown")


def row_id(row: dict[str, Any], fallback: int) -> str:
    return str(row.get("id") or row.get("sample_id") or row.get("uid") or fallback)


def duplicate_ids(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row_id(row, idx) for idx, row in enumerate(rows))
    return {rid: count for rid, count in sorted(counts.items()) if count > 1}


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        rid = row_id(row, idx)
        if rid in seen:
            continue
        seen.add(rid)
        out.append(row)
    return out


def sample_by_family(rows: list[dict[str, Any]], per_family: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[family(row)].append(row)
    sampled: list[dict[str, Any]] = []
    for fam in sorted(buckets):
        pool = list(buckets[fam])
        rng.shuffle(pool)
        sampled.extend(pool[:per_family])
    sampled.sort(key=lambda row: row_id(row, 0))
    return sampled


def sample_by_subtype(rows: list[dict[str, Any]], per_subtype: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[f"{family(row)}:{subtype(row)}"].append(row)
    sampled_by_id: dict[str, dict[str, Any]] = {}
    for key in sorted(buckets):
        pool = list(buckets[key])
        rng.shuffle(pool)
        for row in pool[:per_subtype]:
            sampled_by_id[row_id(row, len(sampled_by_id))] = row
    sampled = list(sampled_by_id.values())
    sampled.sort(key=lambda row: row_id(row, 0))
    return sampled


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fam = Counter(family(row) for row in rows)
    sub = Counter(f"{family(row)}:{subtype(row)}" for row in rows)
    dupes = duplicate_ids(rows)
    return {
        "num_rows": len(rows),
        "num_duplicate_ids": sum(count - 1 for count in dupes.values()),
        "duplicate_ids": dupes,
        "families": dict(fam.most_common()),
        "subtypes": dict(sub.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local labeled eval manifests for cheap GPU inference.")
    parser.add_argument("--proxy-all", type=Path, default=Path("data/processed/proxy_all_family_valid.jsonl"))
    parser.add_argument("--hard", type=Path, default=Path("data/processed/stage2_official_valid_hard_triad.jsonl"))
    parser.add_argument("--distill-valid", type=Path, default=Path("data/processed/stage2_distill_valid.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/local_eval_manifests"))
    parser.add_argument("--seed", type=int, default=20260424)
    args = parser.parse_args()

    proxy = load_jsonl(resolve_existing_input(args.proxy_all))
    hard = load_jsonl(resolve_existing_input(args.hard))
    distill_valid = load_jsonl(resolve_existing_input(args.distill_valid))
    combined = dedupe_rows(proxy + hard)

    manifests: dict[str, list[dict[str, Any]]] = {
        "proxy_all_full": proxy,
        "hard_triad_full": hard,
        "distill_valid_full": distill_valid,
        "proxy_all_balanced_32pf": sample_by_family(proxy, 32, args.seed),
        "proxy_all_balanced_64pf": sample_by_family(proxy, 64, args.seed),
        "combined_balanced_48pf": sample_by_family(combined, 48, args.seed),
        "combined_subtype_24ps": sample_by_subtype(combined, 24, args.seed),
        "smoke_6pf": sample_by_family(combined, 6, args.seed),
    }

    summary: dict[str, Any] = {}
    for name, rows in manifests.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        summary[name] = {"path": str(path), **summarize(rows)}
        if summary[name]["num_duplicate_ids"]:
            raise SystemExit(
                f"manifest {name} contains duplicate ids: "
                f"{summary[name]['duplicate_ids']}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_lines = [
        "# Local Eval Manifests",
        "",
        "Use these labeled JSONL files as cloud inference inputs. Score predictions locally with `scripts/evaluate_predictions_exact.py`.",
        "",
        "Default main manifest: `combined_balanced_48pf`. Auxiliary manifests: `proxy_all_balanced_64pf`, `hard_triad_full`.",
        "",
        "| manifest | rows | duplicate ids | families | path |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for name, info in summary.items():
        md_lines.append(
            f"| {name} | {info['num_rows']} | {info['num_duplicate_ids']} | "
            f"`{json.dumps(info['families'], ensure_ascii=False)}` | `{info['path']}` |"
        )
    (args.output_dir / "README.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "manifests": len(manifests)}, indent=2))


if __name__ == "__main__":
    main()
