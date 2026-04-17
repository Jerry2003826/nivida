from __future__ import annotations

import argparse

from src.common.io import read_json
from src.teacher.global_rule_graph import GlobalRuleGraph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reusable global rule graph artifact from benchmark/baseline records.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/global_rule_graph.json")
    args = parser.parse_args()

    payload = read_json(args.input)
    graph = GlobalRuleGraph.from_records(payload.get("records", []))
    graph.save(args.output)


if __name__ == "__main__":
    main()
