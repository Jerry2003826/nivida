from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.common.io import read_json, write_json
from src.teacher.chain_search import CandidateChain


def _normalise_steps(row: dict[str, Any]) -> list[str]:
    steps = row.get("steps", [])
    if not steps:
        return []
    if isinstance(steps[0], dict):
        return [str(step.get("op_name", "")) for step in steps if step.get("op_name")]
    return [str(step) for step in steps if step]


@dataclass(slots=True)
class GlobalRuleGraph:
    node_weights: dict[str, float] = field(default_factory=dict)
    edge_weights: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    family_node_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    family_edge_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    step_position_weights: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_chains(cls, chains: Iterable[CandidateChain], *, family: str = "unknown") -> "GlobalRuleGraph":
        records = []
        for chain in chains:
            records.append({"family": family, "steps": [step.op_name for step in chain.steps]})
        return cls.from_records(records)

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]]) -> "GlobalRuleGraph":
        node_counts = Counter()
        edge_counts = Counter()
        family_node_counts: dict[str, Counter[str]] = {}
        family_edge_counts: dict[str, Counter[str]] = {}
        position_counts: dict[str, Counter[str]] = {}
        total_records = 0

        for row in records:
            steps = _normalise_steps(row)
            if not steps:
                continue
            family = str(row.get("family", "unknown"))
            total_records += 1
            family_node_counts.setdefault(family, Counter())
            family_edge_counts.setdefault(family, Counter())
            for position, op_name in enumerate(steps):
                node_counts[op_name] += 1
                family_node_counts[family][op_name] += 1
                position_counts.setdefault(str(position), Counter())[op_name] += 1
            for lhs, rhs in zip(steps, steps[1:]):
                edge_name = f"{lhs}->{rhs}"
                edge_counts[edge_name] += 1
                family_edge_counts[family][edge_name] += 1

        node_weights = {name: count / max(1, total_records) for name, count in node_counts.items()}
        edge_total = sum(edge_counts.values()) or 1
        edge_weights = {name: count / edge_total for name, count in edge_counts.items()}
        family_node_weights = {
            family: {
                name: count / max(1, sum(counter.values()))
                for name, count in counter.items()
            }
            for family, counter in family_node_counts.items()
        }
        family_edge_weights = {
            family: {
                name: count / max(1, sum(counter.values()))
                for name, count in counter.items()
            }
            for family, counter in family_edge_counts.items()
        }
        step_position_weights = {
            position: {
                name: count / max(1, sum(counter.values()))
                for name, count in counter.items()
            }
            for position, counter in position_counts.items()
        }
        return cls(
            node_weights=node_weights,
            edge_weights=edge_weights,
            counts=dict(node_counts),
            family_node_weights=family_node_weights,
            family_edge_weights=family_edge_weights,
            step_position_weights=step_position_weights,
        )

    def update(self, records: Iterable[dict[str, Any]]) -> None:
        updated = GlobalRuleGraph.from_records(records)
        for name, value in updated.node_weights.items():
            self.node_weights[name] = (self.node_weights.get(name, 0.0) + value) / 2.0
        for name, value in updated.edge_weights.items():
            self.edge_weights[name] = (self.edge_weights.get(name, 0.0) + value) / 2.0
        for name, value in updated.counts.items():
            self.counts[name] = self.counts.get(name, 0) + value
        for family, weights in updated.family_node_weights.items():
            bucket = self.family_node_weights.setdefault(family, {})
            for name, value in weights.items():
                bucket[name] = (bucket.get(name, 0.0) + value) / 2.0
        for family, weights in updated.family_edge_weights.items():
            bucket = self.family_edge_weights.setdefault(family, {})
            for name, value in weights.items():
                bucket[name] = (bucket.get(name, 0.0) + value) / 2.0
        for position, weights in updated.step_position_weights.items():
            bucket = self.step_position_weights.setdefault(position, {})
            for name, value in weights.items():
                bucket[name] = (bucket.get(name, 0.0) + value) / 2.0

    def start_prior(self, op_name: str, family: str | None = None, position: int = 0) -> float:
        if family and family in self.family_node_weights:
            return self.family_node_weights[family].get(op_name, self.node_weights.get(op_name, 0.0))
        position_bucket = self.step_position_weights.get(str(position), {})
        return position_bucket.get(op_name, self.node_weights.get(op_name, 0.0))

    def transition_prior(self, lhs: str, rhs: str, family: str | None = None) -> float:
        edge_name = f"{lhs}->{rhs}"
        if family and family in self.family_edge_weights:
            return self.family_edge_weights[family].get(edge_name, self.edge_weights.get(edge_name, 0.0))
        return self.edge_weights.get(edge_name, 0.0)

    def save(self, path: str | Path) -> None:
        write_json(
            path,
            {
                "node_weights": self.node_weights,
                "edge_weights": self.edge_weights,
                "counts": self.counts,
                "family_node_weights": self.family_node_weights,
                "family_edge_weights": self.family_edge_weights,
                "step_position_weights": self.step_position_weights,
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "GlobalRuleGraph":
        payload = read_json(path)
        return cls(
            node_weights=payload.get("node_weights", {}),
            edge_weights=payload.get("edge_weights", {}),
            counts=payload.get("counts", {}),
            family_node_weights=payload.get("family_node_weights", {}),
            family_edge_weights=payload.get("family_edge_weights", {}),
            step_position_weights=payload.get("step_position_weights", {}),
        )
