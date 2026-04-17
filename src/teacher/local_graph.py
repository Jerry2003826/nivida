from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.competition.schema import PuzzleExample
from src.teacher.chain_search import ChainSearchEngine


@dataclass(slots=True)
class LocalNode:
    node_id: str
    op_name: str
    family: str
    weight: float
    params: dict[str, Any]


@dataclass(slots=True)
class LocalEdge:
    source: str
    target: str
    weight: float


@dataclass(slots=True)
class LocalCandidateGraph:
    nodes: list[LocalNode]
    edges: list[LocalEdge]
    metadata: dict[str, Any]


def build_local_candidate_graph(example: PuzzleExample, *, top_k: int = 5) -> LocalCandidateGraph:
    engine = ChainSearchEngine()
    candidates = engine.solve_example(example, top_k=top_k)
    nodes: list[LocalNode] = []
    edges: list[LocalEdge] = []
    for chain_index, candidate in enumerate(candidates):
        previous_node_id: str | None = None
        for step_index, step in enumerate(candidate.steps):
            node_id = f"chain_{chain_index}_step_{step_index}"
            nodes.append(LocalNode(node_id=node_id, op_name=step.op_name, family=step.family, weight=step.step_score, params=step.params))
            if previous_node_id is not None:
                edges.append(LocalEdge(source=previous_node_id, target=node_id, weight=candidate.score))
            previous_node_id = node_id
    return LocalCandidateGraph(nodes=nodes, edges=edges, metadata={"example_id": example.id, "num_candidates": len(candidates)})
