from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PuzzlePair:
    input: str
    output: str


@dataclass(slots=True)
class PuzzleMetadata:
    source: str = "unknown"
    split: str = "unknown"
    family_tags: list[str] = field(default_factory=list)
    family_scores: dict[str, float] = field(default_factory=dict)
    difficulty: float | None = None
    composition_key: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PuzzleExample:
    id: str
    raw_prompt: str
    train_pairs: list[PuzzlePair]
    query_input: str
    target_answer: str | None = None
    metadata: PuzzleMetadata = field(default_factory=PuzzleMetadata)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PuzzleExample":
        return cls(
            id=str(payload["id"]),
            raw_prompt=payload.get("raw_prompt", ""),
            train_pairs=[PuzzlePair(**pair) for pair in payload.get("train_pairs", [])],
            query_input=payload.get("query_input", ""),
            target_answer=payload.get("target_answer"),
            metadata=PuzzleMetadata(**payload.get("metadata", {})),
        )
