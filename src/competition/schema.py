from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PuzzlePair:
    input: str
    output: str


@dataclass(slots=True)
class PuzzleMetadata:
    official_family: str | None = None
    subtype: str | None = None
    family_scores: dict[str, float] = field(default_factory=dict)
    teacher_confidence: float | None = None
    program_signature: str | None = None
    difficulty: float | None = None
    source: str = "unknown"
    split: str = "unknown"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PuzzleExample:
    id: str
    raw_prompt: str
    official_instruction: str
    parsed_examples: list[PuzzlePair]
    query: str
    target_answer: str | None = None
    metadata: PuzzleMetadata = field(default_factory=PuzzleMetadata)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PuzzleExample":
        metadata_payload = dict(payload.get("metadata", {}))
        family_tags = metadata_payload.pop("family_tags", None)
        metadata_payload.pop("composition_key", None)
        if metadata_payload.get("official_family") is None and family_tags:
            metadata_payload["official_family"] = str(family_tags[0])
        return cls(
            id=str(payload["id"]),
            raw_prompt=payload.get("raw_prompt", ""),
            official_instruction=payload.get("official_instruction", ""),
            parsed_examples=[
                PuzzlePair(**pair)
                for pair in payload.get("parsed_examples", payload.get("train_pairs", []))
            ],
            query=payload.get("query", payload.get("query_input", "")),
            target_answer=payload.get("target_answer"),
            metadata=PuzzleMetadata(**metadata_payload),
        )
