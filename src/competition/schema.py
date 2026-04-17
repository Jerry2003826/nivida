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
    family_tags: list[str] = field(default_factory=list)
    family_scores: dict[str, float] = field(default_factory=dict)
    teacher_confidence: float | None = None
    program_signature: str | None = None
    composition_key: str | None = None
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
        family_tags = metadata_payload.get("family_tags")
        if metadata_payload.get("official_family") is None and family_tags:
            metadata_payload["official_family"] = str(family_tags[0])
        if not metadata_payload.get("family_tags") and metadata_payload.get("official_family"):
            tags = [str(metadata_payload["official_family"])]
            if metadata_payload.get("subtype"):
                tags.append(str(metadata_payload["subtype"]))
            metadata_payload["family_tags"] = tags
        if metadata_payload.get("composition_key") is None:
            metadata_payload["composition_key"] = metadata_payload.get("extras", {}).get("composition_key")
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
