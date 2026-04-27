from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PromptProfile:
    name: str
    suffix: str
    description: str


PROMPT_PROFILES: dict[str, PromptProfile] = {
    "chat_thinking": PromptProfile(
        name="chat_thinking",
        suffix="",
        description="Default Kaggle-runtime prompt; no extra instruction beyond the official guard.",
    ),
    "short_answer_biased": PromptProfile(
        name="short_answer_biased",
        suffix=(
            "\n\nSolve the transformation and keep the response concise. "
            "Put only the final result inside one \\boxed{}."
        ),
        description="Biases the model away from long traces while preserving the official boxed contract.",
    ),
    "format_strict": PromptProfile(
        name="format_strict",
        suffix=(
            "\n\nReturn exactly one final answer. Do not include alternatives. "
            "The last answer must be written as \\boxed{answer}."
        ),
        description="Adds a stronger single-boxed-answer formatting instruction.",
    ),
}


def materialize_prompt_profile_row(row: dict[str, Any], profile_name: str) -> dict[str, Any]:
    if profile_name not in PROMPT_PROFILES:
        raise ValueError(f"Unknown prompt profile: {profile_name}")
    profile = PROMPT_PROFILES[profile_name]
    output = dict(row)
    base_prompt = str(output.get("prompt", output.get("raw_prompt", "")))
    if profile.suffix and profile.suffix not in base_prompt:
        output["prompt"] = base_prompt.rstrip() + profile.suffix
    elif "prompt" not in output and "raw_prompt" in output:
        output["prompt"] = base_prompt
    metadata = output.get("metadata")
    metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}
    metadata_dict["prompt_profile"] = profile.name
    output["metadata"] = metadata_dict
    output["prompt_profile"] = profile.name
    return output


def profile_summary() -> dict[str, dict[str, str]]:
    return {
        name: {"suffix": profile.suffix, "description": profile.description}
        for name, profile in PROMPT_PROFILES.items()
    }

