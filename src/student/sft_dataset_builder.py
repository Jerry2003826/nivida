from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import warnings

from src.common.io import load_jsonl, read_json, read_yaml, write_json, write_jsonl
from src.competition.harness_prompt import (
    build_chat_thinking_prompt,
    wrap_as_thinking,
)
from src.competition.prompt_templates import (
    PROMPT_MODE_CHAT_THINKING,
    PROMPT_MODE_GENERIC,
    PROMPT_MODE_RAW_WITH_GUARD,
    build_competition_prompt,
)
from src.competition.schema import PuzzleExample
from src.teacher.chain_search import ChainSearchEngine
from src.teacher.error_taxonomy import classify_error
from src.teacher.family_tagger import apply_family_tags
from src.teacher.program_signature import annotate_example_from_candidates
from src.teacher.trace_compiler import compile_completion, compile_completion_body


TRACE_STYLES = {"answer_only", "short_trace", "token_trace"}
PROMPT_MODES = {
    PROMPT_MODE_RAW_WITH_GUARD,
    PROMPT_MODE_GENERIC,
    PROMPT_MODE_CHAT_THINKING,
}
HARD_TRIAD_FAMILIES = ("cipher", "bit", "equation")
BALANCED_FAMILY_ORDER = ("cipher", "bit", "equation", "numeral", "unit", "gravity")

REPAIR_ARTIFACT_REMEDIATION = (
    "Regenerate the artifact with a competition-correct evaluator "
    "(for example run_baseline or eval_competition_replica) so each record "
    "includes at least 'id' and 'competition_correct'."
)


class RepairArtifactSchemaError(ValueError):
    """Raised when a repair artifact does not satisfy the current baseline schema.

    The most common trigger is a legacy ``baseline_eval.json`` produced before the
    ``competition_correct`` field existed. Silently treating such artifacts as
    "all-failures" would contaminate stage3 training data, so this surfaces as a
    hard error instead.
    """


_REQUIRED_FAILURE_FIELDS: tuple[str, ...] = ("id", "competition_correct")


def _parse_input_paths(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _load_examples(input_paths: list[str]) -> list[PuzzleExample]:
    examples: list[PuzzleExample] = []
    for input_path in input_paths:
        examples.extend(PuzzleExample.from_dict(row) for row in load_jsonl(input_path))
    return examples


HARD_TRIAD_RESCUE_FAMILIES: frozenset[str] = frozenset({"equation", "cipher", "bit"})

# Fallback subtypes produced by :mod:`src.teacher.family_tagger` when the
# heuristic tagger cannot pin down a more specific subtype:
#
# - ``equation_symbolic`` is the ``_classify_equation`` default branch,
# - ``cipher_vocab`` is the ``_classify_cipher`` final fallback,
# - ``bit_affine`` is the ``_classify_bit`` final fallback.
#
# Samples sitting on these fallbacks are the prime candidates for a
# subtype-prior rescue because the chain-search op priority derived from them
# is the most generic (see ``ChainSearchEngine._prioritized_op_names``).
STAGE2_FALLBACK_SUBTYPES: dict[str, frozenset[str]] = {
    "equation": frozenset({"equation_symbolic"}),
    "cipher": frozenset({"cipher_vocab"}),
    "bit": frozenset({"bit_affine"}),
}


def _annotate_examples(
    examples: list[PuzzleExample],
    *,
    beam_width: int = 8,
    max_depth: int = 2,
    top_k: int = 2,
) -> list[PuzzleExample]:
    apply_family_tags(examples)
    engine = ChainSearchEngine(beam_width=beam_width, max_depth=max_depth)
    for example in examples:
        if example.metadata.program_signature and example.metadata.teacher_confidence is not None:
            continue
        candidates = engine.solve_example(example, top_k=top_k)
        annotate_example_from_candidates(example, candidates)
    return examples


def _step_names_from_example(example: PuzzleExample) -> set[str]:
    """Return the set of op names recorded on ``top_candidate_steps``.

    ``annotate_example_from_candidates`` writes ``top_candidate_steps`` as a
    list of op-name strings (see ``src.teacher.program_signature``). The
    synth generator writes the same shape. A missing or malformed value
    yields an empty set so callers can treat "no evidence" uniformly.
    """
    steps = _metadata_value(example, "top_candidate_steps", []) or []
    if not isinstance(steps, list):
        return set()
    return {str(step) for step in steps if step}


def _infer_subtype_hint_from_top_steps(example: PuzzleExample) -> str | None:
    """Infer a more specific subtype than the fallback based on top_candidate_steps.

    Only runs for hard-triad official families whose current subtype is one
    of :data:`STAGE2_FALLBACK_SUBTYPES` (i.e., the generic tagger fallback).
    The returned string matches :meth:`ChainSearchEngine._prioritized_op_names`
    expectations; a second-pass search run with this hint will receive a more
    targeted op priority list.

    The function is deliberately conservative:
    - never returns the current (fallback) subtype itself, to avoid a no-op
      transition that would still show up in diagnostics;
    - only emits a cipher hint when there is *evidence* that the generic
      vocabulary prior should be replaced (``vocabulary_cipher`` alone is
      itself token/vocab-level evidence and already matches the default
      op priority for ``cipher_vocab``, so no hint is produced).
    """
    family = str(_metadata_value(example, "official_family") or "")
    current_subtype = str(_metadata_value(example, "subtype") or "")

    if current_subtype not in STAGE2_FALLBACK_SUBTYPES.get(family, frozenset()):
        return None

    step_set = _step_names_from_example(example)
    if not step_set:
        return None

    if family == "equation":
        if "operator_template" in step_set:
            return "equation_template"
        if "position_transducer" in step_set:
            return "equation_position"
        if "delete_characters" in step_set:
            return "equation_delete"
        if step_set & {
            "add_constant",
            "multiply_constant",
            "affine_transform",
            "evaluate_expression",
            "binary_equation_rule",
        }:
            # Do NOT return "equation_numeric" here.
            # ChainSearchEngine._prioritized_op_names routes numeric equation
            # ops via ``equation_mode == "numeric"`` (decided purely by the
            # input pattern regex inside ``_equation_mode``), never via
            # ``subtype == "equation_numeric"``. Returning that string as a
            # hint would silently fall through to the default SYMBOLIC op
            # priority (``[position_transducer, operator_template,
            # delete_characters]``), so the rescue would cost a deeper
            # beam/depth search without actually changing the op prior.
            # Worse, any rescue that happened to promote under the deeper
            # search would be (mis-)attributed to the hint in
            # ``rescue_promoted_with_hint`` and corrupt the A/B signal.
            # Emit no hint until chain_search.py learns a dedicated
            # ``equation_numeric`` branch (deferred; not part of v1 scope).
            return None
        return None

    if family == "cipher":
        # Important:
        # Do not infer a new subtype from ``vocabulary_cipher`` alone. The
        # fallback ``cipher_vocab`` already maps to
        # ``[vocabulary_cipher, fixed_substitution]`` in the op priority
        # (see ChainSearchEngine._prioritized_op_names), so a
        # ``cipher_vocab -> cipher_vocab``-shaped hint would be a silent
        # no-op and would still pollute diagnostics. Only emit a hint when
        # we can see *substitution + permutation* evidence (cipher_perm)
        # or a Caesar-shift family signature (cipher_char_sub).
        has_perm = bool(step_set & {"reverse_tokens", "sort_tokens"})
        has_substitution = bool(step_set & {"vocabulary_cipher", "fixed_substitution"})
        if has_perm and has_substitution:
            return "cipher_perm"
        if step_set & {"caesar_shift"}:
            return "cipher_char_sub"
        return None

    if family == "bit":
        if step_set & {"binary_rotate_left", "binary_rotate_right"}:
            return "bit_rotate"
        if step_set & {
            "binary_xor_mask",
            "binary_and_mask",
            "binary_or_mask",
            "bitwise_xor_constant",
            "bitwise_and_constant",
            "bitwise_or_constant",
        }:
            return "bit_xor_mask"
        if step_set & {"swap_nibbles", "binary_nibble_map"}:
            return "bit_nibble"
        if step_set & {"binary_permutation", "reverse_bits"}:
            return "bit_permutation"
        if "binary_affine_transform" in step_set:
            # Current subtype is already ``bit_affine``; returning it would
            # be a no-op hint, so we keep it as ``None``. Kept here as a
            # comment to make the mapping explicit and to aid v2 planning.
            return None
        return None

    return None


def _attach_search_subtype_hints(examples: list[PuzzleExample]) -> dict[str, Any]:
    """Write ``search_subtype_hint`` into ``extras`` for eligible samples.

    This is a pre-rescue read-only analysis pass. It never mutates
    ``example.metadata.subtype`` itself; the temporary override (required
    because ``ChainSearchEngine.solve_example`` reads
    ``example.metadata.subtype`` directly) happens inside
    :func:`_rescue_official_hard_triad_examples` so that non-promoted
    rescue attempts can roll it back safely.

    Returns per-family counts plus a ``from->to`` transition counter so the
    stage2 report can quantify how often the hint path activated.
    """
    hinted: dict[str, int] = {}
    transitions: dict[str, int] = {}

    for example in examples:
        if _source_kind(example) != "official":
            continue

        family = str(_metadata_value(example, "official_family") or "")
        if family not in HARD_TRIAD_FAMILIES:
            continue

        current_subtype = str(_metadata_value(example, "subtype") or "")
        hint = _infer_subtype_hint_from_top_steps(example)
        if not hint:
            continue
        if hint == current_subtype:
            # Defensive: _infer_subtype_hint_from_top_steps should already
            # never return the current fallback, but guard anyway.
            continue

        example.metadata.extras = {
            **dict(example.metadata.extras or {}),
            "search_subtype_hint": hint,
        }

        hinted[family] = hinted.get(family, 0) + 1
        key = f"{current_subtype}->{hint}"
        transitions[key] = transitions.get(key, 0) + 1

    return {
        "hinted": dict(sorted(hinted.items())),
        "transitions": dict(sorted(transitions.items())),
    }


def _annotation_quality_tuple(example: PuzzleExample) -> tuple:
    """Comparable tuple used to decide whether a rescue pass improved annotation.

    Ordering (lexicographic, higher-is-better):

    1. ``solver_verifiable`` (bool)
    2. ``support_coverage`` (float, 0..1)
    3. presence of a non-empty ``program_signature`` (bool)
    4. ``teacher_confidence`` (float)
    5. ``top1_top2_margin`` (float)
    """
    extras = example.metadata.extras or {}
    return (
        bool(extras.get("solver_verifiable", False)),
        float(extras.get("support_coverage", 0.0) or 0.0),
        bool(example.metadata.program_signature),
        float(example.metadata.teacher_confidence or 0.0),
        float(extras.get("top1_top2_margin", 0.0) or 0.0),
    )


def _rescue_official_hard_triad_examples(
    examples: list[PuzzleExample],
    *,
    families: frozenset[str] | set[str] = HARD_TRIAD_RESCUE_FAMILIES,
    beam_width: int = 12,
    max_depth: int = 4,
    top_k: int = 3,
    use_search_subtype_hint: bool = False,
) -> dict[str, Any]:
    """Second-pass stronger chain-search over official hard-triad samples that
    failed the strict stage2 gate.

    Runs only on ``official`` samples whose ``official_family`` is in ``families``
    and which did *not* pass ``_select_official_stage2_strict`` on the first
    pass. For each rescue candidate the current annotation is snapshotted, a
    deeper search is run, and the new annotation is promoted **only if the
    quality tuple strictly improves**. Otherwise the original annotation is
    restored so rescue cannot degrade any candidate.

    When ``use_search_subtype_hint`` is True, examples carrying an
    ``extras["search_subtype_hint"]`` (written by
    :func:`_attach_search_subtype_hints`) have their
    ``example.metadata.subtype`` **temporarily** overridden with that hint
    before the second-pass search runs. This is required because
    :meth:`ChainSearchEngine.solve_example` reads ``example.metadata.subtype``
    directly (not from ``extras``) to pick an op priority list, so changing
    only ``extras`` would have no effect on the search. If the rescue is not
    promoted, the original subtype is restored along with every other
    annotation field.

    Promoted examples gain ``extras.second_pass_rescue_applied=True`` plus a
    ``second_pass_settings`` snapshot for traceability; when the hint was
    used they also gain ``rescue_subtype_hint`` / ``rescue_original_subtype``
    markers so downstream tools can audit which subtype changes landed.

    Returns per-family ``rescue_attempted`` / ``rescue_promoted`` counters so
    the stage2 report can tell at a glance how many samples the rescue pass
    recovered for each hard-triad family, plus optional per-family
    ``rescue_hint_attempted`` / ``rescue_promoted_with_hint`` counters and a
    ``rescue_hint_transitions`` ``from->to`` histogram.
    """
    allowed_families = set(families) if not isinstance(families, set) else families
    rescue_attempted: dict[str, int] = {}
    rescue_promoted: dict[str, int] = {}
    rescue_hint_attempted: dict[str, int] = {}
    rescue_promoted_with_hint: dict[str, int] = {}
    rescue_hint_transitions: dict[str, int] = {}
    if not allowed_families:
        return {
            "rescue_families": [],
            "rescue_settings": {
                "beam_width": beam_width,
                "max_depth": max_depth,
                "top_k": top_k,
                "use_search_subtype_hint": bool(use_search_subtype_hint),
            },
            "rescue_attempted": {},
            "rescue_promoted": {},
            "rescue_hint_attempted": {},
            "rescue_promoted_with_hint": {},
            "rescue_hint_transitions": {},
        }

    engine = ChainSearchEngine(beam_width=beam_width, max_depth=max_depth)
    for example in examples:
        if _source_kind(example) != "official":
            continue
        family = _metadata_value(example, "official_family")
        if family not in allowed_families:
            continue
        if example.target_answer is None or not example.query:
            continue
        ok_strict, _reason = _select_official_stage2_strict(example)
        if ok_strict:
            continue
        family_key = str(family)
        rescue_attempted[family_key] = rescue_attempted.get(family_key, 0) + 1

        old_quality = _annotation_quality_tuple(example)
        # ``subtype`` MUST be snapshotted here: when
        # ``use_search_subtype_hint`` is True we temporarily override
        # ``example.metadata.subtype`` below, and a non-promoted rescue must
        # be able to roll the override back. Without this snapshot a failed
        # rescue would silently leak the hinted subtype into the first-pass
        # annotation.
        snapshot = {
            "program_signature": example.metadata.program_signature,
            "teacher_confidence": example.metadata.teacher_confidence,
            "composition_key": example.metadata.composition_key,
            "subtype": example.metadata.subtype,
            "extras": dict(example.metadata.extras or {}),
        }

        hint: str | None = None
        if use_search_subtype_hint:
            raw_hint = (example.metadata.extras or {}).get("search_subtype_hint")
            hint = str(raw_hint) if raw_hint else None
            if hint:
                current_subtype = str(example.metadata.subtype or "")
                if hint != current_subtype:
                    transition_key = f"{current_subtype}->{hint}"
                    rescue_hint_transitions[transition_key] = (
                        rescue_hint_transitions.get(transition_key, 0) + 1
                    )
                    rescue_hint_attempted[family_key] = (
                        rescue_hint_attempted.get(family_key, 0) + 1
                    )

                    # ChainSearchEngine.solve_example reads
                    # example.metadata.subtype directly; extras alone is
                    # insufficient. Record the override on extras for the
                    # report, but keep the original subtype in ``snapshot``
                    # so a failed rescue rolls everything back.
                    example.metadata.subtype = hint
                    example.metadata.extras = {
                        **dict(example.metadata.extras or {}),
                        "rescue_subtype_hint": hint,
                        "rescue_original_subtype": current_subtype,
                    }
                else:
                    # Hint matches the current subtype: nothing to do, and we
                    # should not even count it as an attempt because the
                    # search behaviour is unchanged.
                    hint = None

        candidates = engine.solve_example(example, top_k=top_k)
        annotate_example_from_candidates(example, candidates)
        new_quality = _annotation_quality_tuple(example)

        if new_quality > old_quality:
            rescue_promoted[family_key] = rescue_promoted.get(family_key, 0) + 1
            if hint:
                rescue_promoted_with_hint[family_key] = (
                    rescue_promoted_with_hint.get(family_key, 0) + 1
                )
            example.metadata.extras = {
                **(example.metadata.extras or {}),
                "second_pass_rescue_applied": True,
                "second_pass_settings": {
                    "beam_width": beam_width,
                    "max_depth": max_depth,
                    "top_k": top_k,
                    "use_search_subtype_hint": bool(use_search_subtype_hint),
                },
            }
        else:
            # Roll back so a weaker-or-equal rescue result does not overwrite
            # the first-pass annotation, including any temporary subtype
            # override from the search-subtype-hint path.
            example.metadata.program_signature = snapshot["program_signature"]
            example.metadata.teacher_confidence = snapshot["teacher_confidence"]
            example.metadata.composition_key = snapshot["composition_key"]
            example.metadata.subtype = snapshot["subtype"]
            example.metadata.extras = dict(snapshot["extras"])

    return {
        "rescue_families": sorted(allowed_families),
        "rescue_settings": {
            "beam_width": beam_width,
            "max_depth": max_depth,
            "top_k": top_k,
            "use_search_subtype_hint": bool(use_search_subtype_hint),
        },
        "rescue_attempted": dict(sorted(rescue_attempted.items())),
        "rescue_promoted": dict(sorted(rescue_promoted.items())),
        "rescue_hint_attempted": dict(sorted(rescue_hint_attempted.items())),
        "rescue_promoted_with_hint": dict(sorted(rescue_promoted_with_hint.items())),
        "rescue_hint_transitions": dict(sorted(rescue_hint_transitions.items())),
    }


def _metadata_value(example: PuzzleExample, key: str, default: Any = None) -> Any:
    if key in example.metadata.extras and example.metadata.extras[key] not in (None, ""):
        return example.metadata.extras[key]
    if hasattr(example.metadata, key):
        value = getattr(example.metadata, key)
        if value not in (None, ""):
            return value
    return default


def _source_prompt_type(example: PuzzleExample, prompt_mode: str) -> str:
    if prompt_mode == PROMPT_MODE_GENERIC:
        return "generic"
    if prompt_mode == PROMPT_MODE_CHAT_THINKING:
        return "chat_thinking"
    if example.raw_prompt.strip():
        return "raw"
    return "generic_fallback"


def _render_prompt_and_completion(
    example: PuzzleExample,
    *,
    prompt_mode: str,
    trace_style: str,
    tokenizer: Any | None,
) -> tuple[str, str]:
    """Return ``(prompt, completion)`` under the requested prompt_mode.

    ``chat_thinking`` requires a tokenizer with ``apply_chat_template`` so the
    training input is byte-identical to what the evaluator feeds to vLLM. Other
    modes keep the legacy text path.
    """
    if prompt_mode == PROMPT_MODE_CHAT_THINKING:
        if tokenizer is None:
            raise ValueError(
                "prompt_mode='chat_thinking' requires a tokenizer with "
                "apply_chat_template(). Pass tokenizer=... to build_sft_record "
                "(or to the profile builder) or switch to raw_with_guard."
            )
        prompt = build_chat_thinking_prompt(example, tokenizer)
        body = compile_completion_body(example, style=trace_style)
        completion = wrap_as_thinking(body, example.target_answer or "")
        return prompt, completion
    prompt = build_competition_prompt(example, mode=prompt_mode)
    completion = compile_completion(example, style=trace_style)
    return prompt, completion


def build_sft_record(
    example: PuzzleExample,
    *,
    stage: str,
    prompt_mode: str = PROMPT_MODE_RAW_WITH_GUARD,
    trace_style: str = "answer_only",
    default_official_family: str = "equation",
    include_metadata: bool = True,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")
    if trace_style not in TRACE_STYLES:
        raise ValueError(f"Unsupported trace style: {trace_style}")

    official_family = str(_metadata_value(example, "official_family", default_official_family))
    subtype = _metadata_value(example, "subtype")
    teacher_confidence = _metadata_value(example, "teacher_confidence")
    program_signature = _metadata_value(example, "program_signature")
    source_dataset = str(_metadata_value(example, "source_dataset", _metadata_value(example, "source", "official")))
    difficulty = _metadata_value(example, "difficulty")
    prompt, completion = _render_prompt_and_completion(
        example,
        prompt_mode=prompt_mode,
        trace_style=trace_style,
        tokenizer=tokenizer,
    )

    record = {
        "id": example.id,
        "prompt": prompt,
        "completion": completion,
        "stage": stage,
        "prompt_mode": prompt_mode,
        "trace_style": trace_style,
        "source_prompt_type": _source_prompt_type(example, prompt_mode),
        "target_answer": example.target_answer,
        "official_family": official_family,
        "subtype": subtype,
        "program_signature": program_signature,
        "teacher_confidence": teacher_confidence,
        "source_dataset": source_dataset,
        "difficulty": difficulty,
        "split": example.metadata.split,
    }
    if include_metadata:
        record["metadata"] = {
            "official_family": official_family,
            "subtype": subtype,
            "program_signature": program_signature,
            "teacher_confidence": teacher_confidence,
            "difficulty": difficulty,
            "source_dataset": source_dataset,
            "extras": dict(example.metadata.extras),
        }
    return record


def _load_split_ids(
    *,
    split_file: str | Path,
    split_name: str,
    split_role: str,
) -> set[str]:
    """Return the set of example ids belonging to ``split_name/split_role``."""
    payload = read_json(split_file)
    if split_name not in payload:
        raise KeyError(f"Split '{split_name}' not found in {split_file}")
    split_payload = payload[split_name]
    key = f"{split_role}_ids"
    if key not in split_payload:
        raise KeyError(f"Split role '{split_role}' not found in split '{split_name}'")
    return {str(item) for item in split_payload[key]}


def filter_examples_by_split(
    examples: list[PuzzleExample],
    *,
    split_file: str | Path,
    split_name: str,
    split_role: str,
    exclude_split_file: str | Path | None = None,
    exclude_split_name: str | None = None,
    exclude_split_role: str | None = None,
) -> list[PuzzleExample]:
    """Filter examples to ``split_name/split_role``, optionally excluding another split.

    The exclude fields let callers produce leak-free train subsets: e.g.,
    ``rule_novelty_all/train`` minus ``hard_triad_rule_novelty/valid``, which is
    required because the two splits are built independently (not nested) in
    :mod:`src.competition.split_builder`.

    All three exclude fields must be provided together or omitted together.
    """
    keep_ids = _load_split_ids(
        split_file=split_file,
        split_name=split_name,
        split_role=split_role,
    )

    exclude_args = (exclude_split_file, exclude_split_name, exclude_split_role)
    has_any_exclude = any(arg is not None for arg in exclude_args)
    has_all_exclude = all(arg is not None for arg in exclude_args)
    if has_any_exclude and not has_all_exclude:
        raise ValueError(
            "exclude_split_file / exclude_split_name / exclude_split_role must "
            "be provided together"
        )

    exclude_ids: set[str] = set()
    if has_all_exclude:
        exclude_ids = _load_split_ids(
            split_file=exclude_split_file,  # type: ignore[arg-type]
            split_name=str(exclude_split_name),
            split_role=str(exclude_split_role),
        )

    filtered: list[PuzzleExample] = []
    for example in examples:
        source_kind = _source_kind(example)
        if example.id in exclude_ids:
            continue
        if example.id in keep_ids:
            filtered.append(example)
            continue
        if split_role == "train" and source_kind in {"synth", "repair"}:
            filtered.append(example)
    return filtered


def export_split_subset(
    examples: list[PuzzleExample],
    *,
    split_file: str | Path,
    split_name: str,
    split_role: str,
    exclude_split_file: str | Path | None = None,
    exclude_split_name: str | None = None,
    exclude_split_role: str | None = None,
) -> list[dict[str, Any]]:
    return [
        example.to_dict()
        for example in filter_examples_by_split(
            examples,
            split_file=split_file,
            split_name=split_name,
            split_role=split_role,
            exclude_split_file=exclude_split_file,
            exclude_split_name=exclude_split_name,
            exclude_split_role=exclude_split_role,
        )
    ]


def _source_kind(example: PuzzleExample) -> str:
    source = str(_metadata_value(example, "source", "official"))
    if source == "synthetic":
        return "synth"
    if source == "repair":
        return "repair"
    return "official"


def _filter_by_source(examples: list[PuzzleExample], source_filter: str | None) -> list[PuzzleExample]:
    if not source_filter:
        return examples
    wanted = source_filter.strip()
    return [example for example in examples if _source_kind(example) == wanted]


def build_stage1_sft(
    examples: list[PuzzleExample],
    *,
    prompt_mode: str = PROMPT_MODE_RAW_WITH_GUARD,
    include_metadata: bool = True,
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    records = []
    for example in examples:
        if _source_kind(example) != "official":
            continue
        if example.target_answer is None or not example.query:
            continue
        records.append(
            build_sft_record(
                example,
                stage="stage1",
                prompt_mode=prompt_mode,
                trace_style="answer_only",
                include_metadata=include_metadata,
                tokenizer=tokenizer,
            )
        )
    return records


STAGE2_REJECTION_REASONS: tuple[str, ...] = (
    "missing_target_or_query",
    "low_teacher_confidence",
    "not_solver_verifiable",
    "partial_support_coverage",
    "query_prediction_mismatch",
    "high_risk_template_trace",
    "missing_program_signature",
)

STAGE2_SILVER_FAMILY_PRIORITY: tuple[str, ...] = ("equation", "cipher", "bit")


def _select_official_stage2_strict(
    example: PuzzleExample,
) -> tuple[bool, str | None]:
    """Strict stage2 gate. Returns ``(accepted, rejection_reason_or_None)``.

    ``rejection_reason`` is ``None`` when the example is not an official
    sample (there is nothing to reject), and otherwise one of
    :data:`STAGE2_REJECTION_REASONS`.
    """
    if _source_kind(example) != "official":
        return False, None
    if example.target_answer is None or not example.query:
        return False, "missing_target_or_query"
    if float(_metadata_value(example, "teacher_confidence", 0.0) or 0.0) < 0.80:
        return False, "low_teacher_confidence"
    if not bool(example.metadata.extras.get("solver_verifiable")):
        return False, "not_solver_verifiable"
    if float(example.metadata.extras.get("support_coverage", 0.0) or 0.0) < 1.0:
        return False, "partial_support_coverage"
    if example.metadata.extras.get("query_solver_correct") is False:
        return False, "query_prediction_mismatch"
    if (
        example.metadata.subtype == "equation_template"
        and example.metadata.extras.get("template_risk_class")
        in {
            "ranker_miss_oracle_hit",
            "operator_gap_oracle_miss",
            "unseen_key_template_miss",
            "unseen_literal_high_risk",
        }
    ):
        return False, "high_risk_template_trace"
    if not _metadata_value(example, "program_signature"):
        return False, "missing_program_signature"
    return True, None


def _select_official_stage2_silver(
    example: PuzzleExample,
    *,
    hard_confidence: float = 0.65,
    hard_support: float = 0.67,
) -> bool:
    """Silver stage2 gate. Train-only. Broadens the strict gate for hard-triad
    official samples so stage2 sees more of the real-distribution coverage.

    Requirements:

    - official sample
    - non-empty target_answer and query
    - family in HARD_TRIAD_FAMILIES
    - teacher_confidence >= ``hard_confidence`` (default 0.65)
    - support_coverage >= ``hard_support`` (default 0.67)
    - at least one of program_signature present / solver_verifiable

    Silver samples are intended to contribute prompt distribution + final
    answer supervision only; callers should pair this with ``answer_only``
    trace style so a weak teacher does not inject unreliable traces.
    """
    if _source_kind(example) != "official":
        return False
    if example.target_answer is None or not example.query:
        return False
    family = _metadata_value(example, "official_family")
    if family not in HARD_TRIAD_FAMILIES:
        return False
    if float(_metadata_value(example, "teacher_confidence", 0.0) or 0.0) < float(hard_confidence):
        return False
    if float(example.metadata.extras.get("support_coverage", 0.0) or 0.0) < float(hard_support):
        return False
    has_signature = bool(_metadata_value(example, "program_signature"))
    solver_ok = bool(example.metadata.extras.get("solver_verifiable"))
    if not (has_signature or solver_ok):
        return False
    return True


def _select_official_stage2(example: PuzzleExample) -> bool:
    """Backward-compatible bool-only wrapper around the strict gate."""
    ok, _ = _select_official_stage2_strict(example)
    return ok


def _select_synth_stage2(example: PuzzleExample, *, official_signatures: set[str]) -> bool:
    if _source_kind(example) != "synth":
        return False
    if example.target_answer is None or not example.query:
        return False
    if not bool(example.metadata.extras.get("solver_verifiable")):
        return False
    signature = _metadata_value(example, "program_signature")
    if not signature:
        return False
    return signature not in official_signatures


def _family_for_record(record: dict[str, Any]) -> str:
    family = record.get("official_family")
    return str(family) if family else "unknown"


def _signature_bucket_key(record: dict[str, Any]) -> str:
    signature = record.get("program_signature")
    if signature:
        return str(signature)
    return f"id:{record.get('id', 'unknown')}"


def _ordered_families(records: list[dict[str, Any]]) -> list[str]:
    seen = {_family_for_record(record) for record in records}
    ordered = [family for family in BALANCED_FAMILY_ORDER if family in seen]
    ordered.extend(sorted(seen - set(BALANCED_FAMILY_ORDER)))
    return ordered


def _truncate_by_signature_bucket(
    records: list[dict[str, Any]],
    *,
    max_per_signature_bucket: int,
) -> list[dict[str, Any]]:
    if max_per_signature_bucket <= 0:
        return list(records)
    signature_counts: dict[str, int] = {}
    kept: list[dict[str, Any]] = []
    for record in records:
        key = _signature_bucket_key(record)
        if signature_counts.get(key, 0) >= max_per_signature_bucket:
            continue
        signature_counts[key] = signature_counts.get(key, 0) + 1
        kept.append(record)
    return kept


def _oversample_hard_triad_records(
    records: list[dict[str, Any]],
    *,
    repeat_factor: int,
) -> list[dict[str, Any]]:
    """Duplicate hard-triad records ``repeat_factor`` times.

    Unlike :func:`_balance_records_by_family`, which only reorders records
    along a family schedule, this helper produces a longer list: each
    hard-triad (``bit`` / ``cipher`` / ``equation``) record appears
    ``repeat_factor`` times, easy-triad records stay as single copies.

    Duplicated records keep their original ``id`` so downstream reports can
    count them as duplications; when the record has a ``metadata`` dict, the
    duplicate copies carry an ``oversample_repeat_index`` marker in
    ``metadata.extras`` for traceability.
    """
    factor = max(1, int(repeat_factor))
    if factor <= 1:
        return list(records)

    expanded: list[dict[str, Any]] = []
    for record in records:
        family = _family_for_record(record)
        repeats = factor if family in HARD_TRIAD_FAMILIES else 1
        for repeat_index in range(repeats):
            clone = dict(record)
            if repeat_index > 0 and isinstance(clone.get("metadata"), dict):
                metadata = dict(clone["metadata"])
                extras_in = metadata.get("extras")
                extras = dict(extras_in) if isinstance(extras_in, dict) else {}
                extras["oversample_repeat_index"] = repeat_index
                metadata["extras"] = extras
                clone["metadata"] = metadata
            expanded.append(clone)
    return expanded


def _balance_records_by_family(
    records: list[dict[str, Any]],
    *,
    hard_triad_repeat_factor: int,
    max_per_signature_bucket: int,
) -> list[dict[str, Any]]:
    if not records:
        return []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(_family_for_record(record), []).append(record)

    family_schedule: list[str] = []
    repeat_factor = max(1, int(hard_triad_repeat_factor))
    for family in _ordered_families(records):
        visits = repeat_factor if family in HARD_TRIAD_FAMILIES else 1
        family_schedule.extend([family] * visits)

    indices = {family: 0 for family in buckets}
    signature_counts: dict[str, int] = {}
    balanced: list[dict[str, Any]] = []
    while True:
        made_progress = False
        for family in family_schedule:
            bucket = buckets.get(family, [])
            while indices.get(family, 0) < len(bucket):
                record = bucket[indices[family]]
                indices[family] += 1
                signature_key = _signature_bucket_key(record)
                if max_per_signature_bucket > 0 and signature_counts.get(signature_key, 0) >= max_per_signature_bucket:
                    continue
                signature_counts[signature_key] = signature_counts.get(signature_key, 0) + 1
                balanced.append(record)
                made_progress = True
                break
        if not made_progress:
            break
    return balanced


def _record_source_dataset(record: dict[str, Any]) -> str:
    source_dataset = record.get("source_dataset")
    return str(source_dataset) if source_dataset else "unknown"


def summarise_selected_sft(records: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(_family_for_record(record) for record in records)
    subtype_counts = Counter(
        f"{_family_for_record(record)}:{record.get('subtype')}"
        for record in records
        if record.get("subtype")
    )
    source_dataset_counts = Counter(_record_source_dataset(record) for record in records)
    hard_triad_records = sum(_family_for_record(record) in HARD_TRIAD_FAMILIES for record in records)
    unique_ids = {
        str(record.get("id"))
        for record in records
        if record.get("id") is not None
    }
    return {
        "num_records": len(records),
        "num_unique_ids": len(unique_ids),
        "duplication_ratio": 0.0 if not records else 1.0 - (len(unique_ids) / len(records)),
        "family_counts": dict(sorted(family_counts.items())),
        "subtype_counts": dict(sorted(subtype_counts.items())),
        "source_dataset_counts": dict(sorted(source_dataset_counts.items())),
        "hard_triad_ratio": 0.0 if not records else hard_triad_records / len(records),
    }


def _mark_stage3_record(
    record: dict[str, Any],
    *,
    source_dataset: str,
    repair_source: str,
) -> dict[str, Any]:
    record["source_dataset"] = source_dataset
    record["repair_source"] = repair_source
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        metadata["source_dataset"] = source_dataset
        extras = dict(metadata.get("extras", {}))
        extras["repair_source"] = repair_source
        metadata["extras"] = extras
    return record


def summarise_repair_sft(records: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(_family_for_record(record) for record in records)
    source_dataset_counts = Counter(_record_source_dataset(record) for record in records)
    repair_sources = Counter(str(record.get("repair_source", "repair")) for record in records)
    return {
        "num_records": len(records),
        "repair_count": repair_sources.get("repair", 0),
        "replay_count": repair_sources.get("replay", 0),
        "family_counts": dict(sorted(family_counts.items())),
        "source_dataset_counts": dict(sorted(source_dataset_counts.items())),
    }


def build_selected_sft_with_report(
    examples: list[PuzzleExample],
    *,
    prompt_mode: str = PROMPT_MODE_RAW_WITH_GUARD,
    trace_style: str = "token_trace",
    include_metadata: bool = True,
    beam_width: int = 8,
    max_depth: int = 2,
    top_k: int = 2,
    balance_by_family: bool = True,
    hard_triad_repeat_factor: int = 2,
    oversample_hard_triad: bool = False,
    max_per_signature_bucket: int = 64,
    tokenizer: Any | None = None,
    enable_silver_official: bool = False,
    silver_hard_confidence: float = 0.65,
    silver_hard_support: float = 0.67,
    silver_max_fraction: float = 0.25,
    silver_max_absolute: int = 800,
    rescue_hard_triad: bool = False,
    rescue_families: frozenset[str] | set[str] | None = None,
    rescue_beam_width: int = 12,
    rescue_max_depth: int = 4,
    rescue_top_k: int = 3,
    stage2_use_search_subtype_hint: bool = False,
) -> dict[str, Any]:
    """Build the stage2 SFT dataset and return both records and selection report.

    ``records`` is the final ordered list of SFT records.
    ``selection_counts`` breaks down how many samples came from each bucket.
    ``official_rejection_diagnostics`` records per-family rejection reasons
    from ``_select_official_stage2_strict`` so the next training iteration can
    tell whether hard-triad official samples were blocked by confidence, by
    missing program_signature, etc.

    When ``enable_silver_official`` is True, hard-triad official samples that
    miss the strict gate but satisfy the silver gate are added to the training
    pool, capped by ``min(silver_max_fraction * (strict + synth), silver_max_absolute)``
    and sampled round-robin in priority order ``equation -> cipher -> bit``.
    Silver samples are always recorded with ``trace_style="answer_only"`` so a
    weak teacher does not inject unreliable traces.

    ``stage2_use_search_subtype_hint`` is an **experimental, off-by-default**
    switch that infers a finer-grained subtype from ``top_candidate_steps``
    after the first-pass annotation (via
    :func:`_attach_search_subtype_hints`) and then lets the rescue pass
    temporarily override ``example.metadata.subtype`` with that hint so the
    second chain search picks a targeted op priority. Canonical stage2 must
    keep this False; enable only through the dedicated subtype-rescue script
    so A/B attribution stays clean.
    """
    annotated = _annotate_examples(
        examples,
        beam_width=beam_width,
        max_depth=max_depth,
        top_k=top_k,
    )

    subtype_hint_diagnostics: dict[str, Any] = {
        "enabled": bool(stage2_use_search_subtype_hint),
        "hinted": {},
        "transitions": {},
    }
    if stage2_use_search_subtype_hint:
        attach_stats = _attach_search_subtype_hints(annotated)
        subtype_hint_diagnostics.update(attach_stats)

    rescue_diagnostics: dict[str, Any] = {
        "rescue_families": [],
        "rescue_settings": None,
        "rescue_attempted": {},
        "rescue_promoted": {},
        "rescue_hint_attempted": {},
        "rescue_promoted_with_hint": {},
        "rescue_hint_transitions": {},
    }
    if rescue_hard_triad:
        effective_families = (
            rescue_families
            if rescue_families is not None
            else HARD_TRIAD_RESCUE_FAMILIES
        )
        rescue_diagnostics = _rescue_official_hard_triad_examples(
            annotated,
            families=effective_families,
            beam_width=rescue_beam_width,
            max_depth=rescue_max_depth,
            top_k=rescue_top_k,
            use_search_subtype_hint=stage2_use_search_subtype_hint,
        )

    # Phase 1: strict official + silver candidates + rejection diagnostics
    selected_strict: list[PuzzleExample] = []
    rejection_counter: dict[str, dict[str, int]] = {}
    silver_candidates_by_family: dict[str, list[PuzzleExample]] = {
        family: [] for family in STAGE2_SILVER_FAMILY_PRIORITY
    }
    for example in annotated:
        if _source_kind(example) != "official":
            continue
        ok_strict, reason = _select_official_stage2_strict(example)
        if ok_strict:
            selected_strict.append(example)
            continue
        family = str(_metadata_value(example, "official_family") or "unknown")
        if reason is not None:
            bucket = rejection_counter.setdefault(family, {})
            bucket[reason] = bucket.get(reason, 0) + 1
        if enable_silver_official and family in silver_candidates_by_family:
            if _select_official_stage2_silver(
                example,
                hard_confidence=silver_hard_confidence,
                hard_support=silver_hard_support,
            ):
                silver_candidates_by_family[family].append(example)

    # Phase 2: synth selection uses strict officials' signatures as a dedup
    # reference; silver signatures are intentionally excluded so a low-confidence
    # silver match does not suppress a higher-quality synth sample.
    strict_signatures = {
        str(_metadata_value(example, "program_signature"))
        for example in selected_strict
        if _metadata_value(example, "program_signature")
    }
    selected_synth = [
        example
        for example in annotated
        if _select_synth_stage2(example, official_signatures=strict_signatures)
    ]

    # Phase 3: silver cap (fraction and absolute, whichever is smaller), then
    # round-robin sample in equation -> cipher -> bit priority order.
    silver_selected: list[PuzzleExample] = []
    if enable_silver_official:
        base_count = len(selected_strict) + len(selected_synth)
        fraction_cap = int(float(silver_max_fraction) * max(1, base_count))
        cap = min(max(0, fraction_cap), max(0, int(silver_max_absolute)))
        if cap > 0:
            indices = {family: 0 for family in STAGE2_SILVER_FAMILY_PRIORITY}
            while len(silver_selected) < cap:
                made_progress = False
                for family in STAGE2_SILVER_FAMILY_PRIORITY:
                    if len(silver_selected) >= cap:
                        break
                    bucket = silver_candidates_by_family[family]
                    idx = indices[family]
                    if idx >= len(bucket):
                        continue
                    silver_selected.append(bucket[idx])
                    indices[family] += 1
                    made_progress = True
                if not made_progress:
                    break

    # Phase 4: dedupe + render records. Silver samples use answer_only; strict
    # and synth samples use the caller-provided trace_style.
    dedupe_keys: set[tuple[str, Any, Any]] = set()
    records: list[dict[str, Any]] = []
    for example, bucket_name in (
        [(example, "strict") for example in selected_strict]
        + [(example, "synth") for example in selected_synth]
        + [(example, "silver") for example in silver_selected]
    ):
        key = (
            example.query,
            example.target_answer,
            _metadata_value(example, "program_signature"),
        )
        if key in dedupe_keys:
            continue
        dedupe_keys.add(key)
        per_record_trace = "answer_only" if bucket_name == "silver" else trace_style
        records.append(
            build_sft_record(
                example,
                stage="stage2",
                prompt_mode=prompt_mode,
                trace_style=per_record_trace,
                include_metadata=include_metadata,
                tokenizer=tokenizer,
            )
        )

    # Phase 5: oversample hard-triad records (train-side weighting), then
    # family-balance with hard_triad_repeat_factor=1 to avoid double-counting.
    if oversample_hard_triad and hard_triad_repeat_factor > 1:
        records = _oversample_hard_triad_records(
            records,
            repeat_factor=hard_triad_repeat_factor,
        )

    if balance_by_family:
        ordered = _balance_records_by_family(
            records,
            hard_triad_repeat_factor=1,
            max_per_signature_bucket=max_per_signature_bucket,
        )
    else:
        ordered = _truncate_by_signature_bucket(
            records, max_per_signature_bucket=max_per_signature_bucket
        )

    merged_subtype_hint_diagnostics = {
        **subtype_hint_diagnostics,
        "rescue_hint_attempted": rescue_diagnostics.get("rescue_hint_attempted", {}),
        "rescue_promoted_with_hint": rescue_diagnostics.get(
            "rescue_promoted_with_hint", {}
        ),
        "rescue_hint_transitions": rescue_diagnostics.get(
            "rescue_hint_transitions", {}
        ),
    }

    return {
        "records": ordered,
        "selection_counts": {
            "official_strict": len(selected_strict),
            "official_silver": len(silver_selected),
            "synth": len(selected_synth),
        },
        "official_rejection_diagnostics": {
            family: dict(sorted(reasons.items()))
            for family, reasons in sorted(rejection_counter.items())
        },
        "rescue_diagnostics": rescue_diagnostics,
        "subtype_hint_diagnostics": merged_subtype_hint_diagnostics,
    }


def build_selected_sft(
    examples: list[PuzzleExample],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Backward-compatible shim: returns just the records list."""
    return build_selected_sft_with_report(examples, **kwargs)["records"]


def _repair_bucket(example: PuzzleExample, failure_row: dict[str, Any]) -> str:
    enriched = {
        **failure_row,
        "official_family": _metadata_value(example, "official_family"),
        "subtype": _metadata_value(example, "subtype"),
        "program_signature": _metadata_value(example, "program_signature"),
        "teacher_confidence": _metadata_value(example, "teacher_confidence"),
    }
    return classify_error(enriched)


def _validate_repair_artifact(
    payload: Any,
    *,
    artifact_path: str | Path,
) -> list[dict[str, Any]]:
    """Validate a repair artifact (typically ``baseline_eval.json``) and return its records.

    Ensures that every downstream assumption in ``build_repair_set`` is met:

    - the payload is a JSON object
    - it carries a non-empty ``records`` (or legacy ``rows``) list
    - every record contains at least ``id`` and ``competition_correct``

    The previous implementation relied on ``row.get("competition_correct", False)``
    which silently treats any legacy record (no such field) as a failure, producing
    a stage3 dataset that spans the entire training set.
    """
    if not isinstance(payload, dict):
        raise RepairArtifactSchemaError(
            f"Repair artifact {artifact_path} is not a JSON object "
            f"(got {type(payload).__name__}). {REPAIR_ARTIFACT_REMEDIATION}"
        )

    records = payload.get("records")
    if records is None:
        records = payload.get("rows")
        if records is None:
            raise RepairArtifactSchemaError(
                f"Repair artifact {artifact_path} is missing both 'records' and 'rows'. "
                f"{REPAIR_ARTIFACT_REMEDIATION}"
            )
        warnings.warn(
            f"Repair artifact {artifact_path} uses the legacy 'rows' field; "
            "the canonical schema uses 'records'. Regenerate via run_baseline.",
            DeprecationWarning,
            stacklevel=3,
        )

    if not isinstance(records, list):
        raise RepairArtifactSchemaError(
            f"Repair artifact {artifact_path} has non-list records "
            f"(got {type(records).__name__}). {REPAIR_ARTIFACT_REMEDIATION}"
        )
    if not records:
        raise RepairArtifactSchemaError(
            f"Repair artifact {artifact_path} has an empty records list. "
            "The baseline run produced no evaluation rows. "
            f"{REPAIR_ARTIFACT_REMEDIATION}"
        )

    missing_counts: dict[str, int] = {}
    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            raise RepairArtifactSchemaError(
                f"Repair artifact {artifact_path} record at index {idx} is "
                f"{type(row).__name__}, expected dict."
            )
        for field in _REQUIRED_FAILURE_FIELDS:
            if field not in row:
                missing_counts[field] = missing_counts.get(field, 0) + 1

    if missing_counts:
        summary = ", ".join(
            f"'{field}' (missing in {count}/{len(records)} records)"
            for field, count in sorted(missing_counts.items())
        )
        raise RepairArtifactSchemaError(
            f"Repair artifact {artifact_path} is missing required fields: {summary}. "
            f"This looks like a legacy baseline schema. {REPAIR_ARTIFACT_REMEDIATION}"
        )

    return records


def build_repair_set(
    examples: list[PuzzleExample],
    *,
    repair_artifact: str | Path,
    prompt_mode: str = PROMPT_MODE_RAW_WITH_GUARD,
    trace_style: str = "short_trace",
    include_metadata: bool = True,
    beam_width: int = 8,
    max_depth: int = 2,
    top_k: int = 2,
    replay_input: str | Path | None = None,
    replay_ratio: float = 0.0,
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    payload = read_json(repair_artifact)
    repair_rows = _validate_repair_artifact(payload, artifact_path=repair_artifact)
    failure_index = {
        str(row["id"]): row
        for row in repair_rows
        if not row["competition_correct"]
    }
    failure_ids = set(failure_index.keys())

    # Resolve replay rows up front so annotation can be restricted to the
    # union of failure + success ids. Without this prefilter the caller pays
    # chain-search cost for every example in the input pool, which is ~10k
    # for the stage3 full-official-train path even though most of them are
    # neither repair nor replay candidates.
    replay_rows: list[dict[str, Any]] = []
    success_ids: set[str] = set()
    if replay_input and replay_ratio > 0.0:
        replay_payload = read_json(replay_input)
        replay_records_payload: list[Any] = []
        if isinstance(replay_payload, dict):
            replay_records_payload = replay_payload.get(
                "records", replay_payload.get("rows", [])
            )
        if replay_records_payload:
            replay_rows = _validate_repair_artifact(
                replay_payload, artifact_path=replay_input
            )
        success_ids = {
            str(row["id"])
            for row in replay_rows
            if bool(row.get("competition_correct", False))
        }

    target_ids = failure_ids | success_ids
    selected_examples = [example for example in examples if example.id in target_ids]
    present_ids = {example.id for example in selected_examples}
    missing_target_ids = sorted(target_ids - present_ids)
    if missing_target_ids:
        warnings.warn(
            f"build_repair_set: {len(missing_target_ids)} ids were present in the "
            "repair/replay artifact but missing from the provided examples input; "
            "those ids will be skipped.",
            stacklevel=2,
        )

    annotated = _annotate_examples(
        selected_examples,
        beam_width=beam_width,
        max_depth=max_depth,
        top_k=top_k,
    )

    repair_records: list[dict[str, Any]] = []
    for example in annotated:
        failure_row = failure_index.get(example.id)
        if failure_row is None:
            continue
        bucket = _repair_bucket(example, failure_row)
        record = build_sft_record(
            example,
            stage="stage3",
            prompt_mode=prompt_mode,
            trace_style=trace_style,
            include_metadata=include_metadata,
            tokenizer=tokenizer,
        )
        record["error_type"] = bucket
        record["target_signature"] = _metadata_value(example, "program_signature")
        record["predicted_signature"] = failure_row.get("predicted_signature")
        repair_records.append(_mark_stage3_record(record, source_dataset="repair", repair_source="repair"))

    replay_records: list[dict[str, Any]] = []
    if success_ids:
        replay_candidates: list[dict[str, Any]] = []
        for example in annotated:
            if example.id not in success_ids:
                continue
            record = build_sft_record(
                example,
                stage="stage3",
                prompt_mode=prompt_mode,
                trace_style=trace_style,
                include_metadata=include_metadata,
                tokenizer=tokenizer,
            )
            replay_candidates.append(_mark_stage3_record(record, source_dataset="replay", repair_source="replay"))
        replay_target = int(len(repair_records) * replay_ratio)
        if replay_target > 0:
            replay_candidates = _balance_records_by_family(
                replay_candidates,
                hard_triad_repeat_factor=1,
                max_per_signature_bucket=64,
            )
            replay_records = replay_candidates[:replay_target]

    return repair_records + replay_records


def _resolve_input_paths(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    if getattr(args, "inputs", None):
        return [str(path) for path in args.inputs]
    return _parse_input_paths(
        args.input
        or config.get("input")
        or config.get("training", {}).get("dataset_input_paths")
        or config.get("training", {}).get("dataset_path")
    )


def _resolve_cli_tokenizer(
    prompt_mode: str,
    tokenizer_path: str | None,
    config: dict[str, Any],
) -> Any | None:
    """Load a tokenizer only when prompt_mode requires one.

    This is a helper for the CLI entry point; library callers should pass
    ``tokenizer`` explicitly to the builder functions. ``transformers`` is
    imported lazily so that non-chat-thinking invocations don't depend on it.
    """
    if prompt_mode != PROMPT_MODE_CHAT_THINKING:
        return None
    path = tokenizer_path or config.get("tokenizer_path")
    if not path:
        raise ValueError(
            "prompt_mode='chat_thinking' requires --tokenizer-path (or "
            "'tokenizer_path' in the config). Populate the directory via "
            "scripts/probe_chat_template.py."
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for prompt_mode='chat_thinking'. "
            "Install with: pip install transformers"
        ) from exc
    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage1/stage2/stage3 SFT datasets.")
    parser.add_argument("--config")
    parser.add_argument("--input")
    parser.add_argument("--inputs", nargs="*")
    parser.add_argument("--output")
    parser.add_argument("--selection-profile", choices=["stage1", "stage2", "stage3"], default="stage2")
    parser.add_argument("--prompt-mode", choices=sorted(PROMPT_MODES), default=PROMPT_MODE_RAW_WITH_GUARD)
    parser.add_argument("--completion-style", choices=sorted(TRACE_STYLES), default="token_trace")
    parser.add_argument("--source-filter", choices=["official", "synth", "repair"])
    parser.add_argument("--split-file")
    parser.add_argument("--split-name")
    parser.add_argument("--split-role", choices=["train", "valid"])
    parser.add_argument(
        "--exclude-split-file",
        help=(
            "Optional splits.json used to drop example ids that belong to an "
            "overlapping split (e.g., rule_novelty_all/train minus "
            "hard_triad_rule_novelty/valid)."
        ),
    )
    parser.add_argument("--exclude-split-name")
    parser.add_argument("--exclude-split-role", choices=["train", "valid"])
    parser.add_argument("--repair-artifact")
    parser.add_argument("--replay-input")
    parser.add_argument("--replay-ratio", type=float, default=0.0)
    parser.add_argument("--default-official-family", default="equation")
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--balance-by-family", dest="balance_by_family", action="store_true")
    parser.add_argument("--no-balance-by-family", dest="balance_by_family", action="store_false")
    parser.set_defaults(balance_by_family=True)
    parser.add_argument("--hard-triad-repeat-factor", type=int, default=2)
    parser.add_argument(
        "--oversample-hard-triad",
        action="store_true",
        help=(
            "Materialise hard-triad duplication as real repeated records in "
            "the stage2 output (stage2 train only). Without this flag, "
            "--hard-triad-repeat-factor only affects family-round-robin "
            "ordering and does not actually increase hard-triad sample count."
        ),
    )
    parser.add_argument("--max-per-signature-bucket", type=int, default=64)
    parser.add_argument("--report-output")
    parser.add_argument(
        "--stage2-enable-silver-official",
        action="store_true",
        help=(
            "Broaden stage2 official hard-triad coverage by also admitting "
            "samples that satisfy the silver gate (weaker confidence/support "
            "thresholds). Train-only; silver samples are always emitted with "
            "trace_style=answer_only."
        ),
    )
    parser.add_argument(
        "--stage2-silver-hard-confidence",
        type=float,
        default=0.65,
        help="Min teacher_confidence required for a silver hard-triad sample.",
    )
    parser.add_argument(
        "--stage2-silver-hard-support",
        type=float,
        default=0.67,
        help="Min support_coverage required for a silver hard-triad sample.",
    )
    parser.add_argument(
        "--stage2-silver-max-fraction",
        type=float,
        default=0.25,
        help=(
            "Silver sample count is capped at "
            "fraction * (strict_official + synth)."
        ),
    )
    parser.add_argument(
        "--stage2-silver-max-absolute",
        type=int,
        default=800,
        help="Absolute upper bound on the number of silver samples.",
    )
    parser.add_argument(
        "--stage2-second-pass-hard-triad",
        action="store_true",
        help=(
            "Enable a second-pass chain search for rejected official hard-triad "
            "samples. Off by default: shell scripts opt in explicitly."
        ),
    )
    parser.add_argument(
        "--stage2-rescue-families",
        default="equation",
        help=(
            "Comma-separated list of families the rescue pass runs on. "
            "Default: equation (the family most consistently missing program_signature "
            "in the strict rejection diagnostics)."
        ),
    )
    parser.add_argument(
        "--stage2-second-pass-beam-width",
        type=int,
        default=12,
        help="Beam width for the rescue ChainSearchEngine.",
    )
    parser.add_argument(
        "--stage2-second-pass-max-depth",
        type=int,
        default=4,
        help="Max depth for the rescue ChainSearchEngine.",
    )
    parser.add_argument(
        "--stage2-second-pass-top-k",
        type=int,
        default=3,
        help="Top-k for the rescue annotate_example_from_candidates.",
    )
    parser.add_argument(
        "--stage2-use-search-subtype-hint",
        action="store_true",
        help=(
            "Experimental. Infer a finer-grained subtype from top_candidate_steps "
            "after the first-pass annotation, and let the rescue second pass "
            "temporarily override example.metadata.subtype with that hint so "
            "ChainSearchEngine picks a targeted op priority. Off by default; "
            "canonical stage2 must leave it off to keep A/B attribution clean."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        help=(
            "Local directory for the tokenizer used by prompt_mode=chat_thinking. "
            "Populate via scripts/probe_chat_template.py (tokenizer-only mode)."
        ),
    )
    args = parser.parse_args()

    config = read_yaml(args.config) if args.config else {}
    input_paths = _resolve_input_paths(args, config)
    if not input_paths:
        raise ValueError("At least one input JSONL path is required.")

    examples = _load_examples(input_paths)
    if args.split_file and args.split_name and args.split_role:
        examples = filter_examples_by_split(
            examples,
            split_file=args.split_file,
            split_name=args.split_name,
            split_role=args.split_role,
            exclude_split_file=args.exclude_split_file,
            exclude_split_name=args.exclude_split_name,
            exclude_split_role=args.exclude_split_role,
        )
    examples = _filter_by_source(examples, args.source_filter)

    prompt_mode = args.prompt_mode
    trace_style = args.completion_style
    include_metadata = args.include_metadata
    tokenizer = _resolve_cli_tokenizer(prompt_mode, args.tokenizer_path, config)

    if args.selection_profile == "stage1":
        dataset = build_stage1_sft(
            examples,
            prompt_mode=prompt_mode,
            include_metadata=include_metadata,
            tokenizer=tokenizer,
        )
        report = {"num_records": len(dataset)}
        default_output = "data/processed/stage1_format_align.jsonl"
    elif args.selection_profile == "stage2":
        rescue_families_arg = {
            token.strip()
            for token in str(args.stage2_rescue_families or "").split(",")
            if token.strip()
        } or None
        stage2_bundle = build_selected_sft_with_report(
            examples,
            prompt_mode=prompt_mode,
            trace_style=trace_style,
            include_metadata=include_metadata,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            top_k=args.top_k,
            balance_by_family=args.balance_by_family,
            hard_triad_repeat_factor=args.hard_triad_repeat_factor,
            oversample_hard_triad=args.oversample_hard_triad,
            max_per_signature_bucket=args.max_per_signature_bucket,
            tokenizer=tokenizer,
            enable_silver_official=args.stage2_enable_silver_official,
            silver_hard_confidence=args.stage2_silver_hard_confidence,
            silver_hard_support=args.stage2_silver_hard_support,
            silver_max_fraction=args.stage2_silver_max_fraction,
            silver_max_absolute=args.stage2_silver_max_absolute,
            rescue_hard_triad=args.stage2_second_pass_hard_triad,
            rescue_families=rescue_families_arg,
            rescue_beam_width=args.stage2_second_pass_beam_width,
            rescue_max_depth=args.stage2_second_pass_max_depth,
            rescue_top_k=args.stage2_second_pass_top_k,
            stage2_use_search_subtype_hint=args.stage2_use_search_subtype_hint,
        )
        dataset = stage2_bundle["records"]
        report = summarise_selected_sft(dataset)
        report["selection_counts"] = stage2_bundle["selection_counts"]
        report["official_rejection_diagnostics"] = stage2_bundle["official_rejection_diagnostics"]
        report["rescue_diagnostics"] = stage2_bundle["rescue_diagnostics"]
        report["subtype_hint_diagnostics"] = stage2_bundle["subtype_hint_diagnostics"]
        default_output = "data/processed/stage2_distill.jsonl"
    else:
        if not args.repair_artifact:
            raise ValueError("--repair-artifact is required for selection-profile=stage3")
        dataset = build_repair_set(
            examples,
            repair_artifact=args.repair_artifact,
            prompt_mode=prompt_mode,
            trace_style=trace_style,
            include_metadata=include_metadata,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            top_k=args.top_k,
            replay_input=args.replay_input,
            replay_ratio=args.replay_ratio,
            tokenizer=tokenizer,
        )
        report = summarise_repair_sft(dataset)
        default_output = "data/processed/stage3_repair.jsonl"

    output_path = args.output or config.get("output") or default_output
    write_jsonl(output_path, dataset)
    if args.report_output:
        write_json(args.report_output, report)


if __name__ == "__main__":
    main()
