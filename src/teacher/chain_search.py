from __future__ import annotations

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import re
from typing import Any

from src.competition.metrics import competition_numeric_match
from src.competition.schema import PuzzleExample
from src.teacher.atomic_ops import AtomicOp
from src.teacher.op_catalog import build_default_catalog
from src.teacher.program_signature import normalize_family_alias


def _similarity(prediction: str, target: str) -> float:
    if prediction == target:
        return 1.0
    if competition_numeric_match(prediction, target):
        return 0.98
    return SequenceMatcher(None, prediction, target).ratio()


def _is_binary_string(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and set(stripped) <= {"0", "1"}


def _query_char_coverage(prediction: str | None, query: str | None) -> float:
    if not prediction or not query:
        return 0.0
    remaining = list(query)
    covered = 0
    for char in prediction:
        if char not in remaining:
            continue
        covered += 1
        remaining.remove(char)
    return covered / max(1, len(prediction))


def _query_length_mode_bonus(prediction: str | None, support_targets: list[str]) -> float:
    if prediction is None or not support_targets:
        return 0.0
    lengths = sorted(len(target) for target in support_targets)
    if not lengths:
        return 0.0
    median_length = lengths[len(lengths) // 2]
    counts = {length: lengths.count(length) for length in set(lengths)}
    modal_length = min(
        counts,
        key=lambda length: (
            -counts[length],
            abs(length - median_length),
            length,
        ),
    )
    return 0.003 if len(prediction) == modal_length else 0.0


_EQUATION_PATTERN = re.compile(r"^\s*\d+\D\d+\s*$")


@dataclass(slots=True)
class ChainStep:
    op_name: str
    family: str
    params: dict[str, Any]
    step_score: float
    explanation: str
    complexity_penalty: float = 0.0


@dataclass(slots=True)
class CandidateChain:
    steps: list[ChainStep]
    score: float
    exact_ratio: float
    confidence: float
    predictions: list[str]
    query_prediction: str | None
    debug: dict[str, Any] = field(default_factory=dict)

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "steps": [asdict(step) for step in self.steps],
            "score": self.score,
            "exact_ratio": self.exact_ratio,
            "confidence": self.confidence,
            "predictions": self.predictions,
            "query_prediction": self.query_prediction,
            "debug": self.debug,
        }


@dataclass(slots=True)
class _SearchState:
    transformed_inputs: list[str]
    query_value: str | None
    steps: list[ChainStep]
    score: float
    exact_ratio: float
    predictions: list[str]
    family_legality: float
    graph_prior_score: float
    complexity_penalty: float


class ChainSearchEngine:
    def __init__(
        self,
        ops: list[AtomicOp] | None = None,
        *,
        beam_width: int = 6,
        max_depth: int = 2,
        graph_prior: Any | None = None,
    ) -> None:
        self.ops = ops or build_default_catalog()
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.graph_prior = graph_prior

    def _equation_mode(
        self,
        examples: list[tuple[str, str]],
        query: str | None,
        family_hints: list[str] | None,
    ) -> str | None:
        normalized_hints = {normalize_family_alias(hint) for hint in family_hints or []}
        if "equation" not in normalized_hints:
            return None
        inputs = [input_text for input_text, _ in examples]
        if query is not None:
            inputs.append(query)
        if inputs and all(_EQUATION_PATTERN.match(text.strip()) for text in inputs):
            return "numeric"
        return "symbolic"

    def _expand_family_hints(self, family_hints: list[str] | None, *, equation_mode: str | None = None) -> set[str]:
        if not family_hints:
            return set()
        alias_map = {
            "bit": {"bit_manipulation", "bit_operations"},
            "gravity": {"gravity"},
            "unit": {"unit_conversion"},
            "cipher": {"cipher", "substitution_cipher"},
            "numeral": {"numeral", "base_conversion"},
            "equation": {"equation"} if equation_mode == "symbolic" else {"equation", "arithmetic_equation"},
        }
        expanded_hints: set[str] = set()
        for hint in list(dict.fromkeys(family_hints)):
            normalized = normalize_family_alias(hint)
            expanded_hints.update(alias_map.get(normalized, {normalized}))
        return expanded_hints

    def _prioritized_op_names(self, family: str | None, subtype: str | None, equation_mode: str | None) -> list[str]:
        family = normalize_family_alias(family)
        if family == "bit":
            if subtype in {"rotate", "bit_rotate"}:
                return ["binary_rotate_left", "binary_rotate_right"]
            if subtype in {"mask_logic", "bit_xor_mask"}:
                return ["binary_xor_mask", "binary_and_mask", "binary_or_mask", "bitwise_xor_constant", "bitwise_and_constant", "bitwise_or_constant"]
            if subtype in {"nibble_permute", "bit_nibble"}:
                return ["swap_nibbles", "binary_permutation", "binary_nibble_map", "reverse_bits"]
            if subtype in {"binary_affine", "bit_affine"}:
                return ["binary_affine_transform"]
            if subtype == "bit_permutation":
                return ["binary_permutation", "swap_nibbles", "reverse_bits"]
        if family == "cipher":
            if subtype in {"token_substitution", "cipher_token_sub"}:
                return ["vocabulary_cipher"]
            if subtype in {"char_substitution", "caesar_affine", "cipher_char_sub"}:
                return ["caesar_shift", "vocabulary_cipher", "fixed_substitution"]
            if subtype in {"substitution_permutation", "cipher_perm"}:
                return ["fixed_substitution", "reverse_tokens", "sort_tokens"]
            if subtype in {"partial_map_completion", "cipher_vocab"}:
                return ["vocabulary_cipher", "fixed_substitution"]
        if family == "equation":
            if equation_mode == "numeric":
                return ["binary_equation_rule", "add_constant", "multiply_constant", "affine_transform", "evaluate_expression"]
            if subtype == "equation_delete":
                return ["delete_characters"]
            if subtype == "equation_template":
                return ["operator_template"]
            if subtype == "equation_position":
                return ["position_transducer"]
            return ["position_transducer", "operator_template", "delete_characters"]
        if family == "unit":
            return ["unit_convert", "scale_measurement"] if subtype in {"convert", "unit_convert"} else ["scale_measurement", "unit_convert"]
        if family == "numeral":
            return ["decimal_to_roman"] if subtype in {"roman", "numeral_roman"} else ["decimal_to_binary", "decimal_to_hex", "binary_to_decimal"]
        if family == "gravity":
            return ["gravity_distance"]
        return []

    def _ordered_ops(
        self,
        family_hints: list[str] | None = None,
        *,
        equation_mode: str | None = None,
        subtype: str | None = None,
    ) -> list[AtomicOp]:
        expanded_hints = self._expand_family_hints(family_hints, equation_mode=equation_mode)
        if not expanded_hints:
            return self.ops
        prioritized = [op for op in self.ops if getattr(op, "family", "") in expanded_hints]
        if equation_mode == "numeric":
            prioritized = [
                op
                for op in prioritized
                if op.name not in {"position_transducer", "operator_template", "delete_characters"}
            ]
        family = normalize_family_alias((family_hints or [None])[0])
        prioritized_names = self._prioritized_op_names(family, subtype, equation_mode)
        if prioritized_names:
            ordered = [op for name in prioritized_names for op in prioritized if op.name == name]
            ordered.extend(op for op in prioritized if op.name not in prioritized_names)
            prioritized = ordered
        strict_hints = {"bit", "gravity", "unit", "numeral", "cipher", "equation"}
        if family_hints and any(normalize_family_alias(hint) in strict_hints for hint in family_hints) and prioritized:
            return prioritized
        remaining = [op for op in self.ops if op not in prioritized]
        return prioritized + remaining

    def _family_legality(self, steps: list[ChainStep], family_hints: list[str] | None, *, equation_mode: str | None = None) -> float:
        expanded_hints = self._expand_family_hints(family_hints, equation_mode=equation_mode)
        if not steps or not expanded_hints:
            return 1.0
        legal = sum(1 for step in steps if step.family in expanded_hints)
        return legal / len(steps)

    def _violates_family_constraints(
        self,
        predictions: list[str],
        query_value: str | None,
        family_hints: list[str] | None,
        *,
        equation_mode: str | None = None,
    ) -> bool:
        if not family_hints:
            return False
        normalized_hints = {normalize_family_alias(hint) for hint in family_hints}
        if "bit" in normalized_hints:
            if any(not _is_binary_string(prediction) for prediction in predictions):
                return True
            if query_value is not None and not _is_binary_string(query_value):
                return True
        return False

    def _graph_prior_bonus(self, steps: list[ChainStep]) -> float:
        if not self.graph_prior or not steps:
            return 0.0
        bonus = float(getattr(self.graph_prior, "start_prior", lambda _: 0.0)(steps[0].op_name)) * 0.05
        if len(steps) >= 2:
            bonus += float(
                getattr(self.graph_prior, "transition_prior", lambda _a, _b: 0.0)(steps[-2].op_name, steps[-1].op_name)
            ) * 0.05
        return bonus

    def _score_state(
        self,
        predictions: list[str],
        targets: list[str],
        depth: int,
        steps: list[ChainStep],
        family_hints: list[str] | None,
        *,
        equation_mode: str | None = None,
        complexity_penalty: float = 0.0,
    ) -> tuple[float, float, float, float]:
        per_example = [_similarity(prediction, target) for prediction, target in zip(predictions, targets)]
        exact_ratio = sum(prediction == target for prediction, target in zip(predictions, targets)) / max(1, len(targets))
        avg_score = sum(per_example) / max(1, len(per_example))
        family_legality = self._family_legality(steps, family_hints, equation_mode=equation_mode)
        graph_prior_bonus = self._graph_prior_bonus(steps)
        total = avg_score + 0.20 * exact_ratio - 0.04 * depth + 0.08 * family_legality + graph_prior_bonus - complexity_penalty
        return total, exact_ratio, family_legality, graph_prior_bonus

    def _state_sort_key(self, item: _SearchState) -> tuple[float, float, int, float, float]:
        return (
            -item.exact_ratio,
            -item.family_legality,
            len(item.steps),
            -item.graph_prior_score,
            -item.score,
        )

    def search(
        self,
        examples: list[tuple[str, str]],
        *,
        query: str | None = None,
        query_input: str | None = None,
        top_k: int = 5,
        family_hints: list[str] | None = None,
        subtype: str | None = None,
    ) -> list[CandidateChain]:
        if not examples:
            return []
        if query is None:
            query = query_input

        targets = [output for _, output in examples]
        initial_inputs = [input_text for input_text, _ in examples]
        equation_mode = self._equation_mode(examples, query, family_hints)
        ops = self._ordered_ops(family_hints, equation_mode=equation_mode, subtype=subtype)
        max_depth = self.max_depth
        normalized_hints = {normalize_family_alias(hint) for hint in family_hints or []}
        if normalized_hints & {"bit", "cipher"}:
            max_depth = max(max_depth, 3)
        beam = [
            _SearchState(
                transformed_inputs=initial_inputs,
                query_value=query,
                steps=[],
                score=0.0,
                exact_ratio=0.0,
                predictions=initial_inputs,
                family_legality=1.0,
                graph_prior_score=0.0,
                complexity_penalty=0.0,
            )
        ]
        completed: list[_SearchState] = []

        for depth in range(1, max_depth + 1):
            expanded: list[_SearchState] = []
            for state in beam:
                current_examples = list(zip(state.transformed_inputs, targets))
                for op in ops:
                    query_aware_params = getattr(op, "candidate_params_for_query", None)
                    if query_aware_params is not None and state.query_value is not None:
                        params_list = query_aware_params(current_examples, state.query_value)
                    else:
                        params_list = op.candidate_params(current_examples)
                    if not params_list:
                        continue
                    for params in params_list[:12]:
                        try:
                            predictions = [op.apply(value, params) for value in state.transformed_inputs]
                            query_value = None if state.query_value is None else op.apply(state.query_value, params)
                        except Exception:
                            continue
                        if self._violates_family_constraints(
                            predictions,
                            query_value,
                            family_hints,
                            equation_mode=equation_mode,
                        ):
                            continue
                        step_penalty = float(getattr(op, "complexity_penalty", lambda _params: 0.0)(params))
                        step = ChainStep(
                            op_name=op.name,
                            family=getattr(op, "family", "generic"),
                            params=params,
                            step_score=0.0,
                            explanation=f"{op.name}({op.describe_params(params)})",
                            complexity_penalty=step_penalty,
                        )
                        total_complexity_penalty = state.complexity_penalty + step_penalty
                        score, exact_ratio, family_legality, graph_prior_score = self._score_state(
                            predictions,
                            targets,
                            depth,
                            state.steps + [step],
                            family_hints,
                            equation_mode=equation_mode,
                            complexity_penalty=total_complexity_penalty,
                        )
                        if equation_mode == "symbolic":
                            score += 0.01 * _query_char_coverage(query_value, query)
                            score += _query_length_mode_bonus(query_value, targets)
                        step.step_score = score
                        expanded.append(
                            _SearchState(
                                transformed_inputs=predictions,
                                query_value=query_value,
                                steps=state.steps + [step],
                                score=score,
                                exact_ratio=exact_ratio,
                                predictions=predictions,
                                family_legality=family_legality,
                                graph_prior_score=graph_prior_score,
                                complexity_penalty=total_complexity_penalty,
                            )
                        )
            expanded.sort(key=self._state_sort_key)
            dedup: dict[str, _SearchState] = {}
            for item in expanded:
                signature = " -> ".join(f"{step.op_name}:{sorted(step.params.items())}" for step in item.steps)
                if signature not in dedup:
                    dedup[signature] = item
            beam = list(dedup.values())[: self.beam_width]
            completed.extend(beam)

        completed.sort(key=self._state_sort_key)
        if not completed:
            return []

        best_score = completed[0].score
        second_score = completed[1].score if len(completed) > 1 else 0.0
        top_candidates: list[CandidateChain] = []
        for state in completed[:top_k]:
            confidence = min(
                1.0,
                max(
                    0.0,
                    0.50 * state.exact_ratio
                    + 0.20 * state.family_legality
                    + 0.20 * state.score
                    + 0.10 * (best_score - second_score),
                ),
            )
            top_candidates.append(
                CandidateChain(
                    steps=state.steps,
                    score=state.score,
                    exact_ratio=state.exact_ratio,
                    confidence=confidence,
                    predictions=state.predictions,
                    query_prediction=state.query_value,
                    debug={
                        "final_predictions": state.predictions,
                        "depth": len(state.steps),
                        "margin_to_best": best_score - state.score,
                        "family_legality": state.family_legality,
                        "graph_prior_score": state.graph_prior_score,
                        "complexity_penalty": state.complexity_penalty,
                        "equation_mode": equation_mode,
                        "query_char_coverage": _query_char_coverage(state.query_value, query),
                        "template_rank_features": [
                            step.params.get("template_rank_features")
                            for step in state.steps
                            if isinstance(step.params, dict)
                            and step.params.get("template_rank_features") is not None
                        ],
                    },
                )
            )
        return top_candidates

    def solve_example(self, example: PuzzleExample, *, top_k: int = 5) -> list[CandidateChain]:
        pairs = [(pair.input, pair.output) for pair in example.parsed_examples]
        return self.search(
            pairs,
            query=example.query,
            top_k=top_k,
            family_hints=[example.metadata.official_family] if example.metadata.official_family else None,
            subtype=example.metadata.subtype,
        )
