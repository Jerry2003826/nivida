from __future__ import annotations

from dataclasses import dataclass

from src.competition.metrics import exact_match
from src.teacher.chain_search import CandidateChain


@dataclass(slots=True)
class PseudoLabelDecision:
    accept: bool
    reason: str
    confidence: float


def accept_programmatic_label(candidate: CandidateChain, target_answer: str | None, *, threshold: float = 0.8) -> PseudoLabelDecision:
    if target_answer is not None and candidate.query_prediction is not None and exact_match(candidate.query_prediction, target_answer):
        return PseudoLabelDecision(accept=True, reason="matches_reference", confidence=1.0)
    if candidate.confidence >= threshold and candidate.exact_ratio >= 1.0:
        return PseudoLabelDecision(accept=True, reason="high_confidence_solver_trace", confidence=candidate.confidence)
    return PseudoLabelDecision(accept=False, reason="low_confidence_or_incorrect", confidence=candidate.confidence)
