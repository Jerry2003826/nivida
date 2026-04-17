from __future__ import annotations

from dataclasses import dataclass


BOXED_TOKEN = r"\boxed{"


@dataclass(slots=True)
class BoxedAnswerResult:
    answer: str | None
    boxed_answers: list[str]
    is_valid: bool
    error: str | None = None


def extract_all_boxed_answers(text: str | None) -> list[str]:
    """Extract all `\\boxed{...}` spans while respecting nested braces."""
    if text is None:
        return []

    answers: list[str] = []
    cursor = 0
    while True:
        start = text.find(BOXED_TOKEN, cursor)
        if start < 0:
            break
        idx = start + len(BOXED_TOKEN)
        depth = 1
        while idx < len(text):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    answers.append(text[start + len(BOXED_TOKEN):idx].strip())
                    cursor = idx + 1
                    break
            idx += 1
        else:
            break
    return answers


def extract_single_boxed_answer(text: str | None) -> BoxedAnswerResult:
    answers = extract_all_boxed_answers(text)
    if not answers:
        return BoxedAnswerResult(answer=None, boxed_answers=[], is_valid=False, error="missing_boxed_answer")
    if len(answers) > 1:
        return BoxedAnswerResult(
            answer=None,
            boxed_answers=answers,
            is_valid=False,
            error="multiple_boxed_answers",
        )
    if not answers[0]:
        return BoxedAnswerResult(answer=None, boxed_answers=answers, is_valid=False, error="empty_boxed_answer")
    return BoxedAnswerResult(answer=answers[0], boxed_answers=answers, is_valid=True, error=None)
