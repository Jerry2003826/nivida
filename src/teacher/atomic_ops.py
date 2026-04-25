from __future__ import annotations

import ast
import itertools
import random
import re
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from difflib import SequenceMatcher
from typing import Any, Iterable


def _format_number(value: float | int) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _safe_int(text: str) -> int | None:
    text = text.strip()
    try:
        if text.lower().startswith(("0x", "-0x", "+0x")):
            return int(text, 16)
        if text.lower().startswith(("0b", "-0b", "+0b")):
            return int(text, 2)
        if text.lower().startswith(("0o", "-0o", "+0o")):
            return int(text, 8)
        return int(text)
    except ValueError:
        return None


def _safe_float(text: str) -> float | None:
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        return None


def _format_fixed(value: float, decimals: int = 2) -> str:
    quantum = Decimal("1") if decimals <= 0 else Decimal("1").scaleb(-decimals)
    return format(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP), "f")


def _format_trimmed(value: float, decimals: int = 2) -> str:
    formatted = _format_fixed(value, decimals)
    if "." not in formatted:
        return formatted
    trimmed = formatted.rstrip("0").rstrip(".")
    if "." not in trimmed:
        return f"{trimmed}.0"
    return trimmed


_MEASUREMENT_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([A-Za-z]+)?\s*$")
_EQUATION_RE = re.compile(r"^\s*(\d+)(\D)(\d+)\s*$")


def _parse_measurement(text: str) -> tuple[float, str | None]:
    match = _MEASUREMENT_RE.match(text.strip())
    if not match:
        raise ValueError("not a measurement")
    value = float(match.group(1))
    unit = match.group(2).lower() if match.group(2) else None
    return value, unit


def _parse_binary_equation(text: str) -> tuple[str, str, str] | None:
    match = _EQUATION_RE.match(text.strip())
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def _decimal_places(text: str) -> int:
    stripped = text.strip()
    if "." not in stripped:
        return 0
    return len(stripped.rsplit(".", 1)[1])


def _most_common_decimal_places(texts: Iterable[str], default: int = 2) -> int:
    counts: dict[int, int] = {}
    for text in texts:
        places = _decimal_places(text)
        counts[places] = counts.get(places, 0) + 1
    if not counts:
        return default
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _normalise_numeric_string(text: str, strip_mode: str = "none") -> str:
    if strip_mode == "strip_leading":
        stripped = text.lstrip("0")
        return stripped or "0"
    if strip_mode == "strip_all":
        stripped = text.strip("0")
        return stripped or "0"
    return text


def _is_binary_string(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and set(stripped) <= {"0", "1"}


def _word_pattern(word: str) -> tuple[int, ...]:
    mapping: dict[str, int] = {}
    pattern: list[int] = []
    next_id = 0
    for char in word:
        if char not in mapping:
            mapping[char] = next_id
            next_id += 1
        pattern.append(mapping[char])
    return tuple(pattern)


def _render_template_tokens(text: str, tokens: list[tuple[str, int | str]]) -> str:
    rendered: list[str] = []
    for kind, value in tokens:
        if kind == "pos":
            rendered.append(text[int(value)])
        else:
            rendered.append(str(value))
    return "".join(rendered)


def _template_generalisation_key(tokens: list[tuple[str, int | str]]) -> tuple[int, int, int, int, tuple[int, ...]]:
    source_positions = [int(value) for kind, value in tokens if kind == "pos"]
    literal_count = sum(1 for kind, _ in tokens if kind == "lit")
    repeated_positions = len(source_positions) - len(set(source_positions))
    backward_edges = sum(
        1 for left, right in zip(source_positions, source_positions[1:]) if right < left
    )
    skipped_unique_positions = 0
    if source_positions:
        span = max(source_positions) - min(source_positions) + 1
        skipped_unique_positions = span - len(set(source_positions))
    return (
        literal_count,
        repeated_positions,
        backward_edges,
        skipped_unique_positions,
        tuple(source_positions),
    )


def _template_rank_features(templates: dict[str, list[tuple[str, int | str]]]) -> dict[str, int]:
    literal_count = 0
    repeated_positions = 0
    backward_edges = 0
    skipped_unique_positions = 0
    for template in templates.values():
        source_positions = [int(value) for kind, value in template if kind == "pos"]
        literal_count += sum(1 for kind, _ in template if kind == "lit")
        repeated_positions += len(source_positions) - len(set(source_positions))
        backward_edges += sum(
            1 for left, right in zip(source_positions, source_positions[1:]) if right < left
        )
        if source_positions:
            span = max(source_positions) - min(source_positions) + 1
            skipped_unique_positions += span - len(set(source_positions))
    return {
        "literal_count": literal_count,
        "repeated_positions": repeated_positions,
        "backward_edges": backward_edges,
        "skipped_unique_positions": skipped_unique_positions,
        "template_count": len(templates),
    }


def _binary_vector(text: str) -> list[int]:
    stripped = text.strip()
    if not _is_binary_string(stripped):
        raise ValueError("not a binary string")
    return [1 if bit == "1" else 0 for bit in stripped]


def _solve_gf2_system(rows: list[list[int]], targets: list[int]) -> list[int] | None:
    if not rows:
        return []
    n_cols = len(rows[0])
    augmented = [row[:] + [target] for row, target in zip(rows, targets)]
    pivot_row = 0
    pivot_cols: list[int] = []

    for col in range(n_cols):
        pivot = next((row_index for row_index in range(pivot_row, len(augmented)) if augmented[row_index][col]), None)
        if pivot is None:
            continue
        augmented[pivot_row], augmented[pivot] = augmented[pivot], augmented[pivot_row]
        pivot_cols.append(col)
        for row_index in range(len(augmented)):
            if row_index != pivot_row and augmented[row_index][col]:
                for inner_col in range(col, n_cols + 1):
                    augmented[row_index][inner_col] ^= augmented[pivot_row][inner_col]
        pivot_row += 1
        if pivot_row == len(augmented):
            break

    for row in augmented:
        if not any(row[:n_cols]) and row[n_cols]:
            return None

    free_cols = [col for col in range(n_cols) if col not in pivot_cols]

    def _solution_for(free_values: tuple[int, ...]) -> list[int]:
        solution = [0] * n_cols
        for col, value in zip(free_cols, free_values):
            solution[col] = int(value) & 1
        for row_index, col in enumerate(pivot_cols):
            value = augmented[row_index][n_cols]
            for free_col in free_cols:
                if augmented[row_index][free_col] and solution[free_col]:
                    value ^= 1
            solution[col] = value
        return solution

    if len(free_cols) > 16:
        return _solution_for(tuple(0 for _ in free_cols))

    solutions = [_solution_for(values) for values in itertools.product((0, 1), repeat=len(free_cols))]
    return min(solutions, key=lambda solution: (sum(solution[:-1]), solution[-1], tuple(solution)))


def _int_to_roman(value: int) -> str:
    if value <= 0 or value >= 4000:
        raise ValueError("roman numeral range must be 1..3999")
    numerals = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    result: list[str] = []
    remaining = value
    for amount, symbol in numerals:
        while remaining >= amount:
            result.append(symbol)
            remaining -= amount
    return "".join(result)


WONDERLAND_VOCAB = {
    "alice",
    "ancient",
    "bird",
    "book",
    "castle",
    "cat",
    "chases",
    "clever",
    "colorful",
    "creates",
    "crystal",
    "curious",
    "dark",
    "discovers",
    "door",
    "dragon",
    "draws",
    "dreams",
    "explores",
    "follows",
    "forest",
    "found",
    "garden",
    "golden",
    "hatter",
    "imagines",
    "in",
    "inside",
    "king",
    "knight",
    "library",
    "magical",
    "mirror",
    "mouse",
    "mountain",
    "mysterious",
    "near",
    "ocean",
    "palace",
    "puzzle",
    "princess",
    "queen",
    "rabbit",
    "reads",
    "secret",
    "sees",
    "silver",
    "story",
    "student",
    "studies",
    "teacher",
    "the",
    "through",
    "turtle",
    "under",
    "valley",
    "village",
    "watches",
    "wise",
    "wizard",
    "wonderland",
}


def _score_predictions(predictions: Iterable[str], targets: Iterable[str]) -> float:
    values = []
    for prediction, target in zip(predictions, targets):
        if prediction == target:
            values.append(1.0)
        else:
            values.append(SequenceMatcher(None, prediction, target).ratio())
    return sum(values) / max(1, len(values))


def _rotate_left(text: str, k: int) -> str:
    if not text:
        return text
    k %= len(text)
    return text[k:] + text[:k]


def _rotate_right(text: str, k: int) -> str:
    if not text:
        return text
    k %= len(text)
    return text[-k:] + text[:-k] if k else text


def _safe_eval_expression(expression: str) -> float:
    node = ast.parse(expression, mode="eval")

    def _eval(item: ast.AST) -> float:
        if isinstance(item, ast.Expression):
            return _eval(item.body)
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            return float(item.value)
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.UAdd, ast.USub)):
            value = _eval(item.operand)
            return value if isinstance(item.op, ast.UAdd) else -value
        if isinstance(item, ast.BinOp) and isinstance(
            item.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
        ):
            lhs = _eval(item.left)
            rhs = _eval(item.right)
            if isinstance(item.op, ast.Add):
                return lhs + rhs
            if isinstance(item.op, ast.Sub):
                return lhs - rhs
            if isinstance(item.op, ast.Mult):
                return lhs * rhs
            if isinstance(item.op, ast.Div):
                return lhs / rhs
            if isinstance(item.op, ast.FloorDiv):
                return lhs // rhs
            if isinstance(item.op, ast.Mod):
                return lhs % rhs
            return lhs**rhs
        raise ValueError(f"Unsupported expression: {ast.dump(item)}")

    return _eval(node)


@dataclass(slots=True)
class FitResult:
    success: bool
    params: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    predictions: list[str] = field(default_factory=list)
    note: str = ""


class AtomicOp:
    name: str = "atomic_op"
    family: str = "generic"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        return [{}]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        raise NotImplementedError

    def fit(self, examples: list[tuple[str, str]]) -> FitResult:
        best = FitResult(success=False, score=-1.0)
        targets = [output for _, output in examples]
        for params in self.candidate_params(examples):
            try:
                predictions = [self.apply(input_text, params) for input_text, _ in examples]
            except Exception:
                continue
            score = _score_predictions(predictions, targets)
            if score > best.score:
                best = FitResult(success=score >= 0.999 or score >= 0.5, params=params, score=score, predictions=predictions, note=str(params))
        return best

    def can_explain(self, examples: list[tuple[str, str]]) -> bool:
        return self.fit(examples).success

    def describe_params(self, params: dict[str, Any]) -> str:
        return ", ".join(f"{key}={value}" for key, value in params.items()) if params else "no_params"

    def complexity_penalty(self, params: dict[str, Any]) -> float:
        return 0.0

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        raise NotImplementedError


class IdentityOp(AtomicOp):
    name = "identity"
    family = "reverse_reorder"

    def apply(self, text: str, params: dict[str, Any]) -> str:
        return text

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "".join(rng.choice("abcdef0123") for _ in range(5))
        return source, source, {}


class ReverseStringOp(AtomicOp):
    name = "reverse_string"
    family = "reverse_reorder"

    def apply(self, text: str, params: dict[str, Any]) -> str:
        return text[::-1]

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "".join(rng.choice("abcdef") for _ in range(6))
        return source, source[::-1], {}


class ReverseTokensOp(AtomicOp):
    name = "reverse_tokens"
    family = "reverse_reorder"

    def apply(self, text: str, params: dict[str, Any]) -> str:
        return " ".join(text.split()[::-1]) if text.strip() else text

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        tokens = ["red", "blue", "green"]
        source = " ".join(tokens)
        return source, " ".join(tokens[::-1]), {}


class RotateLeftOp(AtomicOp):
    name = "rotate_left"
    family = "reverse_reorder"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        lengths = [len(input_text) for input_text, _ in examples if input_text]
        if not lengths:
            return []
        max_len = min(max(lengths), 8)
        return [{"k": k} for k in range(1, max_len)]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        return _rotate_left(text, int(params["k"]))

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "".join(rng.choice("abcdef") for _ in range(6))
        k = rng.randint(1, len(source) - 1)
        return source, _rotate_left(source, k), {"k": k}


class RotateRightOp(AtomicOp):
    name = "rotate_right"
    family = "reverse_reorder"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        lengths = [len(input_text) for input_text, _ in examples if input_text]
        if not lengths:
            return []
        max_len = min(max(lengths), 8)
        return [{"k": k} for k in range(1, max_len)]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        return _rotate_right(text, int(params["k"]))

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "".join(rng.choice("abcdef") for _ in range(6))
        k = rng.randint(1, len(source) - 1)
        return source, _rotate_right(source, k), {"k": k}


class SortCharsOp(AtomicOp):
    name = "sort_chars"
    family = "reverse_reorder"

    def apply(self, text: str, params: dict[str, Any]) -> str:
        return "".join(sorted(text))

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "".join(rng.choice("cabfed") for _ in range(6))
        return source, "".join(sorted(source)), {}


class SortTokensOp(AtomicOp):
    name = "sort_tokens"
    family = "reverse_reorder"

    def apply(self, text: str, params: dict[str, Any]) -> str:
        tokens = text.split()
        return " ".join(sorted(tokens)) if tokens else text

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        tokens = ["pear", "apple", "banana"]
        rng.shuffle(tokens)
        source = " ".join(tokens)
        return source, "apple banana pear", {}


class UniqueCharsOp(AtomicOp):
    name = "stable_unique_chars"
    family = "count_filter_aggregation"

    def apply(self, text: str, params: dict[str, Any]) -> str:
        seen: set[str] = set()
        result: list[str] = []
        for char in text:
            if char not in seen:
                seen.add(char)
                result.append(char)
        return "".join(result)

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "aabbccdde"
        return source, "abcde", {}


class CaesarShiftOp(AtomicOp):
    name = "caesar_shift"
    family = "substitution_cipher"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        candidates: set[int] = set()
        for input_text, output_text in examples:
            if len(input_text) != len(output_text):
                continue
            for src, dst in zip(input_text, output_text):
                if src.isalpha() and dst.isalpha():
                    candidates.add((ord(dst.lower()) - ord(src.lower())) % 26)
                    break
        if not candidates:
            candidates = set(range(-3, 4))
        return [{"shift": shift} for shift in sorted(candidates)]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        shift = int(params["shift"])
        output_chars: list[str] = []
        for char in text:
            if char.isalpha():
                base = ord("A" if char.isupper() else "a")
                output_chars.append(chr(base + ((ord(char) - base + shift) % 26)))
            else:
                output_chars.append(char)
        return "".join(output_chars)

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "code"
        shift = rng.randint(1, 5)
        return source, self.apply(source, {"shift": shift}), {"shift": shift}


class FixedSubstitutionOp(AtomicOp):
    name = "fixed_substitution"
    family = "substitution_cipher"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        mapping: dict[str, str] = {}
        reverse_mapping: dict[str, str] = {}
        for input_text, output_text in examples:
            if len(input_text) != len(output_text):
                return []
            for src, dst in zip(input_text, output_text):
                if mapping.setdefault(src, dst) != dst:
                    return []
                if reverse_mapping.setdefault(dst, src) != src:
                    return []
        return [{"mapping": mapping}] if mapping else []

    def apply(self, text: str, params: dict[str, Any]) -> str:
        mapping = params["mapping"]
        return "".join(mapping.get(char, char) for char in text)

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        mapping = {"a": "x", "b": "y", "c": "z"}
        source = "abca"
        return source, self.apply(source, {"mapping": mapping}), {"mapping": mapping}


class VocabularyCipherOp(AtomicOp):
    name = "vocabulary_cipher"
    family = "cipher"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        mapping: dict[str, str] = {}
        reverse_mapping: dict[str, str] = {}
        token_frequency: dict[str, int] = {}
        for input_text, output_text in examples:
            if len(input_text) != len(output_text):
                return []
            for src, dst in zip(input_text, output_text):
                if src == " " and dst == " ":
                    continue
                if src == " " or dst == " ":
                    return []
                if src.isalpha() and dst.isalpha():
                    if src in mapping and mapping[src] != dst:
                        # Keep partial mapping for letters that are already consistent elsewhere.
                        continue
                    if dst in reverse_mapping and reverse_mapping[dst] != src:
                        continue
                    mapping[src] = dst
                    reverse_mapping[dst] = src
            for token in output_text.split():
                token_frequency[token] = token_frequency.get(token, 0) + 1
        vocab = sorted(WONDERLAND_VOCAB | set(token_frequency))
        return [{"mapping": mapping, "reverse_mapping": reverse_mapping, "vocab": vocab, "token_frequency": token_frequency}]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        mapping = dict(params.get("mapping", {}))
        reverse_mapping = dict(params.get("reverse_mapping", {}))
        vocab = list(params.get("vocab", []))
        token_frequency = dict(params.get("token_frequency", {}))

        decoded_tokens: list[str] = []
        for token in text.split():
            direct = "".join(mapping.get(char, "?") for char in token)
            if "?" not in direct:
                decoded_tokens.append(direct)
                continue

            candidates: list[tuple[int, str]] = []
            token_pattern = _word_pattern(token)
            for candidate in vocab:
                if len(candidate) != len(token):
                    continue
                if _word_pattern(candidate) != token_pattern:
                    continue
                local_ok = True
                score = token_frequency.get(candidate, 0)
                temp_new: list[tuple[str, str]] = []
                for enc_char, plain_char in zip(token, candidate):
                    existing_plain = mapping.get(enc_char)
                    if existing_plain is not None and existing_plain != plain_char:
                        local_ok = False
                        break
                    existing_enc = reverse_mapping.get(plain_char)
                    if existing_enc is not None and existing_enc != enc_char:
                        local_ok = False
                        break
                    if existing_plain == plain_char:
                        score += 3
                    elif existing_plain is None:
                        temp_new.append((enc_char, plain_char))
                if local_ok:
                    # Prefer candidates that explain more unknown letters while remaining common.
                    score += len(temp_new)
                    candidates.append((score, candidate))
            if candidates:
                candidates.sort(key=lambda item: (-item[0], item[1]))
                chosen = candidates[0][1]
                for enc_char, plain_char in zip(token, chosen):
                    mapping.setdefault(enc_char, plain_char)
                    reverse_mapping.setdefault(plain_char, enc_char)
                decoded_tokens.append(chosen)
            else:
                decoded_tokens.append("".join(mapping.get(char, char) for char in token))
        return " ".join(decoded_tokens)

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        mapping = {"a": "x", "b": "y", "c": "z", "t": "q"}
        source = "xyz qyz"
        params = {"mapping": {v: k for k, v in mapping.items()}, "reverse_mapping": mapping, "vocab": sorted(WONDERLAND_VOCAB), "token_frequency": {}}
        return source, self.apply(source, params), params


class DecimalToBinaryOp(AtomicOp):
    name = "decimal_to_binary"
    family = "base_conversion"

    def apply(self, text: str, params: dict[str, Any]) -> str:
        value = _safe_int(text)
        if value is None:
            raise ValueError("not a decimal integer")
        return format(value, "b")

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        value = rng.randint(1, 32)
        return str(value), format(value, "b"), {}


class BinaryToDecimalOp(AtomicOp):
    name = "binary_to_decimal"
    family = "base_conversion"

    def apply(self, text: str, params: dict[str, Any]) -> str:
        stripped = text.strip()
        if not stripped or any(char not in "01" for char in stripped):
            raise ValueError("not a binary integer")
        return str(int(stripped, 2))

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        value = rng.randint(1, 31)
        source = format(value, "b")
        return source, str(value), {}


class DecimalToHexOp(AtomicOp):
    name = "decimal_to_hex"
    family = "base_conversion"

    def apply(self, text: str, params: dict[str, Any]) -> str:
        value = _safe_int(text)
        if value is None:
            raise ValueError("not a decimal integer")
        return format(value, "x")

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        value = rng.randint(16, 255)
        return str(value), format(value, "x"), {}


class DecimalToRomanOp(AtomicOp):
    name = "decimal_to_roman"
    family = "numeral"

    def apply(self, text: str, params: dict[str, Any]) -> str:
        value = _safe_int(text)
        if value is None:
            raise ValueError("not an integer")
        return _int_to_roman(value)

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        value = rng.randint(1, 199)
        return str(value), _int_to_roman(value), {}


class OperatorTemplateOp(AtomicOp):
    name = "operator_template"
    family = "equation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        if not examples:
            return []
        input_lengths = {len(input_text) for input_text, _ in examples}
        if len(input_lengths) != 1:
            return []
        input_length = input_lengths.pop()
        if input_length <= 1:
            return []
        if not any(any(not char.isalnum() for char in input_text) for input_text, _ in examples):
            return []

        def _enumerate_templates(input_text: str, output_text: str, *, max_results: int = 64) -> list[list[tuple[str, int | str]]]:
            results: list[list[tuple[str, int | str]]] = []

            def _search(out_idx: int, current: list[tuple[str, int | str]]) -> None:
                if len(results) >= max_results:
                    return
                if out_idx >= len(output_text):
                    results.append(list(current))
                    return
                target_char = output_text[out_idx]
                matched_positions = [index for index, source_char in enumerate(input_text) if source_char == target_char]
                for index in matched_positions:
                    current.append(("pos", index))
                    _search(out_idx + 1, current)
                    current.pop()
                current.append(("lit", target_char))
                _search(out_idx + 1, current)
                current.pop()

            _search(0, [])
            results.sort(key=_template_generalisation_key)
            return results

        key_positions = [
            index
            for index in range(input_length)
            if any(not example_input[index].isalnum() for example_input, _ in examples)
        ]
        primary_candidates: list[dict[str, Any]] = []
        alternate_candidates: list[dict[str, Any]] = []
        for key_position in key_positions:
            template_options_by_key: dict[str, list[list[tuple[str, int | str]]]] = {}
            valid = True
            for key_char in sorted({input_text[key_position] for input_text, _ in examples}):
                grouped_examples = [(input_text, output_text) for input_text, output_text in examples if input_text[key_position] == key_char]
                template_options = _enumerate_templates(grouped_examples[0][0], grouped_examples[0][1])
                valid_templates = [
                    template
                    for template in template_options
                    if all(_render_template_tokens(input_text, template) == output_text for input_text, output_text in grouped_examples)
                ]
                if not valid_templates:
                    valid = False
                    break
                template_options_by_key[key_char] = valid_templates[:4]
            if valid and template_options_by_key:
                key_chars = sorted(template_options_by_key)
                combinations = itertools.product(*(template_options_by_key[key_char] for key_char in key_chars))
                key_position_candidates: list[dict[str, Any]] = []
                for choice_index, choice in enumerate(combinations):
                    if choice_index >= 24:
                        break
                    templates = dict(zip(key_chars, choice))
                    key_position_candidates.append(
                        {
                            "key_position": key_position,
                            "templates": templates,
                            "input_length": input_length,
                            "template_rank_features": _template_rank_features(templates),
                        }
                    )
                if key_position_candidates:
                    primary_candidates.append(key_position_candidates[0])
                    alternate_candidates.extend(key_position_candidates[1:])
        alternate_candidates.sort(
            key=lambda params: (
                params["template_rank_features"]["literal_count"],
                params["template_rank_features"]["repeated_positions"],
                params["template_rank_features"]["backward_edges"],
                params["template_rank_features"]["skipped_unique_positions"],
                params["key_position"],
            )
        )
        return primary_candidates + alternate_candidates

    def apply(self, text: str, params: dict[str, Any]) -> str:
        if len(text) != int(params["input_length"]):
            raise ValueError("unexpected input length")
        key_char = text[int(params["key_position"])]
        templates = params["templates"]
        if key_char not in templates:
            raise ValueError("unsupported operator token")
        result = _render_template_tokens(text, templates[key_char])
        if not result:
            raise ValueError("empty template output")
        return result

    def complexity_penalty(self, params: dict[str, Any]) -> float:
        templates = params.get("templates", {})
        literal_count = sum(1 for template in templates.values() for kind, _ in template if kind == "lit")
        repeated_positions = sum(
            len([value for kind, value in template if kind == "pos"])
            - len({int(value) for kind, value in template if kind == "pos"})
            for template in templates.values()
        )
        backward_edges = sum(
            sum(
                1
                for left, right in zip(
                    [int(value) for kind, value in template if kind == "pos"],
                    [int(value) for kind, value in template if kind == "pos"][1:],
                )
                if right < left
            )
            for template in templates.values()
        )
        return (
            0.01 * len(templates)
            + 0.005 * literal_count
            + 0.004 * repeated_positions
            + 0.002 * backward_edges
        )

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        params = {
            "key_position": 2,
            "templates": {
                "*": [("pos", 0), ("pos", 1), ("pos", 3), ("pos", 4)],
                "-": [("pos", 2), ("pos", 1)],
            },
            "input_length": 5,
        }
        source = r"\(*[^"
        return source, self.apply(source, params), params


class PositionTransducerOp(AtomicOp):
    name = "position_transducer"
    family = "equation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        if not examples:
            return []
        if any(not output_text for _, output_text in examples):
            return []
        if not any(
            any(not char.isalnum() for char in input_text + output_text)
            for input_text, output_text in examples
        ):
            return []
        input_lengths = {len(input_text) for input_text, _ in examples}
        if len(input_lengths) != 1:
            return []
        input_length = input_lengths.pop()
        if any(len(output_text) > input_length for _, output_text in examples):
            return []

        position_maps: list[dict[str, str]] = [dict() for _ in range(input_length)]

        def _enumerate_assignments(
            input_text: str,
            output_text: str,
            base_maps: list[dict[str, str]],
            *,
            max_results: int = 24,
        ) -> list[list[dict[str, str]]]:
            results: list[list[dict[str, str]]] = []

            def _search(
                pos: int,
                out_idx: int,
                current_maps: list[dict[str, str]],
            ) -> None:
                if len(results) >= max_results:
                    return
                if pos == input_length:
                    if out_idx == len(output_text):
                        results.append([mapping.copy() for mapping in current_maps])
                    return

                char = input_text[pos]
                existing = current_maps[pos].get(char)

                if existing in (None, ""):
                    next_maps = current_maps
                    if existing is None:
                        next_maps = [mapping.copy() for mapping in current_maps]
                        next_maps[pos][char] = ""
                    _search(pos + 1, out_idx, next_maps)

                if out_idx >= len(output_text):
                    return

                emit_char = output_text[out_idx]
                if existing in (None, emit_char):
                    next_maps = current_maps
                    if existing is None:
                        next_maps = [mapping.copy() for mapping in current_maps]
                        next_maps[pos][char] = emit_char
                    _search(pos + 1, out_idx + 1, next_maps)

            _search(0, 0, [mapping.copy() for mapping in base_maps])
            results.sort(
                key=lambda candidate: (
                    sum(
                        1
                        for index, mapping in enumerate(candidate)
                        for char, emit in mapping.items()
                        if base_maps[index].get(char) is None and emit not in ("", char)
                    ),
                    sum(
                        1
                        for index, mapping in enumerate(candidate)
                        for char, emit in mapping.items()
                        if base_maps[index].get(char) is None and emit == ""
                    ),
                )
            )
            return results

        def _backtrack(example_idx: int, current_maps: list[dict[str, str]]) -> list[dict[str, str]] | None:
            if example_idx >= len(examples):
                return current_maps
            input_text, output_text = examples[example_idx]
            for candidate_maps in _enumerate_assignments(input_text, output_text, current_maps):
                solved = _backtrack(example_idx + 1, candidate_maps)
                if solved is not None:
                    return solved
            return None

        solved_maps = _backtrack(0, position_maps)
        if solved_maps is None:
            return []

        defaults: list[str] = []
        for position_map in solved_maps:
            values = list(position_map.items())
            if values and all(emit == "" for _, emit in values):
                defaults.append("drop")
                continue
            if values and all(emit == char for char, emit in values):
                defaults.append("identity")
                continue
            defaults.append("drop")

        return [{"position_maps": solved_maps, "defaults": defaults, "input_length": input_length}]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        position_maps = params["position_maps"]
        defaults = params.get("defaults", [])
        expected_length = int(params.get("input_length", len(position_maps)))
        if len(text) != expected_length:
            raise ValueError("unexpected input length")
        output_chars: list[str] = []
        for index, char in enumerate(text):
            position_map = position_maps[index]
            emit = position_map.get(char)
            if emit is None:
                emit = char if index < len(defaults) and defaults[index] == "identity" else ""
            output_chars.append(emit)
        result = "".join(output_chars)
        if not result:
            raise ValueError("empty template output")
        return result

    def complexity_penalty(self, params: dict[str, Any]) -> float:
        position_maps = params.get("position_maps", [])
        defaults = params.get("defaults", [])
        penalty = 0.0
        for index, position_map in enumerate(position_maps):
            default = defaults[index] if index < len(defaults) else "drop"
            for char, emit in position_map.items():
                if default == "identity" and emit == char:
                    continue
                penalty += 0.004 if emit == "" else 0.006
        return penalty

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        params = {
            "position_maps": [
                {"a": "a", "x": "x"},
                {"*": "", "+": ""},
                {"b": "b", "y": "y"},
            ],
            "defaults": ["identity", "drop", "identity"],
            "input_length": 3,
        }
        source = "a*b"
        return source, self.apply(source, params), params


class DeleteCharactersOp(AtomicOp):
    name = "delete_characters"
    family = "equation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        delete_chars: set[str] = set()
        for input_text, output_text in examples:
            input_chars = list(input_text)
            output_chars = list(output_text)
            output_index = 0
            current_delete: set[str] = set()
            for char in input_chars:
                if output_index < len(output_chars) and char == output_chars[output_index]:
                    output_index += 1
                else:
                    current_delete.add(char)
            if output_index != len(output_chars):
                return []
            reconstructed = "".join(char for char in input_text if char not in current_delete)
            if reconstructed != output_text:
                return []
            delete_chars |= current_delete
        if not delete_chars:
            return []
        return [{"delete_chars": "".join(sorted(delete_chars))}]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        delete_chars = set(params["delete_chars"])
        return "".join(char for char in text if char not in delete_chars)

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "a*b*c"
        params = {"delete_chars": "*"}
        return source, "abc", params


class BinaryEquationRuleOp(AtomicOp):
    name = "binary_equation_rule"
    family = "equation"

    def _feature_map(self, left_text: str, right_text: str) -> dict[str, str]:
        left_value = int(left_text)
        right_value = int(right_text)
        features = {
            "left": left_text,
            "right": right_text,
            "concat_lr": left_text + right_text,
            "concat_rl": right_text + left_text,
            "add": str(left_value + right_value),
            "sub": str(left_value - right_value),
            "rsub": str(right_value - left_value),
            "abs_sub": str(abs(left_value - right_value)),
            "mul": str(left_value * right_value),
        }
        if right_value != 0 and left_value % right_value == 0:
            features["div"] = str(left_value // right_value)
        if left_value != 0 and right_value % left_value == 0:
            features["rdiv"] = str(right_value // left_value)
        if right_value != 0:
            features["mod"] = str(left_value % right_value)
        if left_value != 0:
            features["rmod"] = str(right_value % left_value)
        if len(left_text) == len(right_text):
            digit_pairs = list(zip(left_text, right_text))
            same_order = [(int(lhs), int(rhs)) for lhs, rhs in digit_pairs]
            cross_order = [(int(lhs), int(rhs)) for lhs, rhs in zip(left_text, right_text[::-1])]
            features.update(
                {
                    "digit_add": "".join(str(lhs + rhs) for lhs, rhs in same_order),
                    "digit_abs": "".join(str(abs(lhs - rhs)) for lhs, rhs in same_order),
                    "digit_mul": "".join(str(lhs * rhs) for lhs, rhs in same_order),
                    "digit_add_cross": "".join(str(lhs + rhs) for lhs, rhs in cross_order),
                    "digit_abs_cross": "".join(str(abs(lhs - rhs)) for lhs, rhs in cross_order),
                    "digit_mul_cross": "".join(str(lhs * rhs) for lhs, rhs in cross_order),
                }
            )
        return features

    def _rule_candidates(self, left_text: str, right_text: str) -> list[dict[str, str]]:
        feature_map = self._feature_map(left_text, right_text)
        candidates: list[dict[str, str]] = []
        for feature_name in sorted(feature_map):
            for strip_mode in ["none", "strip_leading", "strip_all"]:
                candidates.append({"kind": "single", "feature": feature_name, "strip_mode": strip_mode})
        sortable = sorted(feature_map)
        for left_feature in sortable:
            for right_feature in sortable:
                if left_feature == right_feature:
                    continue
                for strip_mode in ["none", "strip_leading"]:
                    candidates.append(
                        {
                            "kind": "concat",
                            "left_feature": left_feature,
                            "right_feature": right_feature,
                            "strip_mode": strip_mode,
                        }
                    )
        return candidates

    def _render_rule(self, left_text: str, right_text: str, rule: dict[str, str]) -> str:
        if rule["kind"] == "lookup":
            key = f"{left_text}\t{right_text}"
            values = rule.get("values", {})
            if key not in values:
                raise ValueError("unsupported equation lookup")
            return str(values[key])
        feature_map = self._feature_map(left_text, right_text)
        if rule["kind"] == "single":
            return _normalise_numeric_string(feature_map[rule["feature"]], rule.get("strip_mode", "none"))
        left_value = _normalise_numeric_string(feature_map[rule["left_feature"]], rule.get("strip_mode", "none"))
        right_value = _normalise_numeric_string(feature_map[rule["right_feature"]], rule.get("strip_mode", "none"))
        return left_value + right_value

    def _rule_sort_key(self, rule: dict[str, str]) -> tuple[int, int, str]:
        feature_priority = {
            "left": 0,
            "right": 1,
            "concat_lr": 2,
            "concat_rl": 3,
            "add": 4,
            "abs_sub": 5,
            "sub": 6,
            "rsub": 7,
            "mul": 8,
            "digit_add": 9,
            "digit_abs": 10,
            "digit_mul": 11,
            "digit_add_cross": 12,
            "digit_abs_cross": 13,
            "digit_mul_cross": 14,
        }
        if rule["kind"] == "lookup":
            return (3, 0, "")
        if rule["kind"] == "single":
            feature = rule.get("feature", "")
            return (0, feature_priority.get(feature, 50), feature)
        left_feature = rule.get("left_feature", "")
        right_feature = rule.get("right_feature", "")
        return (
            1,
            feature_priority.get(left_feature, 50) + feature_priority.get(right_feature, 50),
            f"{left_feature}:{right_feature}",
        )

    def _lookup_rule(self, operator_examples: list[tuple[str, str, str]]) -> dict[str, Any] | None:
        values: dict[str, str] = {}
        for left_text, right_text, output_text in operator_examples:
            key = f"{left_text}\t{right_text}"
            if key in values and values[key] != output_text:
                return None
            values[key] = output_text
        return {"kind": "lookup", "values": values}

    def _default_rule_for_operator(self, operator: str) -> dict[str, str] | None:
        if operator == "+":
            return {"kind": "single", "feature": "concat_rl", "strip_mode": "none"}
        if operator == "-":
            return {"kind": "single", "feature": "sub", "strip_mode": "none"}
        return None

    def _candidate_params(
        self,
        examples: list[tuple[str, str]],
        *,
        query_operator: str | None = None,
    ) -> list[dict[str, Any]]:
        parsed_examples: list[tuple[str, str, str, str]] = []
        for input_text, output_text in examples:
            parsed = _parse_binary_equation(input_text)
            if parsed is None:
                return []
            left_text, operator, right_text = parsed
            parsed_examples.append((left_text, operator, right_text, output_text.strip()))
        rules_by_operator_options: dict[str, list[dict[str, Any]]] = {}
        for operator in sorted({operator for _, operator, _, _ in parsed_examples}):
            operator_examples = [(left_text, right_text, output_text) for left_text, op, right_text, output_text in parsed_examples if op == operator]
            common_features = sorted(
                set.intersection(
                    *(set(self._feature_map(left_text, right_text)) for left_text, right_text, _ in operator_examples)
                )
            )
            if not common_features:
                lookup = self._lookup_rule(operator_examples)
                if lookup is None:
                    return []
                rules_by_operator_options[operator] = [lookup]
                continue
            candidate_rules: list[dict[str, str]] = []
            for feature_name in common_features:
                for strip_mode in ["none", "strip_leading", "strip_all"]:
                    candidate_rules.append({"kind": "single", "feature": feature_name, "strip_mode": strip_mode})
            for left_feature in common_features:
                for right_feature in common_features:
                    if left_feature == right_feature:
                        continue
                    for strip_mode in ["none", "strip_leading"]:
                        candidate_rules.append(
                            {
                                "kind": "concat",
                                "left_feature": left_feature,
                                "right_feature": right_feature,
                                "strip_mode": strip_mode,
                            }
                        )
            for rule in candidate_rules:
                rendered = [self._render_rule(left_text, right_text, rule) for left_text, right_text, _ in operator_examples]
                if rendered == [output_text for _, _, output_text in operator_examples]:
                    rules_by_operator_options.setdefault(operator, []).append(rule)
            if not rules_by_operator_options.get(operator):
                lookup = self._lookup_rule(operator_examples)
                if lookup is None:
                    return []
                rules_by_operator_options[operator] = [lookup]
            else:
                rules_by_operator_options[operator].sort(key=self._rule_sort_key)
                rules_by_operator_options[operator] = rules_by_operator_options[operator][:4]
        if query_operator is not None and query_operator not in rules_by_operator_options:
            default_rule = self._default_rule_for_operator(query_operator)
            if default_rule is not None:
                rules_by_operator_options[query_operator] = [default_rule]
        if not any(
            rule.get("kind") != "lookup"
            for rules in rules_by_operator_options.values()
            for rule in rules
        ):
            return []

        operators = sorted(rules_by_operator_options)
        params: list[dict[str, Any]] = []
        for combination in itertools.product(*(rules_by_operator_options[operator] for operator in operators)):
            params.append({"rules_by_operator": dict(zip(operators, combination))})
            if len(params) >= 24:
                break
        return params

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        return self._candidate_params(examples)

    def candidate_params_for_query(self, examples: list[tuple[str, str]], query: str) -> list[dict[str, Any]]:
        parsed_query = _parse_binary_equation(query)
        query_operator = None if parsed_query is None else parsed_query[1]
        return self._candidate_params(examples, query_operator=query_operator)

    def apply(self, text: str, params: dict[str, Any]) -> str:
        parsed = _parse_binary_equation(text)
        if parsed is None:
            raise ValueError("not a binary equation pattern")
        left_text, operator, right_text = parsed
        rules_by_operator = params["rules_by_operator"]
        if operator not in rules_by_operator:
            raise ValueError("unsupported operator")
        result = self._render_rule(left_text, right_text, rules_by_operator[operator])
        if not result:
            raise ValueError("empty numeric result")
        return result

    def complexity_penalty(self, params: dict[str, Any]) -> float:
        penalty = 0.0
        for rule in params.get("rules_by_operator", {}).values():
            if rule.get("kind") == "lookup":
                penalty += 0.02 * len(rule.get("values", {}))
                continue
            if rule.get("kind") == "concat":
                penalty += 0.01
            strip_mode = rule.get("strip_mode", "none")
            if strip_mode == "strip_all":
                penalty += 0.006
            elif strip_mode == "strip_leading":
                penalty += 0.002
        return penalty

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        params = {
            "rules_by_operator": {
                "$": {"kind": "single", "feature": "mul", "strip_mode": "none"},
                ")": {"kind": "single", "feature": "abs_sub", "strip_mode": "none"},
            }
        }
        source = "25$96"
        return source, self.apply(source, params), params


class AddConstantOp(AtomicOp):
    name = "add_constant"
    family = "arithmetic_equation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        deltas: list[int] = []
        for input_text, output_text in examples:
            src = _safe_int(input_text)
            dst = _safe_int(output_text)
            if src is None or dst is None:
                continue
            deltas.append(dst - src)
        if not deltas:
            deltas = list(range(-5, 6))
        return [{"delta": delta} for delta in sorted(set(deltas))]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        value = _safe_int(text)
        if value is None:
            raise ValueError("not an integer")
        return str(value + int(params["delta"]))

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        value = rng.randint(1, 20)
        delta = rng.randint(-5, 5)
        return str(value), str(value + delta), {"delta": delta}


class MultiplyConstantOp(AtomicOp):
    name = "multiply_constant"
    family = "arithmetic_equation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        factors: list[int] = []
        for input_text, output_text in examples:
            src = _safe_int(input_text)
            dst = _safe_int(output_text)
            if src in (None, 0) or dst is None:
                continue
            if dst % src == 0:
                factors.append(dst // src)
        if not factors:
            factors = [2, -1, -2, 3]
        return [{"factor": factor} for factor in sorted(set(factors))]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        value = _safe_int(text)
        if value is None:
            raise ValueError("not an integer")
        return str(value * int(params["factor"]))

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        value = rng.randint(1, 12)
        factor = rng.choice([2, 3, -1])
        return str(value), str(value * factor), {"factor": factor}


class AffineTransformOp(AtomicOp):
    name = "affine_transform"
    family = "arithmetic_equation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        numeric_pairs = [(_safe_int(src), _safe_int(dst)) for src, dst in examples]
        numeric_pairs = [(src, dst) for src, dst in numeric_pairs if src is not None and dst is not None]
        if len(numeric_pairs) < 2:
            return []
        candidates: list[dict[str, Any]] = []
        for idx in range(len(numeric_pairs) - 1):
            src1, dst1 = numeric_pairs[idx]
            src2, dst2 = numeric_pairs[idx + 1]
            if src1 == src2:
                continue
            slope = (dst2 - dst1) / (src2 - src1)
            intercept = dst1 - slope * src1
            if abs(slope - round(slope)) < 1e-9 and abs(intercept - round(intercept)) < 1e-9:
                candidates.append({"a": int(round(slope)), "b": int(round(intercept))})
        return candidates

    def apply(self, text: str, params: dict[str, Any]) -> str:
        value = _safe_int(text)
        if value is None:
            raise ValueError("not an integer")
        return str(int(params["a"]) * value + int(params["b"]))

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        value = rng.randint(1, 10)
        a = rng.choice([2, 3])
        b = rng.randint(-3, 3)
        return str(value), str(a * value + b), {"a": a, "b": b}


class DigitSumOp(AtomicOp):
    name = "digit_sum"
    family = "count_filter_aggregation"

    def apply(self, text: str, params: dict[str, Any]) -> str:
        digits = [int(char) for char in text if char.isdigit()]
        if not digits:
            raise ValueError("no digits")
        return str(sum(digits))

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "".join(rng.choice("123456789") for _ in range(4))
        return source, str(sum(int(char) for char in source)), {}


class FilterCharactersOp(AtomicOp):
    name = "filter_characters"
    family = "count_filter_aggregation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        return [{"mode": mode} for mode in ["digits", "letters", "vowels", "consonants", "alnum"]]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        mode = params["mode"]
        if mode == "digits":
            return "".join(char for char in text if char.isdigit())
        if mode == "letters":
            return "".join(char for char in text if char.isalpha())
        if mode == "vowels":
            return "".join(char for char in text if char.lower() in "aeiou")
        if mode == "consonants":
            return "".join(char for char in text if char.isalpha() and char.lower() not in "aeiou")
        if mode == "alnum":
            return "".join(char for char in text if char.isalnum())
        raise ValueError(f"Unsupported mode: {mode}")

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "a1b2c3"
        return source, "123", {"mode": "digits"}


class CountItemsOp(AtomicOp):
    name = "count_items"
    family = "count_filter_aggregation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        return [{"mode": mode} for mode in ["chars", "tokens", "digits", "letters", "unique_chars", "vowels"]]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        mode = params["mode"]
        if mode == "chars":
            return str(len(text))
        if mode == "tokens":
            return str(len(text.split()))
        if mode == "digits":
            return str(sum(char.isdigit() for char in text))
        if mode == "letters":
            return str(sum(char.isalpha() for char in text))
        if mode == "unique_chars":
            return str(len(set(text.replace(" ", ""))))
        if mode == "vowels":
            return str(sum(char.lower() in "aeiou" for char in text))
        raise ValueError(f"Unsupported mode: {mode}")

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "red blue green"
        return source, "3", {"mode": "tokens"}


class EvaluateExpressionOp(AtomicOp):
    name = "evaluate_expression"
    family = "arithmetic_equation"

    def apply(self, text: str, params: dict[str, Any]) -> str:
        return _format_number(_safe_eval_expression(text))

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "3 + 4 * 2"
        return source, "11", {}


class UnitConvertOp(AtomicOp):
    name = "unit_convert"
    family = "unit_conversion"
    _SUPPORTED = {
        ("km", "m"): 1000.0,
        ("m", "cm"): 100.0,
        ("kg", "g"): 1000.0,
        ("h", "min"): 60.0,
    }

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        return [
            {"source_unit": source_unit, "target_unit": target_unit, "factor": factor}
            for (source_unit, target_unit), factor in self._SUPPORTED.items()
        ]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        value_text, unit = text.strip().split()
        if unit.lower() != params["source_unit"]:
            raise ValueError("unexpected unit")
        return f"{_format_number(float(value_text) * float(params['factor']))} {params['target_unit']}"

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        value = rng.randint(1, 5)
        params = {"source_unit": "km", "target_unit": "m", "factor": 1000.0}
        return f"{value} km", f"{value * 1000} m", params


class ScaleMeasurementOp(AtomicOp):
    name = "scale_measurement"
    family = "unit_conversion"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        ratios: list[float] = []
        source_unit: str | None = None
        output_decimals = [_decimal_places(output_text) for _, output_text in examples]
        decimals = max(output_decimals, default=2)
        for input_text, output_text in examples:
            try:
                source_value, unit = _parse_measurement(input_text)
            except ValueError:
                return []
            target_value = _safe_float(output_text)
            if target_value is None or abs(source_value) < 1e-12:
                return []
            if source_unit is None:
                source_unit = unit
            elif unit != source_unit:
                return []
            ratios.append(target_value / source_value)
        if not ratios:
            return []
        factor = sum(ratios) / len(ratios)
        max_deviation = max(abs(ratio - factor) for ratio in ratios)
        if max_deviation > 5e-3:
            return []
        candidates: list[dict[str, Any]] = []
        trim_preferred = len(set(output_decimals)) > 1
        for trim_trailing_zeros in ([trim_preferred, not trim_preferred] if trim_preferred else [False, True]):
            if all(
                (
                    _format_trimmed(source_value * factor, decimals)
                    if trim_trailing_zeros
                    else _format_fixed(source_value * factor, decimals)
                )
                == output_text.strip()
                for (input_text, output_text), source_value in (
                    ((input_text, output_text), _parse_measurement(input_text)[0])
                    for input_text, output_text in examples
                )
            ):
                candidates.append(
                    {
                        "factor": factor,
                        "source_unit": source_unit,
                        "decimals": decimals,
                        "trim_trailing_zeros": trim_trailing_zeros,
                    }
                )
        return candidates or [{"factor": factor, "source_unit": source_unit, "decimals": decimals, "trim_trailing_zeros": trim_preferred}]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        value, unit = _parse_measurement(text)
        expected = params.get("source_unit")
        if expected is not None and unit != expected:
            raise ValueError("unexpected unit")
        decimals = int(params.get("decimals", 2))
        if params.get("trim_trailing_zeros", False):
            return _format_trimmed(value * float(params["factor"]), decimals)
        return _format_fixed(value * float(params["factor"]), decimals)

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        factor = rng.choice([0.5, 0.75, 1.25])
        value = round(rng.uniform(5.0, 40.0), 2)
        decimals = 2
        params = {"factor": factor, "source_unit": "m", "decimals": decimals}
        return f"{value:.2f} m", _format_fixed(value * factor, decimals), params


class GravityDistanceOp(AtomicOp):
    name = "gravity_distance"
    family = "gravity"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        estimates: list[float] = []
        output_decimals = [_decimal_places(output_text) for _, output_text in examples]
        decimals = max(output_decimals, default=2)
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []
        for input_text, output_text in examples:
            time_value = _safe_float(input_text)
            distance_value = _safe_float(output_text)
            if time_value is None or distance_value is None or abs(time_value) < 1e-12:
                return []
            estimates.append((2.0 * distance_value) / (time_value * time_value))
            tolerance = 0.5 * (10 ** (-_decimal_places(output_text)))
            lower_bounds.append((2.0 * (distance_value - tolerance)) / (time_value * time_value))
            upper_bounds.append((2.0 * (distance_value + tolerance)) / (time_value * time_value))
        if not estimates:
            return []
        g_value = sum(estimates) / len(estimates)
        g_candidates: list[float] = [g_value]
        lower_bound = max(lower_bounds)
        upper_bound = min(upper_bounds)
        if lower_bound <= upper_bound:
            midpoint = (lower_bound + upper_bound) / 2.0
            g_candidates.append(midpoint)
            precision = decimals + 2
            scale = 10**precision
            start = int(lower_bound * scale)
            end = int(upper_bound * scale) + 1
            if end - start <= 64:
                for raw in range(start, end + 1):
                    candidate = raw / scale
                    if lower_bound <= candidate <= upper_bound:
                        g_candidates.append(candidate)
        trim_preferred = len(set(output_decimals)) > 1
        candidates: list[dict[str, Any]] = []
        unique_g = sorted(set(round(candidate, decimals + 4) for candidate in g_candidates))
        style_order = [trim_preferred, not trim_preferred] if trim_preferred else [False, True]
        for trim_trailing_zeros in style_order:
            valid_g: list[float] = []
            for candidate_g in unique_g:
                rendered = []
                for input_text, _ in examples:
                    time_value = _safe_float(input_text)
                    if time_value is None:
                        rendered = []
                        break
                    distance = 0.5 * candidate_g * time_value * time_value
                    rendered.append(
                        _format_trimmed(distance, decimals)
                        if trim_trailing_zeros
                        else _format_fixed(distance, decimals)
                    )
                if rendered == [output_text.strip() for _, output_text in examples]:
                    valid_g.append(candidate_g)
            if valid_g:
                center = (lower_bound + upper_bound) / 2.0 if lower_bound <= upper_bound else g_value
                valid_g.sort(key=lambda candidate_g: (abs(candidate_g - center), -candidate_g))
                candidates.extend(
                    {
                        "g": candidate_g,
                        "decimals": decimals,
                        "trim_trailing_zeros": trim_trailing_zeros,
                    }
                    for candidate_g in valid_g[:12]
                )
        if candidates:
            return candidates
        return [{"g": g_value, "decimals": decimals, "trim_trailing_zeros": trim_preferred}]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        time_value = _safe_float(text)
        if time_value is None:
            raise ValueError("not a time scalar")
        distance = 0.5 * float(params["g"]) * time_value * time_value
        decimals = int(params.get("decimals", 2))
        if params.get("trim_trailing_zeros", False):
            return _format_trimmed(distance, decimals)
        return _format_fixed(distance, decimals)

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        g_value = rng.uniform(6.0, 18.0)
        time_value = round(rng.uniform(1.0, 5.0), 2)
        params = {"g": g_value, "decimals": 2}
        return f"{time_value:.2f}", _format_fixed(0.5 * g_value * time_value * time_value, 2), params


class BinaryAffineTransformOp(AtomicOp):
    name = "binary_affine_transform"
    family = "bit_manipulation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        if not examples:
            return []
        valid_count = sum(
            1
            for src, dst in examples
            if _is_binary_string(src) and _is_binary_string(dst) and len(src.strip()) == len(dst.strip())
        )
        widths = {
            len(src.strip())
            for src, dst in examples
            if _is_binary_string(src) and _is_binary_string(dst) and len(src.strip()) == len(dst.strip())
        }
        if len(widths) != 1 or valid_count != len(examples):
            return []
        width = widths.pop()
        design_rows = [_binary_vector(src) + [1] for src, _ in examples]
        matrix: list[list[int]] = []
        bias: list[int] = []
        for column in range(width):
            targets = [1 if dst.strip()[column] == "1" else 0 for _, dst in examples]
            solution = _solve_gf2_system(design_rows, targets)
            if solution is None:
                return []
            matrix.append(solution[:-1])
            bias.append(solution[-1])
        return [{"matrix": matrix, "bias": bias, "width": width}]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        bits = _binary_vector(text)
        width = int(params.get("width", len(bits)))
        if len(bits) != width:
            raise ValueError("unexpected binary width")
        matrix = params["matrix"]
        bias = params.get("bias", [0] * width)
        output_bits: list[str] = []
        for row, row_bias in zip(matrix, bias):
            value = int(row_bias) & 1
            for coeff, bit in zip(row, bits):
                if coeff:
                    value ^= bit
            output_bits.append("1" if value else "0")
        return "".join(output_bits)

    def complexity_penalty(self, params: dict[str, Any]) -> float:
        matrix = params.get("matrix", [])
        width = max(1, int(params.get("width", len(matrix) or 1)))
        active = sum(sum(int(coeff) for coeff in row) for row in matrix)
        density = active / float(width * width)
        return 0.02 + 0.05 * density

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        params = {
            "matrix": [
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            "bias": [0, 0, 1, 0],
            "width": 4,
        }
        source = "1010"
        return source, self.apply(source, params), params


class BinaryPermutationOp(AtomicOp):
    name = "binary_permutation"
    family = "bit_manipulation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        if not examples:
            return []
        widths = {
            len(src.strip())
            for src, dst in examples
            if _is_binary_string(src) and _is_binary_string(dst) and len(src.strip()) == len(dst.strip())
        }
        if len(widths) != 1 or len(examples) != sum(
            1
            for src, dst in examples
            if _is_binary_string(src) and _is_binary_string(dst) and len(src.strip()) == len(dst.strip())
        ):
            return []
        width = widths.pop()
        permutation: list[int] = []
        used_indices: set[int] = set()
        for output_index in range(width):
            valid_inputs = []
            for input_index in range(width):
                if input_index in used_indices:
                    continue
                if all(src.strip()[input_index] == dst.strip()[output_index] for src, dst in examples):
                    valid_inputs.append(input_index)
            if len(valid_inputs) != 1:
                return []
            permutation.append(valid_inputs[0])
            used_indices.add(valid_inputs[0])
        return [{"permutation": permutation, "width": width}]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        stripped = text.strip()
        if not _is_binary_string(stripped):
            raise ValueError("not a binary string")
        permutation = [int(index) for index in params["permutation"]]
        if len(stripped) != int(params["width"]):
            raise ValueError("unexpected binary width")
        return "".join(stripped[index] for index in permutation)

    def complexity_penalty(self, params: dict[str, Any]) -> float:
        return 0.004

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        params = {"permutation": [2, 0, 3, 1], "width": 4}
        source = "1010"
        return source, self.apply(source, params), params


class BinaryNibbleMapOp(AtomicOp):
    name = "binary_nibble_map"
    family = "bit_manipulation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        if not examples:
            return []
        widths = {
            len(src.strip())
            for src, dst in examples
            if _is_binary_string(src) and _is_binary_string(dst) and len(src.strip()) == len(dst.strip()) and len(src.strip()) % 4 == 0
        }
        if len(widths) != 1 or len(examples) != sum(
            1
            for src, dst in examples
            if _is_binary_string(src) and _is_binary_string(dst) and len(src.strip()) == len(dst.strip()) and len(src.strip()) % 4 == 0
        ):
            return []
        width = widths.pop()
        num_nibbles = width // 4
        nibble_maps: list[dict[str, str]] = [dict() for _ in range(num_nibbles)]
        for src, dst in examples:
            stripped_src = src.strip()
            stripped_dst = dst.strip()
            for index in range(num_nibbles):
                src_nibble = stripped_src[index * 4 : (index + 1) * 4]
                dst_nibble = stripped_dst[index * 4 : (index + 1) * 4]
                existing = nibble_maps[index].get(src_nibble)
                if existing is not None and existing != dst_nibble:
                    return []
                nibble_maps[index][src_nibble] = dst_nibble
        return [{"nibble_maps": nibble_maps, "width": width}]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        stripped = text.strip()
        if not _is_binary_string(stripped) or len(stripped) % 4 != 0:
            raise ValueError("not a nibble-aligned binary string")
        width = int(params["width"])
        if len(stripped) != width:
            raise ValueError("unexpected binary width")
        outputs: list[str] = []
        nibble_maps = params["nibble_maps"]
        for index, nibble_map in enumerate(nibble_maps):
            src_nibble = stripped[index * 4 : (index + 1) * 4]
            if src_nibble not in nibble_map:
                raise ValueError("unseen nibble")
            outputs.append(nibble_map[src_nibble])
        return "".join(outputs)

    def complexity_penalty(self, params: dict[str, Any]) -> float:
        nibble_maps = params.get("nibble_maps", [])
        size = sum(len(nibble_map) for nibble_map in nibble_maps)
        return 0.01 + 0.003 * size

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        params = {
            "nibble_maps": [
                {"1010": "0101", "1100": "0011"},
                {"0011": "1100", "1111": "0000"},
            ],
            "width": 8,
        }
        source = "10100011"
        return source, self.apply(source, params), params


class BinaryInvertOp(AtomicOp):
    name = "binary_invert"
    family = "bit_manipulation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        if all(_is_binary_string(src) and _is_binary_string(dst) for src, dst in examples):
            return [{}]
        return []

    def apply(self, text: str, params: dict[str, Any]) -> str:
        if not _is_binary_string(text):
            raise ValueError("not a binary string")
        return "".join("1" if bit == "0" else "0" for bit in text.strip())

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "0101"
        return source, "1010", {}


class ReverseBitsOp(AtomicOp):
    name = "reverse_bits"
    family = "bit_manipulation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        if all(_is_binary_string(src) and _is_binary_string(dst) for src, dst in examples):
            return [{}]
        return []

    def apply(self, text: str, params: dict[str, Any]) -> str:
        if not _is_binary_string(text):
            raise ValueError("not a binary string")
        return text.strip()[::-1]

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "10110000"
        return source, source[::-1], {}


class SwapNibblesOp(AtomicOp):
    name = "swap_nibbles"
    family = "bit_manipulation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        if all(_is_binary_string(src) and len(src.strip()) % 2 == 0 for src, _ in examples):
            return [{}]
        return []

    def apply(self, text: str, params: dict[str, Any]) -> str:
        stripped = text.strip()
        if not _is_binary_string(stripped) or len(stripped) % 2 != 0:
            raise ValueError("not an even-length binary string")
        half = len(stripped) // 2
        return stripped[half:] + stripped[:half]

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "10100011"
        return source, "00111010", {}


class BinaryRotateLeftOp(AtomicOp):
    name = "binary_rotate_left"
    family = "bit_manipulation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        lengths = [len(src.strip()) for src, dst in examples if _is_binary_string(src) and _is_binary_string(dst)]
        if len(lengths) != len(examples) or not lengths:
            return []
        max_len = min(max(lengths), 8)
        return [{"k": k} for k in range(1, max_len)]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        stripped = text.strip()
        if not _is_binary_string(stripped):
            raise ValueError("not a binary string")
        return _rotate_left(stripped, int(params["k"]))

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "11010010"
        return source, _rotate_left(source, 2), {"k": 2}


class BinaryRotateRightOp(AtomicOp):
    name = "binary_rotate_right"
    family = "bit_manipulation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        lengths = [len(src.strip()) for src, dst in examples if _is_binary_string(src) and _is_binary_string(dst)]
        if len(lengths) != len(examples) or not lengths:
            return []
        max_len = min(max(lengths), 8)
        return [{"k": k} for k in range(1, max_len)]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        stripped = text.strip()
        if not _is_binary_string(stripped):
            raise ValueError("not a binary string")
        return _rotate_right(stripped, int(params["k"]))

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "11010010"
        return source, _rotate_right(source, 2), {"k": 2}


class BinaryXorMaskOp(AtomicOp):
    name = "binary_xor_mask"
    family = "bit_manipulation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        masks: list[int] = []
        width = None
        for src, dst in examples:
            if not (_is_binary_string(src) and _is_binary_string(dst)) or len(src.strip()) != len(dst.strip()):
                return []
            width = len(src.strip())
            masks.append(int(src, 2) ^ int(dst, 2))
        if not masks or len(set(masks)) != 1:
            return []
        return [{"mask": masks[0], "width": width}]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        stripped = text.strip()
        if not _is_binary_string(stripped):
            raise ValueError("not a binary string")
        width = int(params.get("width", len(stripped)))
        return format(int(stripped, 2) ^ int(params["mask"]), f"0{width}b")

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "10100101"
        params = {"mask": int("11110000", 2), "width": 8}
        return source, self.apply(source, params), params


class BinaryAndMaskOp(AtomicOp):
    name = "binary_and_mask"
    family = "bit_manipulation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        masks: list[int] = []
        width = None
        for src, dst in examples:
            if not (_is_binary_string(src) and _is_binary_string(dst)) or len(src.strip()) != len(dst.strip()):
                return []
            width = len(src.strip())
            src_val = int(src, 2)
            dst_val = int(dst, 2)
            if dst_val & ~src_val:
                return []
            masks.append(src_val ^ (src_val & dst_val))
            inferred_mask = 0
            for bit_index in range(width):
                src_bit = (src_val >> bit_index) & 1
                dst_bit = (dst_val >> bit_index) & 1
                if src_bit == 1 and dst_bit == 1:
                    inferred_mask |= (1 << bit_index)
            masks.append(inferred_mask)
        if not masks:
            return []
        candidate_mask = masks[-1]
        return [{"mask": candidate_mask, "width": width}]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        stripped = text.strip()
        if not _is_binary_string(stripped):
            raise ValueError("not a binary string")
        width = int(params.get("width", len(stripped)))
        return format(int(stripped, 2) & int(params["mask"]), f"0{width}b")

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "10100101"
        params = {"mask": int("11110000", 2), "width": 8}
        return source, self.apply(source, params), params


class BinaryOrMaskOp(AtomicOp):
    name = "binary_or_mask"
    family = "bit_manipulation"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        masks: list[int] = []
        width = None
        for src, dst in examples:
            if not (_is_binary_string(src) and _is_binary_string(dst)) or len(src.strip()) != len(dst.strip()):
                return []
            width = len(src.strip())
            src_val = int(src, 2)
            dst_val = int(dst, 2)
            if src_val & ~dst_val:
                return []
            masks.append(dst_val ^ src_val)
        if not masks or len(set(masks)) != 1:
            return []
        return [{"mask": masks[0], "width": width}]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        stripped = text.strip()
        if not _is_binary_string(stripped):
            raise ValueError("not a binary string")
        width = int(params.get("width", len(stripped)))
        return format(int(stripped, 2) | int(params["mask"]), f"0{width}b")

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        source = "10100101"
        params = {"mask": int("00001111", 2), "width": 8}
        return source, self.apply(source, params), params


class BitwiseXorConstantOp(AtomicOp):
    name = "bitwise_xor_constant"
    family = "bit_operations"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        constants: list[int] = []
        for input_text, output_text in examples:
            src = _safe_int(input_text)
            dst = _safe_int(output_text)
            if src is None or dst is None:
                continue
            constants.append(src ^ dst)
        if not constants:
            constants = [1, 2, 3]
        return [{"value": value} for value in sorted(set(constants))]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        value = _safe_int(text)
        if value is None:
            raise ValueError("not an integer")
        return str(value ^ int(params["value"]))

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        value = rng.randint(0, 15)
        const = rng.randint(1, 7)
        return str(value), str(value ^ const), {"value": const}


class BitwiseAndConstantOp(AtomicOp):
    name = "bitwise_and_constant"
    family = "bit_operations"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        return [{"value": value} for value in [1, 3, 7, 15]]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        value = _safe_int(text)
        if value is None:
            raise ValueError("not an integer")
        return str(value & int(params["value"]))

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        value = rng.randint(0, 15)
        const = 7
        return str(value), str(value & const), {"value": const}


class BitwiseOrConstantOp(AtomicOp):
    name = "bitwise_or_constant"
    family = "bit_operations"

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        return [{"value": value} for value in [1, 3, 7, 15]]

    def apply(self, text: str, params: dict[str, Any]) -> str:
        value = _safe_int(text)
        if value is None:
            raise ValueError("not an integer")
        return str(value | int(params["value"]))

    def generate_random_instance(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        value = rng.randint(0, 15)
        const = 3
        return str(value), str(value | const), {"value": const}
