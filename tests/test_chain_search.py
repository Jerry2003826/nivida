from __future__ import annotations

from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair
from src.teacher.atomic_ops import AtomicOp
from src.teacher.chain_search import ChainSearchEngine


class _ScoreTieOp(AtomicOp):
    family = "equation"

    def __init__(self, name: str, query_prediction: str, penalty: float) -> None:
        self.name = name
        self._query_prediction = query_prediction
        self._penalty = penalty

    def candidate_params(self, examples: list[tuple[str, str]]) -> list[dict[str, object]]:
        return [{}]

    def apply(self, text: str, params: dict[str, object]) -> str:
        return self._query_prediction if text == "query" else "target"

    def complexity_penalty(self, params: dict[str, object]) -> float:
        return self._penalty


class _GraphPriorStub:
    def start_prior(self, op_name: str) -> float:
        return 10.0 if op_name == "high_prior_low_score" else 0.0

    def transition_prior(self, left: str, right: str) -> float:
        return 0.0


def test_chain_search_prefers_total_score_over_graph_prior_tie() -> None:
    engine = ChainSearchEngine(
        ops=[
            _ScoreTieOp("high_prior_low_score", "bad", penalty=1.0),
            _ScoreTieOp("low_prior_high_score", "good", penalty=0.0),
        ],
        max_depth=1,
        beam_width=2,
        graph_prior=_GraphPriorStub(),
    )

    candidates = engine.search(
        [("left", "target"), ("right", "target")],
        query="query",
        family_hints=["equation_template"],
        top_k=2,
    )

    assert [candidate.query_prediction for candidate in candidates] == ["good", "bad"]
    assert candidates[0].score > candidates[1].score


def test_chain_search_solves_atomic_reverse() -> None:
    engine = ChainSearchEngine(max_depth=1)
    candidates = engine.search([("abc", "cba"), ("lamp", "pmal")], query="stun", top_k=3)
    assert candidates
    assert candidates[0].query_prediction == "nuts"
    assert candidates[0].steps[0].op_name == "reverse_string"


def test_chain_search_finds_two_step_composition() -> None:
    engine = ChainSearchEngine(max_depth=2, beam_width=8)
    candidates = engine.search([("abac", "dbcb"), ("xyxz", "ayzy")], query="mnmo", top_k=5)
    assert candidates
    assert candidates[0].query_prediction == "pnon"
    assert [step.op_name for step in candidates[0].steps] == ["reverse_string", "caesar_shift"]


def test_chain_search_solves_gravity_example() -> None:
    engine = ChainSearchEngine(max_depth=1, beam_width=6)
    example = PuzzleExample(
        id="gravity",
        raw_prompt="",
        official_instruction="",
        parsed_examples=[
            PuzzlePair(input="1.37", output="14.92"),
            PuzzlePair(input="4.27", output="144.96"),
            PuzzlePair(input="3.28", output="85.54"),
        ],
        query="4.41",
        metadata=PuzzleMetadata(official_family="gravity", subtype="fit_constant"),
    )
    candidates = engine.solve_example(example, top_k=3)
    assert candidates
    assert candidates[0].query_prediction == "154.62"
    assert candidates[0].steps[0].op_name == "gravity_distance"


def test_chain_search_solves_roman_numeral_example() -> None:
    engine = ChainSearchEngine(max_depth=1, beam_width=6)
    example = PuzzleExample(
        id="numeral",
        raw_prompt="",
        official_instruction="",
        parsed_examples=[
            PuzzlePair(input="11", output="XI"),
            PuzzlePair(input="15", output="XV"),
            PuzzlePair(input="94", output="XCIV"),
            PuzzlePair(input="19", output="XIX"),
        ],
        query="38",
        metadata=PuzzleMetadata(official_family="numeral", subtype="roman"),
    )
    candidates = engine.solve_example(example, top_k=3)
    assert candidates
    assert candidates[0].query_prediction == "XXXVIII"
    assert candidates[0].steps[0].op_name == "decimal_to_roman"


def test_chain_search_solves_cipher_example() -> None:
    engine = ChainSearchEngine(max_depth=1, beam_width=8)
    example = PuzzleExample(
        id="cipher",
        raw_prompt="",
        official_instruction="",
        parsed_examples=[
            PuzzlePair(input="ucoov pwgtfyoqg vorq yrjjoe", output="queen discovers near valley"),
            PuzzlePair(input="pqrsfv pqorzg wvgwpo trgbjo", output="dragon dreams inside castle"),
            PuzzlePair(input="gbcpovb tqorbog bxo zrswtrj pffq", output="student creates the magical door"),
            PuzzlePair(input="bxo sfjpov pqrsfv dfjjfig", output="the golden dragon follows"),
        ],
        query="trb wzrswvog hffk",
        metadata=PuzzleMetadata(official_family="cipher", subtype="token_substitution"),
    )
    candidates = engine.solve_example(example, top_k=3)
    assert candidates
    assert candidates[0].query_prediction == "cat imagines book"
    assert candidates[0].steps[0].op_name == "vocabulary_cipher"


def test_chain_search_solves_delete_character_equation() -> None:
    engine = ChainSearchEngine(max_depth=1, beam_width=8)
    example = PuzzleExample(
        id="equation_delete",
        raw_prompt="",
        official_instruction="",
        parsed_examples=[
            PuzzlePair(input="a*b*c", output="abc"),
            PuzzlePair(input="1-2-3", output="123"),
            PuzzlePair(input="%|\"|", output="%|\"|"),
        ],
        query="x*y*z",
        metadata=PuzzleMetadata(official_family="equation", subtype="symbolic"),
    )
    candidates = engine.solve_example(example, top_k=3)
    assert candidates
    assert candidates[0].query_prediction == "xyz"
    assert candidates[0].steps[0].op_name == "delete_characters"


def test_chain_search_solves_position_transducer_equation() -> None:
    engine = ChainSearchEngine(max_depth=1, beam_width=8)
    example = PuzzleExample(
        id="equation_position_transducer",
        raw_prompt="",
        official_instruction="",
        parsed_examples=[
            PuzzlePair(input="%|*\"|", output="%|\"|"),
            PuzzlePair(input="\\(*[^", output="\\([^"),
            PuzzlePair(input="(%+[@", output="(%[@"),
            PuzzlePair(input="|[*([", output="|[(["), 
        ],
        query="\\(*[#",
        metadata=PuzzleMetadata(official_family="equation", subtype="symbolic"),
    )
    candidates = engine.solve_example(example, top_k=3)
    assert candidates
    assert candidates[0].query_prediction == "\\([#"
    assert candidates[0].steps[0].op_name in {"delete_characters", "position_transducer", "operator_template"}


def test_chain_search_solves_operator_template_equation() -> None:
    engine = ChainSearchEngine(max_depth=1, beam_width=8)
    example = PuzzleExample(
        id="equation_operator_template",
        raw_prompt="",
        official_instruction="",
        parsed_examples=[
            PuzzlePair(input="%|*\"|", output="%|\"|"),
            PuzzlePair(input="\\(*[^", output="\\([^"),
            PuzzlePair(input="(%+[@", output="(%[@"),
            PuzzlePair(input="|[*([", output="|[(["), 
            PuzzlePair(input="[^-[(", output="-^"),
        ],
        query="\\(*[#",
        metadata=PuzzleMetadata(official_family="equation", subtype="symbolic"),
    )
    candidates = engine.solve_example(example, top_k=3)
    assert candidates
    assert candidates[0].query_prediction == "\\([#"
    assert candidates[0].steps[0].op_name == "operator_template"


def test_chain_search_prefers_query_derived_symbolic_template_tie() -> None:
    engine = ChainSearchEngine(max_depth=1, beam_width=24)
    example = PuzzleExample(
        id="equation_template_query_prior",
        raw_prompt="",
        official_instruction="",
        parsed_examples=[
            PuzzlePair(input="\"#+@[", output="\"#@["),
            PuzzlePair(input="[>*|(", output="@:<"),
            PuzzlePair(input="<(-[:", output="|:"),
            PuzzlePair(input="|#-<\"", output="(<"),
            PuzzlePair(input="@@*(>", output="(\"|:"),
        ],
        query="(%+|[",
        metadata=PuzzleMetadata(official_family="equation", subtype="equation_template"),
    )

    candidates = engine.solve_example(example, top_k=3)

    assert candidates
    assert candidates[0].query_prediction == "(%|["
    assert candidates[0].steps[0].op_name == "operator_template"


def test_chain_search_prefers_symbolic_template_modal_output_length_tie() -> None:
    engine = ChainSearchEngine(max_depth=1, beam_width=24)
    example = PuzzleExample(
        id="equation_template_length_prior",
        raw_prompt="",
        official_instruction="",
        parsed_examples=[
            PuzzlePair(input=":'*?}", output=":'?}"),
            PuzzlePair(input="})+::", output="){:"),
            PuzzlePair(input="##-\\{", output="-'#"),
        ],
        query="?'-\\\\",
        metadata=PuzzleMetadata(official_family="equation", subtype="equation_template"),
    )

    candidates = engine.solve_example(example, top_k=3)

    assert candidates
    assert candidates[0].query_prediction == "-'?"
    assert candidates[0].steps[0].op_name == "operator_template"


def test_chain_search_prefers_unique_query_symbol_template_tie() -> None:
    engine = ChainSearchEngine(max_depth=1, beam_width=24)
    example = PuzzleExample(
        id="equation_template_unique_symbol_prior",
        raw_prompt="",
        official_instruction="",
        parsed_examples=[
            PuzzlePair(input="$^+|)", output="?\'|"),
            PuzzlePair(input="&'*)/", output="&??|"),
            PuzzlePair(input="&^-&^", output="&"),
        ],
        query="$'-^$",
        metadata=PuzzleMetadata(official_family="equation", subtype="equation_template"),
    )

    candidates = engine.solve_example(example, top_k=4)

    assert candidates
    assert candidates[0].query_prediction == "^"
    assert candidates[0].steps[0].op_name == "operator_template"


def test_chain_search_prefers_vocab_completion_for_char_substitution() -> None:
    engine = ChainSearchEngine(max_depth=1, beam_width=8)
    example = PuzzleExample(
        id="cipher_vocab_completion",
        raw_prompt="",
        official_instruction="",
        parsed_examples=[
            PuzzlePair(input="lzddf adda edm", output="queen sees key"),
            PuzzlePair(input="uqd iztshza udwiqdt sgwvsfda", output="the curious teacher imagines"),
            PuzzlePair(input="pstc dokbhtda xhtdau", output="bird explores forest"),
            PuzzlePair(input="udwiqdt dokbhtda fdwt nwbbdm", output="teacher explores near valley"),
        ],
        query="uqd rsad ctwvhf rwuiqda",
        metadata=PuzzleMetadata(official_family="cipher", subtype="cipher_char_sub"),
    )
    candidates = engine.solve_example(example, top_k=3)
    assert candidates
    assert candidates[0].query_prediction == "the wise dragon watches"
    assert candidates[0].steps[0].op_name == "vocabulary_cipher"


def test_chain_search_solves_binary_equation_rule_example() -> None:
    engine = ChainSearchEngine(max_depth=1, beam_width=8)
    example = PuzzleExample(
        id="equation_numeric",
        raw_prompt="",
        official_instruction="",
        parsed_examples=[
            PuzzlePair(input="96$54", output="5184"),
            PuzzlePair(input="50$41", output="2050"),
            PuzzlePair(input="51$95", output="4845"),
            PuzzlePair(input="89$47", output="4183"),
        ],
        query="59$49",
        metadata=PuzzleMetadata(official_family="equation", subtype="numeric"),
    )
    candidates = engine.solve_example(example, top_k=3)
    assert candidates
    assert candidates[0].query_prediction == "2891"
    assert candidates[0].steps[0].op_name == "binary_equation_rule"


def test_chain_search_uses_query_operator_prior_for_numeric_equation() -> None:
    engine = ChainSearchEngine(max_depth=1, beam_width=8)
    example = PuzzleExample(
        id="equation_numeric_query_prior",
        raw_prompt="",
        official_instruction="",
        parsed_examples=[
            PuzzlePair(input="73*57", output="6772"),
            PuzzlePair(input="29*49", output="9468"),
            PuzzlePair(input="56+16", output="1656"),
        ],
        query="22-84",
        metadata=PuzzleMetadata(official_family="equation", subtype="numeric"),
    )
    candidates = engine.solve_example(example, top_k=3)

    assert candidates
    assert candidates[0].query_prediction == "-62"
    assert candidates[0].steps[0].op_name == "binary_equation_rule"


def test_chain_search_solves_binary_xor_mask_example() -> None:
    engine = ChainSearchEngine(max_depth=1, beam_width=8)
    example = PuzzleExample(
        id="bit_xor",
        raw_prompt="",
        official_instruction="",
        parsed_examples=[
            PuzzlePair(input="10100101", output="01010101"),
            PuzzlePair(input="00001111", output="11111111"),
            PuzzlePair(input="11000011", output="00110011"),
            PuzzlePair(input="11110000", output="00000000"),
            PuzzlePair(input="01010101", output="10100101"),
            PuzzlePair(input="00111100", output="11001100"),
        ],
        query="10011001",
        metadata=PuzzleMetadata(official_family="bit", subtype="mask_logic"),
    )
    candidates = engine.solve_example(example, top_k=3)
    assert candidates
    assert candidates[0].query_prediction == "01101001"
    assert candidates[0].steps[0].op_name in {"binary_xor_mask", "binary_affine_transform"}


def test_chain_search_penalizes_complex_boolean_bit_overfit() -> None:
    engine = ChainSearchEngine(max_depth=4, beam_width=12)
    pairs = [
        ("11010011", "01011011"),
        ("00001101", "01001101"),
        ("11110011", "11001011"),
        ("01011001", "00110111"),
        ("10001100", "10001001"),
        ("00011001", "00010111"),
        ("11101001", "10101111"),
        ("11101111", "10110101"),
    ]

    candidates = engine.search(
        pairs,
        query="10100101",
        family_hints=["bit"],
        subtype="bit_permutation",
        top_k=3,
    )

    assert candidates
    assert candidates[0].query_prediction == "10111001"
    assert [step.op_name for step in candidates[0].steps] == ["binary_affine_transform"]


def test_chain_search_rejects_non_binary_bit_states() -> None:
    engine = ChainSearchEngine(max_depth=2, beam_width=8)
    example = PuzzleExample(
        id="bit_guard",
        raw_prompt="",
        official_instruction="",
        parsed_examples=[
            PuzzlePair(input="11111111", output="11101111"),
            PuzzlePair(input="00001111", output="00011111"),
        ],
        query="00110011",
        metadata=PuzzleMetadata(official_family="bit", subtype="mask_logic"),
    )
    candidates = engine.solve_example(example, top_k=5)
    assert candidates
    assert all(set(candidate.query_prediction or "") <= {"0", "1"} for candidate in candidates if candidate.query_prediction)
