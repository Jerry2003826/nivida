from __future__ import annotations

from src.teacher.chain_search import ChainSearchEngine
from src.competition.schema import PuzzleExample, PuzzleMetadata, PuzzlePair


def test_chain_search_solves_atomic_reverse() -> None:
    engine = ChainSearchEngine(max_depth=1)
    candidates = engine.search([("abc", "cba"), ("lamp", "pmal")], query_input="stun", top_k=3)
    assert candidates
    assert candidates[0].query_prediction == "nuts"
    assert candidates[0].steps[0].op_name == "reverse_string"


def test_chain_search_finds_two_step_composition() -> None:
    engine = ChainSearchEngine(max_depth=2, beam_width=8)
    candidates = engine.search([("abac", "dbcb"), ("xyxz", "ayzy")], query_input="mnmo", top_k=5)
    assert candidates
    assert candidates[0].query_prediction == "pnon"
    assert [step.op_name for step in candidates[0].steps] == ["reverse_string", "caesar_shift"]


def test_chain_search_solves_gravity_example() -> None:
    engine = ChainSearchEngine(max_depth=1, beam_width=6)
    example = PuzzleExample(
        id="gravity",
        raw_prompt="",
        train_pairs=[
            PuzzlePair(input="1.37", output="14.92"),
            PuzzlePair(input="4.27", output="144.96"),
            PuzzlePair(input="3.28", output="85.54"),
        ],
        query_input="4.41",
        metadata=PuzzleMetadata(family_tags=["gravity"]),
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
        train_pairs=[
            PuzzlePair(input="11", output="XI"),
            PuzzlePair(input="15", output="XV"),
            PuzzlePair(input="94", output="XCIV"),
            PuzzlePair(input="19", output="XIX"),
        ],
        query_input="38",
        metadata=PuzzleMetadata(family_tags=["numeral"]),
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
        train_pairs=[
            PuzzlePair(input="ucoov pwgtfyoqg vorq yrjjoe", output="queen discovers near valley"),
            PuzzlePair(input="pqrsfv pqorzg wvgwpo trgbjo", output="dragon dreams inside castle"),
            PuzzlePair(input="gbcpovb tqorbog bxo zrswtrj pffq", output="student creates the magical door"),
            PuzzlePair(input="bxo sfjpov pqrsfv dfjjfig", output="the golden dragon follows"),
        ],
        query_input="trb wzrswvog hffk",
        metadata=PuzzleMetadata(family_tags=["cipher"]),
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
        train_pairs=[
            PuzzlePair(input="a*b*c", output="abc"),
            PuzzlePair(input="1-2-3", output="123"),
            PuzzlePair(input="%|\"|", output="%|\"|"),
        ],
        query_input="x*y*z",
        metadata=PuzzleMetadata(family_tags=["equation"]),
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
        train_pairs=[
            PuzzlePair(input="%|*\"|", output="%|\"|"),
            PuzzlePair(input="\\(*[^", output="\\([^"),
            PuzzlePair(input="(%+[@", output="(%[@"),
            PuzzlePair(input="|[*([", output="|[(["),
        ],
        query_input="\\(*[#",
        metadata=PuzzleMetadata(family_tags=["equation"]),
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
        train_pairs=[
            PuzzlePair(input="%|*\"|", output="%|\"|"),
            PuzzlePair(input="\\(*[^", output="\\([^"),
            PuzzlePair(input="(%+[@", output="(%[@"),
            PuzzlePair(input="|[*([", output="|[(["), 
            PuzzlePair(input="[^-[(", output="-^"),
        ],
        query_input="\\(*[#",
        metadata=PuzzleMetadata(family_tags=["equation"]),
    )
    candidates = engine.solve_example(example, top_k=3)
    assert candidates
    assert candidates[0].query_prediction == "\\([#"
    assert candidates[0].steps[0].op_name == "operator_template"


def test_chain_search_solves_binary_equation_rule_example() -> None:
    engine = ChainSearchEngine(max_depth=1, beam_width=8)
    example = PuzzleExample(
        id="equation_numeric",
        raw_prompt="",
        train_pairs=[
            PuzzlePair(input="96$54", output="5184"),
            PuzzlePair(input="50$41", output="2050"),
            PuzzlePair(input="51$95", output="4845"),
            PuzzlePair(input="89$47", output="4183"),
        ],
        query_input="59$49",
        metadata=PuzzleMetadata(family_tags=["equation"]),
    )
    candidates = engine.solve_example(example, top_k=3)
    assert candidates
    assert candidates[0].query_prediction == "2891"
    assert candidates[0].steps[0].op_name == "binary_equation_rule"


def test_chain_search_solves_binary_xor_mask_example() -> None:
    engine = ChainSearchEngine(max_depth=1, beam_width=8)
    example = PuzzleExample(
        id="bit_xor",
        raw_prompt="",
        train_pairs=[
            PuzzlePair(input="10100101", output="01010101"),
            PuzzlePair(input="00001111", output="11111111"),
            PuzzlePair(input="11000011", output="00110011"),
            PuzzlePair(input="11110000", output="00000000"),
            PuzzlePair(input="01010101", output="10100101"),
            PuzzlePair(input="00111100", output="11001100"),
        ],
        query_input="10011001",
        metadata=PuzzleMetadata(family_tags=["bit_manipulation"]),
    )
    candidates = engine.solve_example(example, top_k=3)
    assert candidates
    assert candidates[0].query_prediction == "01101001"
    assert candidates[0].steps[0].op_name in {"binary_xor_mask", "binary_affine_transform"}


def test_chain_search_rejects_non_binary_bit_states() -> None:
    engine = ChainSearchEngine(max_depth=2, beam_width=8)
    example = PuzzleExample(
        id="bit_guard",
        raw_prompt="",
        train_pairs=[
            PuzzlePair(input="11111111", output="11101111"),
            PuzzlePair(input="00001111", output="00011111"),
        ],
        query_input="00110011",
        metadata=PuzzleMetadata(family_tags=["bit_manipulation"]),
    )
    candidates = engine.solve_example(example, top_k=5)
    assert candidates
    assert all(set(candidate.query_prediction or "") <= {"0", "1"} for candidate in candidates if candidate.query_prediction)
