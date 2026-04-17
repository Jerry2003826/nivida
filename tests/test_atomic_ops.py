from __future__ import annotations

from src.teacher.atomic_ops import (
    AddConstantOp,
    BinaryAffineTransformOp,
    BinaryEquationRuleOp,
    BinaryNibbleMapOp,
    BinaryPermutationOp,
    BinaryRotateLeftOp,
    BinaryXorMaskOp,
    CaesarShiftOp,
    CountItemsOp,
    DecimalToBinaryOp,
    DecimalToRomanOp,
    DeleteCharactersOp,
    FilterCharactersOp,
    GravityDistanceOp,
    OperatorTemplateOp,
    PositionTransducerOp,
    ReverseStringOp,
    RotateLeftOp,
    ScaleMeasurementOp,
    VocabularyCipherOp,
)


def test_reverse_string_op() -> None:
    op = ReverseStringOp()
    assert op.apply("abcd", {}) == "dcba"


def test_rotate_left_op() -> None:
    op = RotateLeftOp()
    assert op.apply("abcd", {"k": 1}) == "bcda"


def test_caesar_shift_op() -> None:
    op = CaesarShiftOp()
    assert op.apply("abc", {"shift": 1}) == "bcd"


def test_decimal_to_binary_op() -> None:
    op = DecimalToBinaryOp()
    assert op.apply("10", {}) == "1010"


def test_add_constant_op() -> None:
    op = AddConstantOp()
    fit = op.fit([("2", "5"), ("10", "13")])
    assert fit.success
    assert fit.params["delta"] == 3


def test_filter_and_count_ops() -> None:
    filter_op = FilterCharactersOp()
    count_op = CountItemsOp()
    assert filter_op.apply("a1b2c3", {"mode": "digits"}) == "123"
    assert count_op.apply("red blue green", {"mode": "tokens"}) == "3"


def test_decimal_to_roman_op() -> None:
    op = DecimalToRomanOp()
    assert op.apply("38", {}) == "XXXVIII"


def test_scale_measurement_op_fit() -> None:
    op = ScaleMeasurementOp()
    fit = op.fit([("10.08 m", "6.69"), ("17.83 m", "11.83"), ("35.85 m", "23.79")])
    assert fit.success
    assert op.apply("25.09 m", fit.params) == "16.65"


def test_gravity_distance_op_fit() -> None:
    op = GravityDistanceOp()
    fit = op.fit([("1.37", "14.92"), ("4.27", "144.96"), ("3.28", "85.54")])
    assert fit.success
    assert op.apply("4.41", fit.params) == "154.62"


def test_gravity_distance_op_mixed_decimal_style() -> None:
    op = GravityDistanceOp()
    fit = op.fit([("2.28", "21.1"), ("1.22", "6.04"), ("1.52", "9.38"), ("4.51", "82.54")])
    assert fit.success
    assert op.apply("3.33", fit.params) == "45.0"


def test_vocabulary_cipher_op_official_style_example() -> None:
    op = VocabularyCipherOp()
    fit = op.fit(
        [
            ("ucoov pwgtfyoqg vorq yrjjoe", "queen discovers near valley"),
            ("pqrsfv pqorzg wvgwpo trgbjo", "dragon dreams inside castle"),
            ("gbcpovb tqorbog bxo zrswtrj pffq", "student creates the magical door"),
            ("bxo sfjpov pqrsfv dfjjfig", "the golden dragon follows"),
        ]
    )
    assert fit.success
    assert op.apply("trb wzrswvog hffk", fit.params) == "cat imagines book"


def test_delete_characters_op_fit() -> None:
    op = DeleteCharactersOp()
    fit = op.fit([("a*b*c", "abc"), ("1-2-3", "123")])
    assert fit.success
    assert op.apply("x*y*z", fit.params) == "xyz"


def test_position_transducer_op_fit() -> None:
    op = PositionTransducerOp()
    fit = op.fit(
        [
            ("%|*\"|", "%|\"|"),
            ("\\(*[^", "\\([^"),
            ("(%+[@", "(%[@"),
            ("|[*([", "|[(["),
        ]
    )
    assert fit.success
    assert op.apply("\\(*[#", fit.params) == "\\([#"


def test_operator_template_op_fit() -> None:
    op = OperatorTemplateOp()
    fit = op.fit(
        [
            ("%|*\"|", "%|\"|"),
            ("\\(*[^", "\\([^"),
            ("(%+[@", "(%[@"),
            ("|[*([", "|[(["), 
            ("[^-[(", "-^"),
        ]
    )
    assert fit.success
    assert op.apply("\\(*[#", fit.params) == "\\([#"


def test_binary_equation_rule_op_fit() -> None:
    op = BinaryEquationRuleOp()
    fit = op.fit([("96$54", "5184"), ("50$41", "2050"), ("51$95", "4845")])
    assert fit.success
    assert op.apply("59$49", fit.params) == "2891"


def test_binary_equation_rule_op_handles_operator_feature_intersection() -> None:
    op = BinaryEquationRuleOp()
    fit = op.fit([("64-65", "201"), ("28-68", "861"), ("82/15", "8241")])
    assert fit.success is False


def test_binary_ops_fit_and_apply() -> None:
    affine = BinaryAffineTransformOp()
    examples = []
    for value in range(16):
        source = format(value, "04b")
        x0, x1, x2, x3 = [int(bit) for bit in source]
        target = f"{x0 ^ x1}{x1}{x2 ^ 1}{x3}"
        examples.append((source, target))
    affine_fit = affine.fit(examples)
    assert affine_fit.success
    assert affine.apply("1110", affine_fit.params) == "0100"

    rotate = BinaryRotateLeftOp()
    rotate_fit = rotate.fit([("11010010", "01001011"), ("10100011", "10001110")])
    assert rotate_fit.success
    assert rotate.apply("00011110", rotate_fit.params) == "01111000"

    xor_mask = BinaryXorMaskOp()
    xor_fit = xor_mask.fit([("10100101", "01010101"), ("00001111", "11111111")])
    assert xor_fit.success
    assert xor_mask.apply("11110000", xor_fit.params) == "00000000"

    permutation = BinaryPermutationOp()
    permutation_fit = permutation.fit([("1000", "0100"), ("0100", "0001"), ("0010", "1000"), ("0001", "0010")])
    assert permutation_fit.success
    assert permutation.apply("0011", permutation_fit.params) == "1010"

    nibble_map = BinaryNibbleMapOp()
    nibble_fit = nibble_map.fit([("10100011", "01011100"), ("11001111", "00110000")])
    assert nibble_fit.success
    assert nibble_map.apply("10101111", nibble_fit.params) == "01010000"
