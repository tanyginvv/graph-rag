"""Тесты для math_normalizer."""

from src.preprocess.math_normalizer import (
    join_broken_paragraphs,
    merge_broken_display_math,
    normalize_math,
)


def test_merge_continuation_formula():
    text = (
        "some text $$Z = [e'_1; \\dots; e'_K]$$\n"
        "\n"
        "$$= [e'_{\\text{top-}K}; \\mathcal{P}] \\qquad (3)$$\n"
        "\n"
        "which will serve as the input."
    )
    result = merge_broken_display_math(text)
    assert "$$= " not in result
    # Формула объединена
    assert "Z = [e'_1" in result
    assert "e'_{\\text{top-}K}" in result
    # Следующий абзац не тронут
    assert "which will serve" in result


def test_merge_standalone_equation_number():
    text = (
        "$$X'_{i,t} = \\gamma_T \\left( X_{i,t} \\right) + \\beta_T$$\n"
        "\n"
        "(1)\n"
        "\n"
        "where $\\gamma_T$ is a parameter."
    )
    result = merge_broken_display_math(text)
    assert "(1)" not in result.split("\n")[0] or "\\tag{1}" in result
    assert "\\tag{1}" in result
    assert "where $\\gamma_T$" in result


def test_no_false_merge():
    """Две независимые display-формулы не склеиваются."""
    text = (
        "$$a + b = c$$\n"
        "\n"
        "$$d + e = f$$\n"
    )
    result = merge_broken_display_math(text)
    assert result.count("$$") == 4  # два отдельных display-блока


def test_join_broken_paragraphs():
    text = (
        "This is the end of a sentence\n"
        "\n"
        "which continues after a false break.\n"
    )
    result = join_broken_paragraphs(text)
    assert "sentence which continues" in result


def test_no_false_join_after_heading():
    text = (
        "## 3.1. Problem Statement\n"
        "\n"
        "we first formalize the problem.\n"
    )
    result = join_broken_paragraphs(text)
    # Заголовок не склеивается с параграфом
    assert "## 3.1. Problem Statement" in result
    lines = [l for l in result.split("\n") if l.strip()]
    assert len(lines) == 2


def test_normalize_math_combined():
    text = (
        "text $$Z = [a; b]$$\n"
        "\n"
        "$$= [c; d] \\qquad (3)$$\n"
        "\n"
        "which serves as input.\n"
    )
    result = normalize_math(text)
    # Формула объединена
    assert "$$= " not in result
    assert "Z = [a; b]" in result
    assert "[c; d]" in result
