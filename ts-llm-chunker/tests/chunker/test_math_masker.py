"""Тесты для math_masker."""

from src.chunker.math_masker import mask_formulas, unmask_formulas


def test_mask_display_math():
    text = "Before $$E = mc^2$$ after."
    masked, mapping = mask_formulas(text)
    assert "$$" not in masked
    assert "[DISPLAY_MATH_000]" in masked
    assert "Before" in masked
    assert "after." in masked
    assert mapping["[DISPLAY_MATH_000]"] == "$$E = mc^2$$"


def test_mask_inline_math():
    text = "Let $x \\in \\mathbb{R}$ be a variable."
    masked, mapping = mask_formulas(text)
    assert "$" not in masked or masked.count("$") == 0
    assert "[MATH_000]" in masked
    assert mapping["[MATH_000]"] == "$x \\in \\mathbb{R}$"


def test_mask_multiple():
    text = "Given $a$ and $b$, then $$a + b = c$$."
    masked, mapping = mask_formulas(text)
    assert "[MATH_000]" in masked
    assert "[MATH_001]" in masked
    assert "[DISPLAY_MATH_000]" in masked
    assert len(mapping) == 3


def test_unmask_roundtrip():
    original = "Let $x$ be given. Then $$f(x) = x^2$$."
    masked, mapping = mask_formulas(original)
    restored = unmask_formulas(masked, mapping)
    assert restored == original


def test_no_false_positive_dollar_amount():
    text = "The cost is $100 per unit."
    masked, mapping = mask_formulas(text)
    # $100 не должен быть замаскирован
    assert "$100" in masked
    assert len(mapping) == 0


def test_empty_mapping_on_no_math():
    text = "No math here."
    masked, mapping = mask_formulas(text)
    assert masked == text
    assert len(mapping) == 0
