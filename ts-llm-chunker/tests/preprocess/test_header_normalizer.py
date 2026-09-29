"""Тесты для header_normalizer."""

from src.preprocess.header_normalizer import (
    normalize_headers,
    remove_author_line,
    split_glued_headings,
)


def test_split_glued_heading():
    text = "Dongjin Song # Abstract"
    result = split_glued_headings(text)
    assert "Dongjin Song" in result
    assert "# Abstract" in result
    # Они на разных строках
    lines = result.split("\n")
    assert any("# Abstract" == l.strip() for l in lines)


def test_no_false_positive_on_normal_heading():
    text = "## 3. Methodology"
    result = split_glued_headings(text)
    assert result == text


def test_remove_author_line():
    text = (
        "# S2IP-LLM: Semantic Space Informed Prompt Learning\n"
        "\n"
        "Zijie Pan  Yushan Jiang  Sahil Garg  Anderson Schneider\n"
        "\n"
        "## Abstract\n"
    )
    result = remove_author_line(text)
    assert "Zijie Pan" not in result
    assert "## Abstract" in result
    assert "# S2IP-LLM" in result


def test_remove_author_line_no_false_positive():
    """Обычный абзац после заголовка не удаляется."""
    text = (
        "# Introduction\n"
        "\n"
        "This paper presents a novel approach.\n"
    )
    result = remove_author_line(text)
    assert "This paper presents" in result


def test_normalize_headers_combined():
    text = (
        "Dongjin Song # Abstract\n"
        "\n"
        "Text here.\n"
    )
    result = normalize_headers(text)
    assert "# Abstract" in result
    assert "Dongjin Song" in result
    lines = result.split("\n")
    heading_line = [l for l in lines if "# Abstract" in l]
    assert len(heading_line) == 1
    assert heading_line[0].strip() == "# Abstract"
