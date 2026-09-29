"""Тесты для structure_parser."""

from src.chunker.structure_parser import MDBlock, parse_blocks


SAMPLE = """\
---
title: Test
---

# Title

## Abstract

Some abstract text.

## 1. Introduction

Intro paragraph one.

Intro paragraph two with $x + y$.

### 1.1. Sub-section

Sub text.

$$E = mc^2 \\tag{1}$$

Figure 1. This is a caption of the figure.

| A | B | C |
|---|---|---|
| 1 | 2 | 3 |

Table 1: Results
| X | Y |
|---|---|
| a | b |
"""


def test_parse_headings():
    blocks = parse_blocks(SAMPLE)
    headings = [b for b in blocks if b.type == "heading"]
    assert len(headings) == 4
    assert headings[0].heading_level == 1
    assert headings[1].content == "## Abstract"
    assert headings[2].heading_number == "1"
    assert headings[3].heading_number == "1.1"


def test_parse_paragraphs():
    blocks = parse_blocks(SAMPLE)
    paragraphs = [b for b in blocks if b.type == "paragraph"]
    assert any("abstract text" in p.content for p in paragraphs)
    assert any("$x + y$" in p.content for p in paragraphs)


def test_parse_display_math():
    blocks = parse_blocks(SAMPLE)
    math_blocks = [b for b in blocks if b.type == "display_math"]
    assert len(math_blocks) == 1
    assert "E = mc^2" in math_blocks[0].content


def test_parse_figure_caption():
    blocks = parse_blocks(SAMPLE)
    figs = [b for b in blocks if b.type == "figure_caption"]
    assert len(figs) == 1
    assert "Figure 1" in figs[0].content


def test_parse_tables():
    blocks = parse_blocks(SAMPLE)
    tables = [b for b in blocks if b.type == "table"]
    assert len(tables) >= 1
    # Хотя бы одна таблица содержит данные
    assert any("| 1 |" in t.content or "| a |" in t.content for t in tables)


def test_parse_table_with_caption():
    blocks = parse_blocks(SAMPLE)
    tables = [b for b in blocks if b.type == "table"]
    # Таблица с caption "Table 1: Results" должна быть одним блоком
    table_with_cap = [t for t in tables if "Table 1" in t.content]
    assert len(table_with_cap) == 1
    assert "| a | b |" in table_with_cap[0].content


def test_frontmatter_skipped():
    blocks = parse_blocks(SAMPLE)
    # Frontmatter не должен попасть в блоки
    assert not any("title: Test" in b.content for b in blocks)


def test_block_order_preserved():
    blocks = parse_blocks(SAMPLE)
    types = [b.type for b in blocks]
    # Первый — heading (# Title), последний — table
    assert types[0] == "heading"
    assert types[-1] == "table"
