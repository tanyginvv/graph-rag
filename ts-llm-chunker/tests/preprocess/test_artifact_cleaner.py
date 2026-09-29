"""Тесты для artifact_cleaner."""

from src.preprocess.artifact_cleaner import (
    clean_artifacts,
    remove_affiliation_tails,
    remove_empty_table_rows,
    remove_image_placeholders,
    remove_inline_emails,
)


def test_remove_image_placeholders():
    text = "Some text.\n\n![](_page_1_Figure_1.jpeg)\n\nMore text."
    result = remove_image_placeholders(text)
    assert "![](" not in result
    assert "Some text." in result
    assert "More text." in result


def test_remove_image_preserves_figure_captions():
    text = "![](_page_3_Figure_1.jpeg)\n\nFigure 2. The model architecture."
    result = remove_image_placeholders(text)
    assert "Figure 2. The model architecture." in result


def test_remove_empty_table_rows():
    text = "| A | B |\n|---|---|\n| | |\n| 1 | 2 |"
    result = remove_empty_table_rows(text)
    assert "| | |" not in result
    assert "| A | B |" in result
    assert "| 1 | 2 |" in result


def test_remove_affiliation_tail():
    paragraph = (
        "These realities present significant challenges for "
        "School of Computing, University of Connecticut, Storrs, USA "
        "Department of Machine Learning Research, Morgan Stanley, New York, USA. "
        "Correspondence to: Yuriy Nevmyvaka <yuriy.nevmyvaka@morganstanley.com>, "
        "Dongjin Song <dongjin.song@uconn.edu>."
    )
    result = remove_affiliation_tails(paragraph)
    assert "Correspondence" not in result
    assert "School of Computing" not in result
    assert "These realities present significant challenges for" in result


def test_remove_inline_emails():
    text = "Contact: <alice@example.com> or <bob@example.org>."
    result = remove_inline_emails(text)
    assert "<alice@example.com>" not in result
    assert "<bob@example.org>" not in result
    assert "Contact:" in result


def test_clean_artifacts_full():
    text = (
        "Text before.\n\n"
        "![](_page_1_Figure_1.jpeg)\n\n"
        "Figure 1. Caption.\n\n"
        "| | |\n"
        "End of paragraph School of X, University Y. "
        "Correspondence to: Name <a@b.com>."
    )
    result = clean_artifacts(text, remove_images=True, remove_emails=True)
    assert "![](" not in result
    assert "Figure 1. Caption." in result
    assert "| | |" not in result
    assert "Correspondence" not in result
    assert "<a@b.com>" not in result
