"""Тесты для frontmatter_builder."""

from src.preprocess.frontmatter_builder import article_id_from_filename, arxiv_id_from_filename


def test_article_id_underscore():
    assert article_id_from_filename("2403_05798.md") == "arxiv_2403_05798"


def test_article_id_dot_normalized():
    """Точка в имени файла нормализуется в подчёркивание."""
    assert article_id_from_filename("2603.29756.md") == "arxiv_2603_29756"


def test_article_id_no_extension():
    assert article_id_from_filename("2403_05798") == "arxiv_2403_05798"


def test_arxiv_id_from_underscore():
    assert arxiv_id_from_filename("2403_05798.md") == "2403.05798"


def test_arxiv_id_from_dot():
    assert arxiv_id_from_filename("2603.29756.md") == "2603.29756"


def test_arxiv_id_no_match():
    assert arxiv_id_from_filename("readme.md") is None
