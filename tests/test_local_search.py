from pathlib import Path

import pytest

from vector_db_rag.chunks import load_chunk_file
from vector_db_rag.local_search import search_local
from vector_db_rag.sparse import tokenize


CORPUS = Path(__file__).resolve().parents[1] / "chunks 2" / "docs" / "dic_chunks.json"


def test_checked_in_corpus_has_unique_chunks_and_twelve_articles():
    corpus = load_chunk_file(CORPUS)
    assert len(corpus.chunks) == 141
    assert len({chunk.article_id for chunk in corpus.chunks}) == 12


def test_local_search_returns_verifiable_source():
    corpus = load_chunk_file(CORPUS)
    results = search_local(corpus, "Time-LLM reprogramming text prototypes", limit=3)
    assert results
    assert all(result["article_id"] == "time-llm" for result in results)
    source_ids = {chunk.chunk_id for chunk in corpus.chunks}
    assert all(result["chunk_id"] in source_ids for result in results)
    assert all(result["text_excerpt"] for result in results)


def test_named_method_query_ranks_its_article_first():
    corpus = load_chunk_file(CORPUS)
    results = search_local(corpus, "How does Time-LLM use text prototypes?", limit=3)
    assert results[0]["article_id"] == "time-llm"


def test_article_filter_and_empty_results():
    corpus = load_chunk_file(CORPUS)
    results = search_local(corpus, "reprogramming", article_id="vitro")
    assert all(result["article_id"] == "vitro" for result in results)
    assert search_local(corpus, "reprogramming", article_id="unknown") == []


def test_russian_query_and_bad_limit():
    corpus = load_chunk_file(CORPUS)
    assert tokenize("прогнозирование временных рядов") == [
        "прогнозирование", "временных", "рядов"
    ]
    assert search_local(corpus, "временных рядов", limit=2)
    with pytest.raises(ValueError, match="limit must be positive"):
        search_local(corpus, "Time-LLM", limit=0)
