import json

import pytest

from vector_db_rag.chunk_adapter import adapt_basic_chunks, write_adapted_corpus
from vector_db_rag.chunks import load_chunk_file


def _source_files(tmp_path):
    chunks = tmp_path / "basic.json"
    chunks.write_text(
        json.dumps(
            {
                "chunker_version": "0.2.0",
                "config_hash": "abc123",
                "chunks": [
                    {
                        "chunk_id": "paper_chunk_000",
                        "article_id": "paper",
                        "section": "Methodology",
                        "index": 0,
                        "text": "## Methodology\n\nThe model uses text prototypes for forecasting.",
                        "token_count": 14,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    articles = tmp_path / "articles"
    articles.mkdir()
    return chunks, articles


def test_basic_chunker_export_becomes_valid_graph_corpus(tmp_path):
    chunks, articles = _source_files(tmp_path)
    (articles / "paper.md").write_text(
        "---\narticle_id: paper\ntitle: Paper Title\nauthors:\n  - Researcher\n"
        "year: 2024\narxiv_id: '2403.05798'\n---\n# Paper Title\n",
        encoding="utf-8",
    )
    converted = adapt_basic_chunks(chunks, articles)
    output = tmp_path / "converted.json"
    write_adapted_corpus(converted, output)
    loaded = load_chunk_file(output)
    assert loaded.chunks[0].article_meta.title == "Paper Title"
    assert loaded.chunks[0].summary.startswith("Methodology The model")
    assert loaded.chunks[0].question_types == []
    assert json.loads(output.read_text(encoding="utf-8"))["provenance"]["semantic_enrichment"] is False


def test_missing_article_metadata_fails_clearly(tmp_path):
    chunks, articles = _source_files(tmp_path)
    with pytest.raises(ValueError, match="Missing Markdown metadata"):
        adapt_basic_chunks(chunks, articles)
